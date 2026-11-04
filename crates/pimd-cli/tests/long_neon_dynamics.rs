//! Long validation: RPMD and TRPMD of liquid neon against i-PI + LAMMPS (MILESTONE_7 required
//! test 12; SPEC Y-1 … Y-5, T-3, T-4, P-3). Feature `long-validation`, run in release by
//! `CHECK_LONG=1 ./scripts/check.sh`.
//!
//! Reference: `data/ipi_neon_dynamics.json`, written by `data/gen_ipi_neon_dynamics.py` (i-PI
//! 3.3.1 + LAMMPS `fix ipi`, same segment protocol, 480 segments per method from 6 independent
//! chains; versions, seeds and wall time recorded in the file). System (read from the file): the
//! M4 liquid, N = 108 neon atoms in a 14.1 Å cube, LJ ε = 35.6 K, σ = 2.749 Å, r_c = 2.2σ shifted,
//! skin 0.3σ, T = 30 K, P = 16, dt = 2 fs, PILE-L τ_0 = 100 fs; 5 000 equilibration steps, then
//! cycles of 250 PILE-L steps + Y-2 redraw + 500-step segment sampled every 2 steps (251 lags,
//! τ up to 1 ps).
//!
//! This side: `pimd_cli::run` with a `[dynamics]` section, as a chain of 120 legs of 4 cycles
//! (3 000 steps; `run`, then `run_from_checkpoint` with n_steps raised each time; restarts are
//! bitwise, tests/dynamics_run.rs), so the chain is one uninterrupted run of 480 segments and the
//! cumulative correlations after each leg give each leg's mean by differencing
//! (n_ℓ M_ℓ − n_{ℓ−1} M_{ℓ−1})/(n_ℓ − n_{ℓ−1}).
//!
//! Error bars (WORKFLOW tolerance policy, σ from segments): C_vv: σ_1 = the run's Y-4 σ over
//! single segments (`c_vv_sem`), σ_4 from the 120 leg means (blocks of 4 segments); asserted
//! σ_4 ≤ 1.5 σ_1 at every lag; σ = max. D(τ): σ from the 120 leg values of the Green–Kubo
//! integral of each leg's mean C_vv (D is linear in C_vv) and from 30 means of 4 legs, the larger.
//! i-PI side: the same procedure on its per-segment values (see the generator). Tolerance:
//! 5 σ_combined at every lag for C_vv and at τ_max for D; C_vv(0) within 5σ of 3/(βm) (exact after
//! the redraw, Y-2/Y-4). Bias allowance 0: both codes integrate the same splitting at the same dt
//! (as the M4 neon comparison, long_neon.rs).
//!
//! Results with the throwaway reference implementation (release, 4-thread pool, 10.6 min):
//! TRPMD C_vv worst lag 250 at 2.41 σ_c, D(1 ps) = 1.4273e-5 ± 1.7e-7 vs i-PI 1.4428e-5 ± 1.6e-7
//! bohr²/(ħ/Eh) (0.66 σ_c), C_vv(0) 0.05σ from 3/(βm); RPMD worst lag 250 at 0.95 σ_c,
//! D = 1.4057e-5 ± 1.8e-7 vs 1.4070e-5 ± 1.7e-7 (0.05 σ_c), C_vv(0) 1.89σ below 3/(βm) (i-PI:
//! 1.5σ below; C_seg(0) averages every sample of the segment, so it carries the NVE time-step
//! bias of the centroid kinetic energy, common to both codes).
#![cfg(feature = "long-validation")]
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

mod common;

use std::path::PathBuf;

use common::{parse, pool, tmp};
use lammps::Json;
use pimd_cli::{run, run_from_checkpoint};

const IPI_JSON: &str = include_str!("data/ipi_neon_dynamics.json");
const LEGS: usize = 120;
const CYCLES_PER_LEG: usize = 4;

fn ipi() -> Json {
    Json::parse(IPI_JSON)
}

fn neon_toml(j: &Json, method: &str, n_steps: usize, seed: u64) -> String {
    let s = j.get("system");
    let p = j.get("protocol");
    let cycle = (p.get("decorrelation_steps").num() + p.get("segment_steps").num()) as usize;
    let method_lines = match method {
        "rpmd" => "method = \"rpmd\"".to_string(),
        _ => format!(
            "method = \"trpmd\"\ntrpmd_lambda = {}",
            s.get("trpmd_lambda").num()
        ),
    };
    format!(
        r#"
[system]
n_atoms = {}
mass_amu = {}
box_angstrom = {}
[potential]
kind = "lj"
epsilon_kelvin = {}
sigma_angstrom = {}
cutoff_sigma = {}
skin_sigma = {}
shift = true
[sampling]
temperature_kelvin = {}
n_beads = {}
dt_fs = {}
tau0_fs = {}
n_equil = {}
n_steps = {n_steps}
stride = 50
seed = {seed}
[output]
checkpoint_every = {}
[dynamics]
{method_lines}
decorrelation_steps = {}
segment_steps = {}
corr_stride = {}
n_lags = {}
"#,
        s.get("n_atoms").num() as usize,
        s.get("mass_amu").num(),
        s.get("box_angstrom").num(),
        s.get("epsilon_kelvin").num(),
        s.get("sigma_angstrom").num(),
        s.get("cutoff_sigma").num(),
        s.get("skin_sigma").num(),
        s.get("temperature_kelvin").num(),
        s.get("n_beads").num() as usize,
        s.get("dt_fs").num(),
        s.get("tau0_fs").num(),
        p.get("n_equil").num() as usize,
        CYCLES_PER_LEG * cycle,
        p.get("decorrelation_steps").num() as usize,
        p.get("segment_steps").num() as usize,
        p.get("corr_stride").num() as usize,
        p.get("n_lags").num() as usize,
    )
}

/// σ of the mean of `blocks` grouped `g` at a time, per lag.
fn group_sigma(blocks: &[Vec<f64>], g: usize) -> Vec<f64> {
    let n = blocks.len() / g;
    (0..blocks[0].len())
        .map(|k| {
            let means: Vec<f64> = (0..n)
                .map(|b| (0..g).map(|i| blocks[b * g + i][k]).sum::<f64>() / g as f64)
                .collect();
            let m = means.iter().sum::<f64>() / n as f64;
            let v = means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
            (v / n as f64).sqrt()
        })
        .collect()
}

struct Chain {
    c_vv: Vec<f64>,
    c_vv_sigma: Vec<f64>,
    d: Vec<f64>,
    d_sigma: Vec<f64>,
    n_segments: usize,
}

/// The 120-leg chain for `method` (4-thread pool).
fn chain(method: &str, seed: u64) -> Chain {
    let j = ipi();
    let p = j.get("protocol");
    let cycle = (p.get("decorrelation_steps").num() + p.get("segment_steps").num()) as usize;
    let leg_steps = CYCLES_PER_LEG * cycle;
    let root = tmp();
    let mut prev: Option<(PathBuf, Vec<f64>)> = None;
    let mut leg_c = Vec::new();
    let mut leg_d = Vec::new();
    let mut last = None;
    pool(4).install(|| {
        for leg in 1..=LEGS {
            let cfg = parse(&neon_toml(&j, method, leg * leg_steps, seed));
            let dir = root.path().join(format!("leg{leg}"));
            let s = match &prev {
                None => run(&cfg, &dir).unwrap(),
                Some((d, _)) => run_from_checkpoint(&cfg, &d.join("checkpoint.bin"), &dir).unwrap(),
            };
            let c = s.correlations.unwrap();
            assert_eq!(c.n_segments, leg * CYCLES_PER_LEG);
            let k = CYCLES_PER_LEG as f64;
            let n_now = c.n_segments as f64;
            let block: Vec<f64> = match &prev {
                None => c.c_vv.clone(),
                Some((_, m_old)) => c
                    .c_vv
                    .iter()
                    .zip(m_old)
                    .map(|(mn, mo)| (n_now * mn - (n_now - k) * mo) / k)
                    .collect(),
            };
            // Y-5 by hand: D(τ_n) = (1/3) trapezoid of the leg's C_vv.
            let mut d = vec![0.0; block.len()];
            for t in 1..block.len() {
                d[t] = d[t - 1] + 0.5 * (block[t - 1] + block[t]) * c.dt_lag / 3.0;
            }
            leg_d.push(d);
            leg_c.push(block);
            prev = Some((dir, c.c_vv.clone()));
            last = Some(c);
        }
    });
    let c = last.unwrap();
    let s4 = group_sigma(&leg_c, 1);
    for k in 0..s4.len() {
        assert!(
            s4[k] <= 1.5 * c.c_vv_sem[k],
            "{method} C_vv lag {k}: σ_4 {:e} > 1.5 σ_1 {:e}",
            s4[k],
            c.c_vv_sem[k]
        );
    }
    let (d1, d4) = (group_sigma(&leg_d, 1), group_sigma(&leg_d, 4));
    Chain {
        c_vv_sigma: c.c_vv_sem.iter().zip(&s4).map(|(a, b)| a.max(*b)).collect(),
        c_vv: c.c_vv,
        d: c.diffusion,
        d_sigma: d1.iter().zip(&d4).map(|(a, b)| a.max(*b)).collect(),
        n_segments: c.n_segments,
    }
}

/// Y-1 … Y-5, T-3, T-4, P-3: for RPMD and TRPMD (λ = 0.5) of liquid neon, C_vv(τ) agrees with
/// i-PI + LAMMPS within 5 σ_combined at every lag, D(τ_max) within 5 σ_combined, and
/// C_vv(0) = 3/(βm) within 5σ.
#[test]
fn neon_dynamics_matches_ipi() {
    let j = ipi();
    let beta = j.get("derived_atomic_units").get("beta").num();
    let mass = j.get("derived_atomic_units").get("mass_me").num();
    for (method, seed) in [("trpmd", 7001), ("rpmd", 7002)] {
        let ch = chain(method, seed);
        let r = j.get(method);
        let (m, s) = (
            r.get("c_vv").get("mean").nums(),
            r.get("c_vv").get("sigma").nums(),
        );
        assert_eq!(m.len(), ch.c_vv.len());
        let mut worst = (0.0f64, 0usize);
        for k in 0..m.len() {
            let sc = (ch.c_vv_sigma[k].powi(2) + s[k].powi(2)).sqrt();
            let z = (ch.c_vv[k] - m[k]).abs() / sc;
            if z > worst.0 {
                worst = (z, k);
            }
            assert!(
                z <= 5.0,
                "{method} C_vv lag {k}: {:e} ± {:e} vs i-PI {:e} ± {:e}",
                ch.c_vv[k],
                ch.c_vv_sigma[k],
                m[k],
                s[k]
            );
        }
        let (dm, ds) = (
            r.get("diffusion").get("mean").nums(),
            r.get("diffusion").get("sigma").nums(),
        );
        let last = dm.len() - 1;
        let sc = (ch.d_sigma[last].powi(2) + ds[last].powi(2)).sqrt();
        let zd = (ch.d[last] - dm[last]).abs() / sc;
        let want0 = 3.0 / (beta * mass);
        let z0 = (ch.c_vv[0] - want0).abs() / ch.c_vv_sigma[0];
        eprintln!(
            "{method}: {} segments; C_vv worst lag {} at {:.2} σ_c; D(τ_max) = {:.5e} ± {:.1e} \
             vs i-PI {:.5e} ± {:.1e} ({zd:.2} σ_c); C_vv(0) {:.6e} vs 3/(βm) {want0:.6e} ({z0:.2} σ)",
            ch.n_segments,
            worst.1,
            worst.0,
            ch.d[last],
            ch.d_sigma[last],
            dm[last],
            ds[last],
            ch.c_vv[0]
        );
        assert!(zd <= 5.0, "{method} D(τ_max): {zd:.2} σ_c");
        assert!(z0 <= 5.0, "{method} C_vv(0): {z0:.2} σ");
    }
}
