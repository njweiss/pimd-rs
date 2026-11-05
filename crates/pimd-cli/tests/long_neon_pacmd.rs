//! Long validation: PA-CMD of liquid neon against i-PI + LAMMPS (MILESTONE_7 M7b required test 10;
//! SPEC Y-6, Y-7, Y-2 amended, Y-1 … Y-5, T-1, T-2, T-4, P-3, E-2, E-4). Feature
//! `long-validation`, run in release by `CHECK_LONG=1 ./scripts/check.sh`.
//!
//! Reference: `data/ipi_neon_cmd.json`, written by `data/gen_ipi_neon_cmd.py` (i-PI 3.3.1
//! `normal_modes` mode `pa-cmd` + LAMMPS `fix ipi`, same segment protocol, 480 segments from 6
//! chains; versions, seeds and wall time in the file). System (read from the file): the M4/M7a
//! liquid, N = 108 neon atoms in a 14.1 Å cube, LJ ε = 35.6 K, σ = 2.749 Å, r_c = 2.2σ shifted,
//! skin 0.3σ, T = 30 K, P = 16, dt = 2 fs, PILE-L τ_0 = 100 fs; PA-CMD at
//! Ω_h = 16^{16/15} k_B T ≈ 401.3 cm⁻¹ (Ω·dt ≈ 0.15; the file's `cmd_omega_cm`), λ = 1;
//! 5 000 equilibration steps, then cycles of 250 PILE-L steps + redraw + 500-step segment sampled
//! every 2 steps (251 lags, τ up to 1 ps).
//!
//! This side: `pimd_cli::run` with `[dynamics] method = "pa-cmd"`, as a chain of 120 legs of 4
//! cycles (3 000 steps; restarts are bitwise, tests/pacmd_run.rs), so one uninterrupted run of 480
//! segments; each leg's mean C_vv by differencing the cumulative means (as long_neon_dynamics.rs).
//!
//! Error bars (WORKFLOW tolerance policy, σ from segments): C_vv: σ_1 = the run's Y-4 σ, σ_4 from
//! the 120 leg means (blocks of 4 segments), σ_4 ≤ 1.5 σ_1 asserted at every lag, σ = max. D(τ):
//! σ from the 120 leg values and from 30 means of 4 legs, the larger. ⟨E_vir⟩ per atom from
//! observables.csv (every 50 steps, all production steps): blocks of one leg (60 samples, 6 ps),
//! checked against 4-leg blocks (`common::mean_and_sigma`), σ the larger. Tolerances:
//! 5 σ_combined at every lag for C_vv and at τ_max for D; C_vv(0) within 5σ of 3/(βm);
//! ⟨E_vir⟩ within 5 σ_combined of the M4 i-PI value in the locked `data/ipi_neon.json` (Y-6: the
//! dynamical masses leave the configurational distribution unchanged, so the statics are the M4
//! PIMD ones). Bias allowance 0: C_vv and D come from the same splitting at the same dt and Ω on
//! both sides; for ⟨E_vir⟩ the M4 measurement put the dt bias of this system (2 fs vs 1 fs) at
//! 0.08 ± 0.53 K/atom, unresolved and below a third of the tolerance, and the dynamical masses
//! here lower the internal-mode frequencies from up to 2ω_P = 667 cm⁻¹ to 401 cm⁻¹.
//!
//! Results with the throwaway reference implementation (release, 4-thread pool, 4.8 min; initial
//! momenta from the amended Y-2 draw, Q32): C_vv worst lag 250 at 0.91 σ_c; D(1 ps) =
//! 1.4369e-5 ± 1.5e-7 vs i-PI 1.4499e-5 ± 1.8e-7 bohr²/(ħ/Eh) (0.56 σ_c); C_vv(0) 0.07 σ from
//! 3/(βm); ⟨E_vir⟩ = −96.61 ± 0.13 K/atom vs M4 i-PI −97.19 ± 0.18 K/atom (2.59 σ_c).
#![cfg(feature = "long-validation")]
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

mod common;

use std::path::PathBuf;

use common::{mean_and_sigma, observables, parse, pool, tmp};
use lammps::Json;
use pimd_cli::{run, run_from_checkpoint};

const IPI_JSON: &str = include_str!("data/ipi_neon_cmd.json");
const M4_JSON: &str = include_str!("data/ipi_neon.json");
const LEGS: usize = 120;
const CYCLES_PER_LEG: usize = 4;
const STRIDE: usize = 50;
const SEED: u64 = 7101;

fn neon_toml(j: &Json, n_steps: usize) -> String {
    let s = j.get("system");
    let p = j.get("protocol");
    let cycle = (p.get("decorrelation_steps").num() + p.get("segment_steps").num()) as usize;
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
stride = {STRIDE}
seed = {SEED}
[output]
checkpoint_every = {}
[dynamics]
method = "pa-cmd"
cmd_omega_cm = {}
cmd_lambda = {}
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
        s.get("cmd_omega_cm").num(),
        s.get("cmd_lambda").num(),
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

const K_B: f64 = 3.1668115634556e-6; // Eh/K (U-2), for printing in kelvin only

/// Y-6, Y-7, Y-1 … Y-5, T-4 with Y-6, P-3, E-4: PA-CMD of liquid neon (Ω_h, λ = 1): C_vv(τ) within
/// 5 σ_combined of i-PI + LAMMPS `pa-cmd` at every lag, D(τ_max) within 5 σ_combined,
/// C_vv(0) = 3/(βm) within 5σ, and ⟨E_vir⟩ of the run within 5 σ_combined of the M4 i-PI PIMD value.
#[test]
fn neon_pacmd_matches_ipi() {
    let j = Json::parse(IPI_JSON);
    let beta = j.get("derived_atomic_units").get("beta").num();
    let mass = j.get("derived_atomic_units").get("mass_me").num();
    let p = j.get("protocol");
    let cycle = (p.get("decorrelation_steps").num() + p.get("segment_steps").num()) as usize;
    let leg_steps = CYCLES_PER_LEG * cycle;
    let root = tmp();
    let mut prev: Option<(PathBuf, Vec<f64>)> = None;
    let (mut leg_c, mut leg_d, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    let mut last = None;
    pool(4).install(|| {
        for leg in 1..=LEGS {
            let cfg = parse(&neon_toml(&j, leg * leg_steps));
            let dir = root.path().join(format!("leg{leg}"));
            let s = match &prev {
                None => run(&cfg, &dir).unwrap(),
                Some((d, _)) => run_from_checkpoint(&cfg, &d.join("checkpoint.bin"), &dir).unwrap(),
            };
            rows.extend(observables(&dir));
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
            if leg > 1 {
                let _ = std::fs::remove_dir_all(root.path().join(format!("leg{}", leg - 1)));
            }
        }
    });
    let c = last.unwrap();
    let s4 = group_sigma(&leg_c, 1);
    for k in 0..s4.len() {
        assert!(
            s4[k] <= 1.5 * c.c_vv_sem[k],
            "C_vv lag {k}: σ_4 {:e} > 1.5 σ_1 {:e}",
            s4[k],
            c.c_vv_sem[k]
        );
    }
    let sig: Vec<f64> = c.c_vv_sem.iter().zip(&s4).map(|(a, b)| a.max(*b)).collect();
    let (d1, d4) = (group_sigma(&leg_d, 1), group_sigma(&leg_d, 4));
    let r = j.get("pa_cmd");
    let (m, s) = (
        r.get("c_vv").get("mean").nums(),
        r.get("c_vv").get("sigma").nums(),
    );
    assert_eq!(m.len(), c.c_vv.len());
    let mut worst = (0.0f64, 0usize);
    for k in 0..m.len() {
        let sc = (sig[k].powi(2) + s[k].powi(2)).sqrt();
        let z = (c.c_vv[k] - m[k]).abs() / sc;
        if z > worst.0 {
            worst = (z, k);
        }
        assert!(
            z <= 5.0,
            "C_vv lag {k}: {:e} ± {:e} vs i-PI {:e} ± {:e}",
            c.c_vv[k],
            sig[k],
            m[k],
            s[k]
        );
    }
    let (dm, ds) = (
        r.get("diffusion").get("mean").nums(),
        r.get("diffusion").get("sigma").nums(),
    );
    let last = dm.len() - 1;
    let d_sig = d1[last].max(d4[last]);
    let zd = (c.diffusion[last] - dm[last]).abs() / (d_sig.powi(2) + ds[last].powi(2)).sqrt();
    let want0 = 3.0 / (beta * mass);
    let z0 = (c.c_vv[0] - want0).abs() / sig[0];

    // ⟨E_vir⟩ per atom (observables.csv column 2) over every production step.
    assert_eq!(rows.len(), LEGS * leg_steps / STRIDE);
    let e: Vec<f64> = rows.iter().map(|r| r[2]).collect();
    let (em, es) = mean_and_sigma(&e, leg_steps / STRIDE, "E_vir");
    let m4 = Json::parse(M4_JSON);
    let (im, is) = (
        m4.get("e_vir").get("mean").num(),
        m4.get("e_vir").get("sigma").num(),
    );
    let ze = (em - im).abs() / (es * es + is * is).sqrt();
    eprintln!(
        "PA-CMD neon: {} segments; C_vv worst lag {} at {:.2} σ_c; D(τ_max) = {:.5e} ± {:.1e} vs \
         i-PI {:.5e} ± {:.1e} ({zd:.2} σ_c); C_vv(0) {:.6e} vs 3/(βm) {want0:.6e} ({z0:.2} σ); \
         ⟨E_vir⟩ = {:.3} ± {:.3} K/atom vs M4 i-PI {:.3} ± {:.3} K/atom ({ze:.2} σ_c)",
        c.n_segments,
        worst.1,
        worst.0,
        c.diffusion[last],
        d_sig,
        dm[last],
        ds[last],
        c.c_vv[0],
        em / K_B,
        es / K_B,
        im / K_B,
        is / K_B,
    );
    assert!(zd <= 5.0, "D(τ_max): {zd:.2} σ_c");
    assert!(z0 <= 5.0, "C_vv(0): {z0:.2} σ");
    assert!(ze <= 5.0, "⟨E_vir⟩: {ze:.2} σ_c");
}
