//! Long validation of liquid-neon PIMD (MILESTONE_4 "Long validation"; SPEC I-1, T-1, T-2,
//! E-1 … E-4, E-7, E-9 note, RP-2, RP-3, RP-6, P-3). Feature `long-validation`, run in release by
//! `CHECK_LONG=1 ./scripts/check.sh`.
//!
//! System (from `data/ipi_neon.json`, written by `data/gen_ipi_neon.py`): N = 108 neon atoms in a
//! 14.1 Å cube (ρ* = 0.80), LJ ε = 35.6 K, σ = 2.749 Å, r_c = 2.2σ shifted, skin 0.3σ, T = 30 K,
//! dt = 2 fs, PILE-L τ_0 = 100 fs, 5 000 equilibration steps, 200 000 production steps (400 ps),
//! recorded every 10 steps. Every engine run goes through `pimd_cli::run` as a chain of 80 legs of
//! 2 500 steps (`run`, then `run_from_checkpoint` with n_steps raised by 2 500 each time); restarts
//! are bitwise (tests/restart.rs), so the chain is one uninterrupted trajectory, and the
//! cumulative RDF after each leg gives the per-leg (block) g(r) by differencing.
//!
//! Error bars (WORKFLOW tolerance policy): energies from block averages of the recorded series
//! with blocks of 250 samples (one leg, 2 500 steps = 5 ps), checked against 1 000-sample blocks
//! (σ_4 ≤ 1.5 σ_1, `common::mean_and_sigma`), σ the larger; g(r) per bin from the 80 leg means and
//! from 20 means of 4 legs, σ the larger. Tolerances are 5σ_combined plus the stated bias
//! allowance.
//!
//! Discretisation bias: the engine and i-PI integrate with the same splitting (BAOAB, exact free
//! ring-polymer propagation, PILE-L with γ_k = 2ω_k; i-PI `splitting="baoab"`,
//! `pile_lambda = 1`) at the same dt, so their O(dt²) bias is common and the comparison
//! allowance is 0. The bias itself was measured with the reference implementation at dt = 2 fs and
//! dt/2 = 1 fs (P = 16, 192 ps of production each, σ from 19-ps blocks): ⟨E_vir⟩ = −96.88 ± 0.33
//! vs −96.80 ± 0.41 K/atom (Δ = 0.08 ± 0.53 K), ⟨K_vir⟩ = 55.899 ± 0.057 vs 55.880 ± 0.063 K/atom
//! (Δ = 0.02 ± 0.09 K), first g(r) peak 2.584 vs 2.574. The change is unresolved and below a third
//! of the energy tolerance (5σ_combined ≈ 1.5–2 K), so the quantum–classical and convergence checks
//! need no allowance either.
//!
//! Results with the throwaway reference implementation (release, this protocol), per atom:
//! P = 16 ⟨E_vir⟩ = −96.72 ± 0.21 K vs i-PI −97.19 ± 0.18 K (1.7 σ_c), ⟨E_prim⟩ = −96.70 ± 0.25 K
//! vs −97.29 ± 0.26 K (1.65 σ_c), g(r) worst bin 2.0 σ_c of 70; quantum − classical shift of
//! ⟨E_vir⟩ +15.96 K (58 σ_c), first-peak height 2.732 ± 0.007 (P = 1) vs 2.463 ± 0.004 (P = 16) at
//! r = 2.95 Å; ⟨E_vir⟩ = −99.00, −97.69, −96.72, −96.74 (± 0.19–0.22) K for P = 4, 8, 16, 32,
//! fit a = −36.6 ± 4.3 K; σ(K_vir) = 0.03–0.04 K vs σ(K_prim) = 0.10–0.17 K. Wall time 8 min
//! (12-core machine, 4-thread pools).
#![cfg(feature = "long-validation")]

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

mod common;

use std::path::PathBuf;
use std::sync::OnceLock;

use common::{mean_and_sigma, observables, parse, pool, tmp};
use lammps::Json;
use pimd_cli::{run, run_from_checkpoint};

const IPI_JSON: &str = include_str!("data/ipi_neon.json");

/// Legs of the chain and steps per leg: 80 × 2 500 = 200 000 production steps. (The potential
/// energy of this 108-atom liquid decorrelates slowly: block errors keep growing up to ~5-10 ps
/// blocks, measured with the reference implementation, so blocks are 5 ps and 20 ps.)
const LEGS: usize = 80;
const LEG_STEPS: usize = 2500;
const N_EQUIL: usize = 5000;
const STRIDE: usize = 10;
const SEED: u64 = 2026;

/// One finished chain: the recorded observables rows and the per-leg g(r) block means.
struct Chain {
    rows: Vec<Vec<f64>>,
    g_blocks: Vec<Vec<f64>>,
    r: Vec<f64>,
}

fn ipi() -> &'static Json {
    static J: OnceLock<Json> = OnceLock::new();
    J.get_or_init(|| Json::parse(IPI_JSON))
}

/// The engine config for `p` beads, built from the generator's `system` block, with production
/// length `n_steps`.
fn neon_toml(p: usize, n_steps: usize) -> String {
    let s = ipi().get("system");
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
n_beads = {p}
dt_fs = {}
tau0_fs = {}
n_equil = {N_EQUIL}
n_steps = {n_steps}
stride = {STRIDE}
seed = {SEED}
[output]
rdf_bins = {}
rdf_rmax_angstrom = {}
checkpoint_every = {LEG_STEPS}
"#,
        s.get("n_atoms").num() as usize,
        s.get("mass_amu").num(),
        s.get("box_angstrom").num(),
        s.get("epsilon_kelvin").num(),
        s.get("sigma_angstrom").num(),
        s.get("cutoff_sigma").num(),
        s.get("skin_sigma").num(),
        s.get("temperature_kelvin").num(),
        s.get("dt_fs").num(),
        s.get("tau0_fs").num(),
        s.get("rdf_bins").num() as usize,
        s.get("rdf_rmax_angstrom").num(),
    )
}

/// Runs the 80-leg chain for `p` beads (4-thread pool).
fn chain(p: usize) -> Chain {
    let root = tmp();
    let mut rows = Vec::new();
    let mut g_blocks = Vec::new();
    let mut prev_g: Option<Vec<f64>> = None;
    let mut prev_dir: Option<PathBuf> = None;
    let mut r = Vec::new();
    let samples_per_leg = (LEG_STEPS / STRIDE) as f64;
    pool(4).install(|| {
        for leg in 1..=LEGS {
            let cfg = parse(&neon_toml(p, leg * LEG_STEPS));
            let dir = root.path().join(format!("leg{leg}"));
            let s = match &prev_dir {
                None => run(&cfg, &dir).unwrap(),
                Some(d) => run_from_checkpoint(&cfg, &d.join("checkpoint.bin"), &dir).unwrap(),
            };
            rows.extend(observables(&dir));
            let (rr, g) = s.rdf.unwrap();
            // Cumulative mean over leg·k samples → mean of this leg's samples.
            let k = leg as f64;
            let block: Vec<f64> = match &prev_g {
                None => g.clone(),
                Some(pg) => g
                    .iter()
                    .zip(pg)
                    .map(|(gn, go)| {
                        (k * samples_per_leg * gn - (k - 1.0) * samples_per_leg * go)
                            / samples_per_leg
                    })
                    .collect(),
            };
            g_blocks.push(block);
            prev_g = Some(g);
            prev_dir = Some(dir);
            r = rr;
        }
    });
    assert_eq!(rows.len(), LEGS * LEG_STEPS / STRIDE);
    for (k, row) in rows.iter().enumerate() {
        assert_eq!(row[0] as usize, (k + 1) * STRIDE, "chain rows out of order");
    }
    Chain { rows, g_blocks, r }
}

/// Cached chains for P ∈ {1, 4, 8, 16, 32} (shared by the tests below).
fn neon(p: usize) -> &'static Chain {
    static C: [OnceLock<Chain>; 5] = [const { OnceLock::new() }; 5];
    let i = [1, 4, 8, 16, 32].iter().position(|&x| x == p).unwrap();
    C[i].get_or_init(|| chain(p))
}

/// (mean, σ) of column `c` (1 = e_prim_eh, 2 = e_vir_eh) of a chain, per atom, Eh.
fn energy(ch: &Chain, c: usize, what: &str) -> (f64, f64) {
    let x: Vec<f64> = ch.rows.iter().map(|r| r[c]).collect();
    mean_and_sigma(&x, LEG_STEPS / STRIDE, what)
}

/// Per-bin (mean, σ) of g from the leg blocks: σ from 80 one-leg blocks and 20 four-leg blocks,
/// the larger.
fn rdf_stats(ch: &Chain) -> (Vec<f64>, Vec<f64>) {
    let nb = ch.g_blocks[0].len();
    let sig = |len: usize, b: usize| {
        let n = ch.g_blocks.len() / len;
        let means: Vec<f64> = (0..n)
            .map(|k| (0..len).map(|i| ch.g_blocks[k * len + i][b]).sum::<f64>() / len as f64)
            .collect();
        let m = means.iter().sum::<f64>() / n as f64;
        let v = means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
        (m, (v / n as f64).sqrt())
    };
    (0..nb)
        .map(|b| {
            let (m, s1) = sig(1, b);
            let (_, s4) = sig(4, b);
            (m, s1.max(s4))
        })
        .unzip()
}

const K_B: f64 = 3.1668115634556e-6; // Eh/K (U-2), for printing in kelvin only

/// U-3: the generator's atomic-unit parameters (its own CODATA 2018 conversion) equal the
/// engine's parsed config to 1e-15, so both codes simulate the same system.
#[test]
fn same_system_as_ipi() {
    let d = ipi().get("derived_atomic_units");
    let c = parse(&neon_toml(16, 10));
    let close = |a: f64, b: f64, w: &str| {
        assert!(
            (a - b).abs() <= 1e-15 * b.abs(),
            "{w}: engine {a} vs generator {b}"
        )
    };
    close(c.system.cell.lengths()[0], d.get("box_bohr").num(), "box");
    close(c.potential.sigma, d.get("sigma_bohr").num(), "sigma");
    close(c.potential.epsilon, d.get("epsilon_eh").num(), "epsilon");
    close(c.potential.cutoff, d.get("cutoff_bohr").num(), "cutoff");
    close(c.system.mass, d.get("mass_me").num(), "mass");
    close(c.sampling.beta, d.get("beta").num(), "beta");
    let pr = ipi().get("protocol");
    assert_eq!(pr.get("n_equil").num() as usize, N_EQUIL);
    assert_eq!(pr.get("n_prod").num() as usize, LEGS * LEG_STEPS);
    assert_eq!(pr.get("stride").num() as usize, STRIDE);
}

/// I-1, T-1, E-1 … E-4, E-7 (vs i-PI): P = 16 ⟨E_vir⟩ and ⟨E_prim⟩ per atom, and g(r) in every
/// bin, agree with i-PI + LAMMPS within 5σ_combined (allowance 0, module comment).
#[test]
fn neon_p16_matches_ipi() {
    let ch = neon(16);
    for (c, key) in [(2usize, "e_vir"), (1, "e_prim")] {
        let (m, s) = energy(ch, c, key);
        let r = ipi().get(key);
        let (mr, sr) = (r.get("mean").num(), r.get("sigma").num());
        let sc = (s * s + sr * sr).sqrt();
        eprintln!(
            "{key}: engine {:.4} ± {:.4} K, i-PI {:.4} ± {:.4} K, diff {:.2} σ_c",
            m / K_B,
            s / K_B,
            mr / K_B,
            sr / K_B,
            (m - mr) / sc
        );
        assert!(
            (m - mr).abs() <= 5.0 * sc,
            "{key}: {m} ± {s} vs i-PI {mr} ± {sr}"
        );
    }
    let (g, sg) = rdf_stats(ch);
    let rdf = ipi().get("rdf");
    let (gr, sr, rr) = (
        rdf.get("g").nums(),
        rdf.get("sigma").nums(),
        rdf.get("r_bohr").nums(),
    );
    assert_eq!(g.len(), gr.len());
    let mut worst: f64 = 0.0;
    for b in 0..g.len() {
        assert!(
            (ch.r[b] - rr[b]).abs() <= 1e-12 * rr[b],
            "bin centres differ"
        );
        let sc = (sg[b] * sg[b] + sr[b] * sr[b]).sqrt();
        let d = (g[b] - gr[b]).abs();
        if sc > 0.0 {
            worst = worst.max(d / sc);
        }
        assert!(
            d <= 5.0 * sc,
            "g bin {b} (r = {} bohr): {} ± {} vs i-PI {} ± {}",
            rr[b],
            g[b],
            sg[b],
            gr[b],
            sr[b]
        );
    }
    eprintln!("g(r): worst bin {worst:.2} σ_c over {} bins", g.len());
}

/// RP-6, RP-3 (quantum vs classical): at 30 K, P = 16 vs P = 1 (classical, RP-6): quantum
/// delocalisation raises ⟨E⟩ (both estimators) and lowers the first peak of g(r) (at the bin of
/// the classical maximum), each by more than 5σ_combined.
#[test]
fn quantum_vs_classical_shift() {
    let (cl, qu) = (neon(1), neon(16));
    for (c, key) in [(2usize, "E_vir"), (1, "E_prim")] {
        let (m1, s1) = energy(cl, c, key);
        let (m16, s16) = energy(qu, c, key);
        let sc = (s1 * s1 + s16 * s16).sqrt();
        eprintln!(
            "{key}: P = 1 {:.3} ± {:.3} K, P = 16 {:.3} ± {:.3} K, shift {:+.3} K ({:.1} σ_c)",
            m1 / K_B,
            s1 / K_B,
            m16 / K_B,
            s16 / K_B,
            (m16 - m1) / K_B,
            (m16 - m1) / sc
        );
        assert!(m16 - m1 > 5.0 * sc, "{key}: no significant quantum rise");
    }
    let (g1, s1) = rdf_stats(cl);
    let (g16, s16) = rdf_stats(qu);
    let b = (0..g1.len())
        .max_by(|&a, &b| g1[a].total_cmp(&g1[b]))
        .unwrap();
    let sc = (s1[b] * s1[b] + s16[b] * s16[b]).sqrt();
    eprintln!(
        "g peak (bin {b}, r = {:.3} Å): P = 1 {:.4} ± {:.4}, P = 16 {:.4} ± {:.4} ({:.1} σ_c)",
        cl.r[b] * 0.529177210903,
        g1[b],
        s1[b],
        g16[b],
        s16[b],
        (g1[b] - g16[b]) / sc
    );
    assert!(
        g1[b] - g16[b] > 5.0 * sc,
        "first peak not lowered significantly"
    );
}

/// E-4, RP-2 (convergence in P): ⟨E_vir⟩ for P ∈ {4, 8, 16, 32} approaches its limit with
/// O(1/P²) differences: a weighted least-squares fit E∞ + a/P² leaves every residual within 5σ,
/// a < 0 (the energy rises with P toward the quantum limit, as the harmonic E-9 sum does), and the
/// total rise E(32) − E(4) exceeds 5σ_combined (reference implementation: −99.3, −97.6, −96.9,
/// −96.8 K/atom for P = 4, 8, 16, 32 in 96-ps runs).
///
/// E-1 vs E-2: the virial kinetic estimator's block-averaged error is smaller than the primitive's
/// at P ≥ 8. The comparison is made on the kinetic estimators K = E − V̄ (per atom, from the
/// e_*_eh and v_eh columns of the same rows): E_prim and E_vir share V̄, whose slow fluctuations
/// dominate both totals' errors at long blocks and would hide the estimator difference
/// (reference implementation, 96-ps runs, 20-ps blocks: σ(K_prim) = 0.12–0.19 K vs σ(K_vir) =
/// 0.05–0.06 K at P = 8–32, while the totals' errors are both ≈ 0.3 K).
#[test]
fn convergence_in_p() {
    let ps = [4usize, 8, 16, 32];
    let chains: Vec<&Chain> = std::thread::scope(|s| {
        let h: Vec<_> = ps.iter().map(|&p| s.spawn(move || neon(p))).collect();
        h.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let block = LEG_STEPS / STRIDE;
    let mut pts = Vec::new();
    for (&p, ch) in ps.iter().zip(&chains) {
        let (mv, sv) = energy(ch, 2, "E_vir");
        let kin = |c: usize| -> Vec<f64> { ch.rows.iter().map(|r| r[c] - r[3]).collect() };
        let (kp, skp) = mean_and_sigma(&kin(1), block, "K_prim");
        let (kv, skv) = mean_and_sigma(&kin(2), block, "K_vir");
        eprintln!(
            "P = {p:2}: E_vir {:.3} ± {:.3} K; K_prim {:.3} ± {:.3} K, K_vir {:.3} ± {:.3} K",
            mv / K_B,
            sv / K_B,
            kp / K_B,
            skp / K_B,
            kv / K_B,
            skv / K_B
        );
        if p >= 8 {
            assert!(
                skv < skp,
                "P = {p}: virial error {skv} not below primitive {skp}"
            );
        }
        pts.push((1.0 / (p * p) as f64, mv, sv));
    }
    // Weighted least squares for y = e + a x, x = 1/P².
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y, s) in &pts {
        let w = 1.0 / (s * s);
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    let det = sw * sxx - sx * sx;
    let a = (sw * sxy - sx * sy) / det;
    let e = (sxx * sy - sx * sxy) / det;
    let sa = (sw / det).sqrt();
    eprintln!(
        "fit: E_inf = {:.3} K, a = {:.1} ± {:.1} K",
        e / K_B,
        a / K_B,
        sa / K_B
    );
    for (&p, &(x, y, s)) in ps.iter().zip(&pts) {
        let res = y - (e + a * x);
        assert!(res.abs() <= 5.0 * s, "P = {p}: residual {res} vs σ {s}");
    }
    assert!(a < 0.0, "E_vir does not rise with P: a = {a} ± {sa}");
    let (e4, s4) = (pts[0].1, pts[0].2);
    let (e32, s32) = (pts[3].1, pts[3].2);
    let sc = (s4 * s4 + s32 * s32).sqrt();
    assert!(
        e32 - e4 > 5.0 * sc,
        "E(32) − E(4) = {} not > 5σ_c = {}",
        e32 - e4,
        5.0 * sc
    );
}
