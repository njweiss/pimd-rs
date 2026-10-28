//! Long validation (feature `long-validation`): PIMD of the quartic double well against exact
//! diagonalisation (SPEC P-2, E-2, E-4, I-1, T-1, RP-3).
//!
//! Reference numbers come from `data/double_well.json`, written by `data/gen_double_well.py`
//! (numpy/scipy; versions recorded in the JSON): the exact quantum thermal energy of the 1D well
//! (sinc-DVR diagonalisation, converged to 1e-12) and the exact finite-P ring-polymer energy
//! −∂ ln Z_P/∂β (transfer matrix, converged to 1e-9) for P = 32 and 64. The 3D `DoubleWell` is
//! separable (P-2), so one atom in 3D is three independent copies: every 1D value is multiplied
//! by 3.
#![cfg(feature = "long-validation")]

mod common;

use common::mean_and_sigma;
use pimd_core::Beads;
use pimd_potentials::DoubleWell;

const DATA: &str = include_str!("data/double_well.json");

/// The number after `"key":` that follows the keys of `path` in order (enough for the flat,
/// generator-written JSON; no JSON dependency).
fn json_number(path: &[&str]) -> f64 {
    let mut pos = 0;
    for key in path {
        let pat = format!("\"{key}\":");
        pos += DATA[pos..]
            .find(&pat)
            .unwrap_or_else(|| panic!("key {key} missing"))
            + pat.len();
    }
    let rest = DATA[pos..].trim_start();
    let end = rest.find([',', '\n', '}']).unwrap();
    rest[..end].trim().parse().unwrap()
}

const DT: f64 = 0.05;
const TAU0: f64 = 1.0; // 20 dt; about 1/ω_well = 0.5 (near-critical damping of the centroid)
const N_EQ: usize = 20_000;
const N_STEPS: usize = 1_000_000;
const BLOCK: usize = 10_000; // 500 a.u.
const DT_BIAS: f64 = 0.003;

/// Runs P beads and checks the two assertions described on the tests.
///
/// Setup: one atom in 3D, m = 1, a = 1/8, b = 1, β = 2 (read from the JSON): minima at ±2,
/// barrier 2 Eh = 4 kT, ω_well = 2 so β ω_well = 4, ground state 0.90 Eh above the well bottom
/// (below the barrier top). All beads start in the minimum (2, 2, 2) with RP-3 thermal momenta.
/// PILE-L, dt = 0.05, τ_0 = 1, 20 000 equilibration + 1 000 000 sampled steps (50 000 a.u.),
/// time-averaged `total_virial`, block σ from 10 000-step blocks checked against 40 000.
///
/// Exact reference (3 × 1D, from the JSON): E_exact = −3.007976; finite P: E_32 = −3.012905,
/// E_64 = −3.009212. Discretisation error E_P − E_exact = −4.93e-3 (P = 32), −1.24e-3 (P = 64);
/// ratio 3.99, i.e. O(1/P²) (the generator asserts it lies in (3.8, 4.2)). Under 1/P² scaling the
/// same errors follow from the difference of the two P values alone: (E_32 − E_64)·4/3 = −4.92e-3
/// and (E_32 − E_64)/3 = −1.23e-3.
///
/// Assertions:
/// 1. |⟨E_vir⟩ − E_P| ≤ 5σ + DT_BIAS (the finite-P value; the sharp test);
/// 2. |⟨E_vir⟩ − E_exact| ≤ 5σ + DT_BIAS + |E_P − E_exact| (the brief's comparison with the
///    exact quantum energy; the extra allowance is the discretisation error above).
///
/// Tolerance derivation (reference implementation, release build, 100 000 a.u. runs):
/// σ ≈ 3.7–4e-3 for these 50 000 a.u. runs (2.5–3.1e-3 for 100 000 a.u.). dt bias of ⟨E_vir⟩:
/// P = 32, two seeds each: mean − E_32 = −0.7e-3, −1.3e-3 at dt = 0.05 and −1.1e-3, −2.9e-3 at
/// dt = 0.025 (σ ≈ 2.8e-3 per run); P = 64: eight seeds at dt = 0.05 give −3.7e-3 … +5.3e-3,
/// mean +0.3e-3 ± 1.0e-3, six seeds at dt = 0.025 give −4.6e-3 … +2.7e-3, mean +0.8e-3 ± 1.1e-3
/// (every run within 2.1σ of E_P). No dt dependence is resolved at the 1–2e-3 level, so
/// DT_BIAS = 3e-3, below a third of the tolerance 5σ + DT_BIAS ≈ 0.022.
/// (The primitive estimator, not tested here, has a large dt bias at these P: −0.015 at P = 32
/// and −0.032 at P = 64 for dt = 0.05, −0.012 at P = 64 for dt = 0.025.)
fn check(p: usize, seed: u64) {
    let (m, a, b, beta) = (
        json_number(&["parameters", "mass"]),
        json_number(&["parameters", "a"]),
        json_number(&["parameters", "b"]),
        json_number(&["parameters", "beta"]),
    );
    let exact = 3.0 * json_number(&["exact_energy_1d"]);
    let e_p = 3.0 * json_number(&["finite_p_energy_1d", &p.to_string()]);
    // Start every bead in the right-hand minimum x = √(b/(2a)) along each axis.
    let x0 = (b / (2.0 * a)).sqrt();
    let mut q = Beads::zeros(p, 1);
    for j in 0..p {
        q.set(j, 0, [x0, x0, x0]);
    }
    let pot = DoubleWell::new(a, b).unwrap();
    let s = common::run_pile(q, &[m], Box::new(pot), beta, DT, TAU0, N_EQ, N_STEPS, seed);
    let (mean, sigma) = mean_and_sigma(&s.vir, BLOCK, "E_vir");
    let tol = 5.0 * sigma + DT_BIAS;
    let disc = (e_p - exact).abs();
    eprintln!(
        "P = {p}: <E_vir> {mean:.6} E_P {e_p:.6} exact {exact:.6} diff_P {:+.2e} diff_exact {:+.2e} sigma {sigma:.2e}",
        mean - e_p,
        mean - exact,
    );
    assert!(
        (mean - e_p).abs() <= tol,
        "P = {p}: {mean} vs finite-P {e_p} (tol {tol:e})"
    );
    assert!(
        (mean - exact).abs() <= tol + disc,
        "P = {p}: {mean} vs exact {exact} (tol {:e})",
        tol + disc
    );
}

/// P-2, E-2, E-4, I-1, T-1: P = 32 against E_32 and against E_exact (setup and tolerance in
/// `check`).
#[test]
fn p32_matches_reference() {
    check(32, 32);
}

/// P-2, E-2, E-4, I-1, T-1: P = 64 against E_64 and against E_exact (setup and tolerance in
/// `check`).
#[test]
fn p64_matches_reference() {
    check(64, 64);
}
