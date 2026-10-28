//! Long validation (feature `long-validation`): PILE-L sampling of the harmonic oscillator
//! against the exact finite-P result E-9 (SPEC E-1 … E-4, E-9, I-1, T-1, T-2, RP-3, RP-6).
//!
//! Settings (all three P): N = 4 atoms (masses 1, 2, 3, 4 m_e; E-9 is mass independent),
//! ω = 1, β = 4 (β ω = 4, quantum regime), dt = 0.05, PILE-L τ_0 = 1 = 20 dt (about 1/ω, which
//! decorrelates the centroid fastest; 100 dt gave about 1.5× larger σ for the same cost),
//! start q = 0 with RP-3 thermal momenta, 5 000 equilibration steps (250 a.u. = 250 τ_0), then
//! 1 000 000 sampled steps (50 000 a.u.). σ: block averages over 4 000 steps (200 a.u.),
//! checked against 16 000-step blocks (`common::mean_and_sigma`).
//!
//! Tolerance: |⟨E⟩ − E-9| ≤ 5σ + BIAS, BIAS = 0.005 for both estimators and every P.
//! The discretisation bias of BAOAB for this linear system is known exactly at finite dt
//! (`data/harmonic_baoab_bias.py`, discrete Lyapunov equation per normal mode):
//!   dt = 0.05:  P = 1: 0;  P = 4: E_prim −2.9e-4, E_vir +3.8e-4;  P = 16: E_prim −3.8e-3, E_vir +5.4e-4
//!   dt = 0.025: P = 1: 0;  P = 4: −7.2e-5, +9.4e-5;               P = 16: −9.5e-4, +1.4e-4
//! (O(dt²): the bias drops 4× at dt/2). Reference implementation, 100 000 a.u. runs, two seeds:
//! every |mean − E-9| at dt = 0.05 and at dt = 0.025 is within 2.1σ of the exact bias above
//! (σ ≈ 2.7e-3 at P = 1, 3–3.7e-3 at P = 4, 5e-3 / 3.2e-3 for E_prim / E_vir at P = 16).
//! The largest bias, 3.8e-3, is below a third of the smallest tolerance (5σ + 0.005 ≈ 0.024).
//! Measured σ in these 50 000 a.u. runs (reference): about 4e-3 (P = 1), 4–5e-3 (P = 4),
//! 7e-3 / 4.6e-3 (P = 16, E_prim / E_vir).
#![cfg(feature = "long-validation")]

mod common;

use std::f64::consts::PI;

use common::{Series, mean_and_sigma, run_pile, sample_variance};
use pimd_core::Beads;
use pimd_potentials::Harmonic;

const OMEGA: f64 = 1.0;
const BETA: f64 = 4.0;
const MASSES: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
const DT: f64 = 0.05;
const TAU0: f64 = 1.0;
const N_EQ: usize = 5_000;
const N_STEPS: usize = 1_000_000;
const BLOCK: usize = 4_000;
const BIAS: f64 = 0.005;

/// E-9: ⟨E⟩ = (3N/β) Σ_{k=0}^{P−1} ω²/(ω_k² + ω²), ω_k the RP-4 frequencies. The multiset
/// {ω_k} is {2 ω_P sin(πk/P) : k = 0..P−1} (eigenvalues of the cyclic spring matrix, RP-5),
/// ω_P = P/β.
fn e9(p: usize) -> f64 {
    let wp = p as f64 / BETA;
    let s: f64 = (0..p)
        .map(|k| {
            let wk = 2.0 * wp * (PI * k as f64 / p as f64).sin();
            OMEGA * OMEGA / (wk * wk + OMEGA * OMEGA)
        })
        .sum();
    3.0 * MASSES.len() as f64 / BETA * s
}

fn run(p: usize, seed: u64) -> Series {
    let pot = Harmonic::new(MASSES.to_vec(), OMEGA).unwrap();
    run_pile(
        Beads::zeros(p, MASSES.len()),
        &MASSES,
        Box::new(pot),
        BETA,
        DT,
        TAU0,
        N_EQ,
        N_STEPS,
        seed,
    )
}

/// Checks both estimators against E-9: |mean − E-9| ≤ 5σ + BIAS.
fn check(p: usize, s: &Series) {
    let want = e9(p);
    for (name, x) in [("E_prim", &s.prim), ("E_vir", &s.vir)] {
        let (m, sigma) = mean_and_sigma(x, BLOCK, name);
        let tol = 5.0 * sigma + BIAS;
        eprintln!(
            "P = {p} {name}: mean {m:.6} want {want:.6} diff {:+.2e} sigma {sigma:.2e} tol {tol:.2e}",
            m - want
        );
        assert!(
            (m - want).abs() <= tol,
            "P = {p} {name}: {m} vs E-9 {want} (sigma {sigma:e}, tol {tol:e})"
        );
    }
}

/// E-9, RP-6, T-1: P = 1 is classical Langevin on one bead; E-9 gives 3N/β = 3 for both
/// estimators (which coincide at P = 1).
#[test]
fn p1_matches_e9() {
    assert!((e9(1) - 3.0).abs() < 1e-12, "E-9 hand value");
    let s = run(1, 101);
    check(1, &s);
}

/// E-9 at P = 4: (3·4/4)·Σ_k 1/(ω_k² + 1) with ω_P = 1, ω_k ∈ {0, √2, √2, 2}:
/// 3·(1 + 1/3 + 1/3 + 1/5) = 5.6.
#[test]
fn p4_matches_e9() {
    assert!((e9(4) - 5.6).abs() < 1e-12, "E-9 hand value");
    let s = run(4, 104);
    check(4, &s);
}

/// E-9 at P = 16: 6.178180 (formula above; the P → ∞ value 3N (ω/2) coth(βω/2) = 6.2239 is not
/// used). Also E-1/E-2: the sample variance of E_vir is smaller than that of E_prim. Reference:
/// var(E_prim) ≈ 4.72, var(E_vir) ≈ 0.67, ratio 0.142 (four runs, all within 0.141–0.143); the
/// test requires ratio < 0.3.
#[test]
fn p16_matches_e9_and_virial_has_lower_variance() {
    let s = run(16, 116);
    check(16, &s);
    let (vp, vv) = (sample_variance(&s.prim), sample_variance(&s.vir));
    eprintln!(
        "P = 16: var(E_prim) {vp:.4} var(E_vir) {vv:.4} ratio {:.3}",
        vv / vp
    );
    assert!(vv < 0.3 * vp, "var(E_vir) = {vv}, var(E_prim) = {vp}");
}
