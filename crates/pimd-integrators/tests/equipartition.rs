//! Long validation (feature `long-validation`): equilibrium normal-mode momenta of a free ring
//! polymer under PILE-L (SPEC T-1, T-2, RP-3, I-1, I-2).
#![cfg(feature = "long-validation")]

mod common;

use common::{ref_matrix, thermal_momenta, to_modes};
use pimd_core::{Beads, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::Harmonic;

/// Mean of `x` and its block-averaged standard error with blocks of `len` samples.
fn block_mean_sigma(x: &[f64], len: usize) -> (f64, f64) {
    let nb = x.len() / len;
    let means: Vec<f64> = (0..nb)
        .map(|b| x[b * len..(b + 1) * len].iter().sum::<f64>() / len as f64)
        .collect();
    let m = means.iter().sum::<f64>() / nb as f64;
    let var = means.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (nb - 1) as f64;
    (m, (var / nb as f64).sqrt())
}

/// T-1, T-2, RP-3: free ring polymer (`Harmonic` with ω = 0, so F = 0), P = 8, N = 16 (masses
/// 1..4 m_e), β = 4, dt = 0.1, PILE-L τ_0 = 1. In equilibrium every normal-mode momentum
/// component has ⟨p̃_k²⟩ = m_i P/β (RP-3); the test averages p̃_k²/m_i over atoms and components
/// and checks P/β = 2 for each k within 5σ.
///
/// Statistics: 2000 equilibration steps (20 a.u. ≫ τ_0 and ≫ 1/γ_k ≤ 1/(2ω_1) = 0.33), then
/// 100 000 sampled steps. σ is the block-averaged standard error with blocks of 500 steps
/// (50 a.u. ≫ τ_0), cross-checked against blocks of 2000 steps (σ_2000 ≤ 1.5 σ_500, i.e. the
/// shorter blocks are already uncorrelated); the larger of the two is used. Relative σ is about
/// 0.08–0.25 % per mode in the reference run (σ ≈ 1.5e-3 … 5e-3, centroid largest).
///
/// Discretisation-bias allowance: zero. With F = 0 the B steps are identities, A is the exact
/// rotation of each mode (I-2), which preserves the Gaussian N(0, m P/β) momentum and
/// N(0, P/(β m ω_k²)) position marginals exactly, and the O step preserves the momentum Gaussian
/// for any c_k. So ⟨p̃_k²⟩ = m P/β holds at every dt. Validated with the reference
/// implementation at dt = 0.1 and dt = 0.05 (same physical time): all eight modes within 1.9σ
/// at both.
#[test]
fn free_ring_mode_momenta_are_thermal() {
    let (p, n, beta, dt, tau0) = (8usize, 16usize, 4.0, 0.1, 1.0);
    let masses: Vec<f64> = (0..n).map(|i| 1.0 + (i % 4) as f64).collect();
    let mut rng = Rng::from_seed(2024);
    let state = RingState {
        q: Beads::zeros(p, n),
        p: thermal_momenta(&mut rng, p, &masses, beta),
        masses: masses.clone(),
    };
    let ig =
        RingPolymerIntegrator::new(dt, beta, p, Thermostat::PileL { tau0: Some(tau0) }).unwrap();
    let pot = Harmonic::new(masses.clone(), 0.0).unwrap();
    let mut sim = RingPolymerSim::new(state, None, ig, Box::new(pot)).unwrap();
    for _ in 0..2000 {
        sim.step(&mut rng).unwrap();
    }
    let c = ref_matrix(p);
    let steps = 100_000;
    let mut series = vec![Vec::with_capacity(steps); p];
    for _ in 0..steps {
        sim.step(&mut rng).unwrap();
        let pt = to_modes(&c, &sim.state().p);
        for (k, s) in series.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, &m) in masses.iter().enumerate() {
                acc += pt.get(k, i).iter().map(|x| x * x).sum::<f64>() / m;
            }
            s.push(acc / (3 * n) as f64);
        }
    }
    let want = p as f64 / beta;
    for (k, s) in series.iter().enumerate() {
        let (mean, s500) = block_mean_sigma(s, 500);
        let (_, s2000) = block_mean_sigma(s, 2000);
        assert!(
            s2000 <= 1.5 * s500,
            "mode {k}: blocks correlated ({s500:e}, {s2000:e})"
        );
        let sigma = s500.max(s2000);
        eprintln!(
            "mode {k}: <p~^2/m> {mean:.5} want {want} sigma {sigma:.2e} ({:+.2} sigma)",
            (mean - want) / sigma
        );
        assert!(
            (mean - want).abs() <= 5.0 * sigma,
            "mode {k}: <p~^2/m> = {mean} vs {want} (sigma {sigma:e})"
        );
    }
}
