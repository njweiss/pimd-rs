//! Tests for `Thermostat::Trpmd` (SPEC T-4; MILESTONE_7 required test 1), with T-1, D-2, D-4.
//!
//! Expectations: the T-4 coefficients c_k = exp(−2λω_k dt) (γ_0 = 0) written in the test from
//! the RP-2/RP-4 frequencies (`m7_common::ref_c_trpmd`), the T-1 O step evaluated in the test's
//! own RP-4 basis with a same-seed `Rng`, and the SPEC identity `Trpmd { lambda: 1.0 }` ≡
//! `PileL { tau0: None }` bitwise.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod m7_common;

use common::{SplitMix64, max_rel_diff, random_beads, ref_freqs, ref_matrix, to_beads, to_modes};
use m7_common::{assert_same_sim, cluster_sim, ref_c_trpmd, ref_o_step_with};
use pimd_core::{Beads, Rng, Transform};
use pimd_integrators::{IntegratorError, RingPolymerIntegrator, RingState, Thermostat};

const MASSES: [f64; 3] = [1.0, 4.0, 1836.0];

/// T-4, D-2, D-4: `Trpmd { lambda: 1.0 }` and `PileL { tau0: None }` give bitwise identical
/// trajectories (state, forces, energies) and RNG positions over 50 BAOAB steps of an 8-atom LJ
/// cluster with P = 4 (SPEC T-4: "`Trpmd { lambda: 1.0 }` is `PileL { tau0: None }` bitwise").
#[test]
fn lambda_one_is_pile_l_without_centroid_friction_bitwise() {
    let mut a = cluster_sim(4, 11, Thermostat::Trpmd { lambda: 1.0 });
    let mut b = cluster_sim(4, 11, Thermostat::PileL { tau0: None });
    let (mut ra, mut rb) = (Rng::from_seed(5), Rng::from_seed(5));
    for s in 1..=50 {
        a.step(&mut ra).unwrap();
        b.step(&mut rb).unwrap();
        assert_same_sim(&a, &b, &format!("step {s}"));
        assert_eq!(
            ra.snapshot(),
            rb.snapshot(),
            "step {s}: RNG positions differ"
        );
    }
}

/// T-4, T-1, D-2, D-4: one O step with λ = 0.5 on a state with known normal-mode momenta p̃
/// (set in the test's RP-4 basis): every mode k ≥ 1 becomes c_k p̃_k + sqrt((1 − c_k²) m_i P/β) ξ
/// with c_k = exp(−λ·2ω_k·dt) (ω_k from RP-2/RP-4, not from the code), the centroid k = 0 keeps
/// p̃_0 (γ_0 = 0: c_0 = 1, zero noise), ξ drawn from a same-seed `Rng` in the order k, atom i,
/// dimension d, and exactly P·N·3 normals are consumed (the k = 0 draws included). Agreement
/// 1e-12 relative (floor 1); odd and even P, both transforms. A separate check recovers each c_k
/// from the update, (p̃'_k − amp_k ξ)/p̃_k, and compares it with exp(−2λω_k dt) to 1e-12, so a
/// friction λω_k or a half-step exp(−γ dt/2) cannot hide behind the noise term.
#[test]
fn o_step_uses_two_lambda_omega_and_leaves_the_centroid() {
    let mut sm = SplitMix64(41);
    let (beta, dt, lambda) = (1.3, 0.3, 0.5);
    for p in [2, 3, 4, 6, 8] {
        for transform in [Transform::Matrix, Transform::Fft] {
            let ig = RingPolymerIntegrator::with_transform(
                dt,
                beta,
                p,
                Thermostat::Trpmd { lambda },
                transform,
            )
            .unwrap();
            let mat = ref_matrix(p);
            // Known normal-mode momenta, all components away from zero.
            let pt0 = random_beads(&mut sm, p, MASSES.len(), 0.5, 3.0);
            let s0 = RingState {
                q: random_beads(&mut sm, p, MASSES.len(), -1.0, 1.0),
                p: to_beads(&mat, &pt0),
                masses: MASSES.to_vec(),
            };
            let seed = 900 + p as u64;
            let mut rng = Rng::from_seed(seed);
            let mut s = s0.clone();
            ig.thermostat_step(&mut s, &mut rng);

            let c = ref_c_trpmd(p, beta, dt, lambda);
            let mut rng_ref = Rng::from_seed(seed);
            let want = ref_o_step_with(&s0.p, &MASSES, beta, &c, &mut rng_ref);
            let d = max_rel_diff(&s.p, &want, 1.0);
            assert!(d <= 1e-12, "P = {p}, {transform:?}: rel diff {d:e}");
            assert_eq!(s.q, s0.q, "the O step must not touch positions");
            assert_eq!(rng.normal(), rng_ref.normal(), "P = {p}: not P·N·3 draws");

            // c_k recovered from the update, mode by mode.
            let pt1 = to_modes(&mat, &s.p);
            let mut xi = Rng::from_seed(seed);
            let w = ref_freqs(p, beta);
            let pf = p as f64;
            for k in 0..p {
                let ck = (-2.0 * lambda * w[k] * dt).exp();
                for (i, &m) in MASSES.iter().enumerate() {
                    let amp = ((1.0 - ck * ck) * m * pf / beta).sqrt();
                    let (a0, a1) = (pt0.get(k, i), pt1.get(k, i));
                    for dd in 0..3 {
                        let got = (a1[dd] - amp * xi.normal()) / a0[dd];
                        assert!(
                            (got - ck).abs() <= 1e-12,
                            "P = {p}, k = {k}, atom {i}: c_k = {got}, want {ck}"
                        );
                    }
                }
            }
            // Centroid (k = 0): unchanged to rounding.
            for i in 0..MASSES.len() {
                let (a0, a1) = (pt0.get(0, i), pt1.get(0, i));
                for dd in 0..3 {
                    assert!(
                        (a1[dd] - a0[dd]).abs() <= 1e-12 * a0[dd].abs().max(1.0),
                        "P = {p}: centroid momentum changed"
                    );
                }
            }
        }
    }
}

/// T-4 (λ dependence): the internal-mode damping scales with λ. For λ = 0.25 and λ = 2 the
/// recovered c_k equal exp(−2λω_k dt) (rules out a hard-coded λ = 1/2 or γ_k independent of λ).
#[test]
fn damping_scales_with_lambda() {
    let (beta, dt, p) = (2.0, 0.2, 4);
    let mat = ref_matrix(p);
    let w = ref_freqs(p, beta);
    let mut sm = SplitMix64(43);
    let pt0 = random_beads(&mut sm, p, 1, 0.5, 3.0);
    for lambda in [0.25, 2.0] {
        let ig = RingPolymerIntegrator::new(dt, beta, p, Thermostat::Trpmd { lambda }).unwrap();
        let mut s = RingState {
            q: Beads::zeros(p, 1),
            p: to_beads(&mat, &pt0),
            masses: vec![2.0],
        };
        let mut rng = Rng::from_seed(3);
        ig.thermostat_step(&mut s, &mut rng);
        let pt1 = to_modes(&mat, &s.p);
        let mut xi = Rng::from_seed(3);
        for k in 0..p {
            let ck = (-2.0 * lambda * w[k] * dt).exp();
            let amp = ((1.0 - ck * ck) * 2.0 * p as f64 / beta).sqrt();
            let (a0, a1) = (pt0.get(k, 0), pt1.get(k, 0));
            for d in 0..3 {
                let got = (a1[d] - amp * xi.normal()) / a0[d];
                assert!(
                    (got - ck).abs() <= 1e-12,
                    "λ = {lambda}, k = {k}: {got} vs {ck}"
                );
            }
        }
    }
}

/// T-4 (P = 1; MILESTONE_7 "P = 1"): with one bead there are no internal modes; c_0 = 1 and the
/// noise amplitude is zero, so a `Trpmd` trajectory equals the `Off` trajectory bitwise, while
/// the `Trpmd` run still draws N·3 normals per step (T-4 draw count) and `Off` draws none (T-3).
#[test]
fn one_bead_trpmd_is_rpmd_but_draws() {
    let mut a = cluster_sim(1, 12, Thermostat::Trpmd { lambda: 0.5 });
    let mut b = cluster_sim(1, 12, Thermostat::Off);
    let (mut ra, mut rb) = (Rng::from_seed(8), Rng::from_seed(8));
    let n3 = 3 * a.state().masses.len();
    for s in 1..=30 {
        a.step(&mut ra).unwrap();
        b.step(&mut rb).unwrap();
        assert_same_sim(&a, &b, &format!("step {s}"));
    }
    assert_eq!(
        rb.snapshot(),
        Rng::from_seed(8).snapshot(),
        "Off drew numbers"
    );
    let mut want = Rng::from_seed(8);
    for _ in 0..30 * n3 {
        want.normal();
    }
    assert_eq!(
        ra.snapshot(),
        want.snapshot(),
        "Trpmd at P = 1: not N·3 draws per step"
    );
}

/// T-4: λ must be positive and finite. λ = 0, −0.5, NaN, +∞, −∞ are
/// `IntegratorError::InvalidValue` from `RingPolymerIntegrator::new` and `with_transform`.
#[test]
fn bad_lambda_rejected() {
    for lambda in [0.0, -0.0, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let r = RingPolymerIntegrator::new(0.1, 1.0, 4, Thermostat::Trpmd { lambda });
        assert!(
            matches!(r, Err(IntegratorError::InvalidValue { .. })),
            "λ = {lambda}: {r:?}"
        );
        let r = RingPolymerIntegrator::with_transform(
            0.1,
            1.0,
            4,
            Thermostat::Trpmd { lambda },
            Transform::Fft,
        );
        assert!(
            matches!(r, Err(IntegratorError::InvalidValue { .. })),
            "λ = {lambda} (FFT): {r:?}"
        );
    }
    // A small positive λ is valid.
    assert!(RingPolymerIntegrator::new(0.1, 1.0, 4, Thermostat::Trpmd { lambda: 1e-6 }).is_ok());
}
