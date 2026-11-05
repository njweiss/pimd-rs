//! PA-CMD dynamical masses and free propagation (MILESTONE_7 M7b required test 1; SPEC Y-6, I-2,
//! RP-6, D-3).
//!
//! Expectations: the Y-6 mass factors written out by hand for P = 4 and P = 8; the I-2 free
//! propagator evaluated in the test's own RP-4 basis with mass m_i·s_k and frequency Ω
//! (`m7b_common::ref_free_propagate_dyn`); the exact period 2π/Ω of every internal mode (all sit at
//! Ω under Y-6); fingerprints of Ω-less trajectories recorded with the code of `main` at e6008c9.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod m7_common;
mod m7b_common;

use std::f64::consts::PI;

use common::{SplitMix64, max_rel_diff, random_beads};
use m7_common::{CLUSTER_BETA, CLUSTER_DT, assert_same_sim, cluster_state, lj};
use m7b_common::{fingerprint, ref_free_propagate_dyn};
use pimd_core::{Beads, Rng, Transform};
use pimd_integrators::{
    IntegratorError, RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat,
};

const MASSES: [f64; 3] = [1.0, 4.0, 1836.0];
const LIGHT: [f64; 3] = [1.0, 2.5, 7.0];

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * b.abs(), "{what}: {a:e} vs {b:e}");
}

/// Y-6 (hand values): s_0 = 1, s_k = (ω_k/Ω)² with ω_k = 2ω_P sin(mπ/P), ω_P = P/β (RP-2, RP-4).
/// P = 4, β = 2 (ω_P = 2: ω_1 = ω_2 = 2√2, ω_3 = 4), Ω = 5: s = [1, 8/25, 8/25, 16/25].
/// P = 8, β = 1 (ω_P = 8: ω_{1,2} = 16 sin(π/8), ω_{3,4} = 8√2, ω_{5,6} = 16 sin(3π/8),
/// ω_7 = 16), Ω = 20: s = [1, (0.8 sin(π/8))² ×2, 0.32 ×2, (0.8 sin(3π/8))² ×2, 0.64].
/// Agreement 1e-14 relative; `adiabatic_frequency()` returns Ω; both transforms and every
/// thermostat give the same factors. Without Ω the factors are exactly 1.0 and
/// `adiabatic_frequency()` is `None`. P = 1 with Ω: s = [1] (RP-6, Y-6).
#[test]
fn mass_factors_by_hand() {
    let a = (0.8 * (PI / 8.0).sin()).powi(2);
    let b = (0.8 * (3.0 * PI / 8.0).sin()).powi(2);
    let cases: [(usize, f64, f64, Vec<f64>); 3] = [
        (4, 2.0, 5.0, vec![1.0, 0.32, 0.32, 0.64]),
        (8, 1.0, 20.0, vec![1.0, a, a, 0.32, 0.32, b, b, 0.64]),
        (1, 1.0, 3.0, vec![1.0]),
    ];
    let ths = [
        Thermostat::Off,
        Thermostat::PileL { tau0: Some(0.5) },
        Thermostat::PileL { tau0: None },
        Thermostat::Trpmd { lambda: 0.5 },
    ];
    for (p, beta, omega, want) in cases {
        for transform in [Transform::Matrix, Transform::Fft] {
            for th in ths {
                let plain =
                    RingPolymerIntegrator::with_transform(0.1, beta, p, th, transform).unwrap();
                assert_eq!(plain.adiabatic_frequency(), None);
                assert_eq!(plain.mass_factors(), vec![1.0; p].as_slice());
                let ig = plain.with_adiabatic_frequency(omega).unwrap();
                assert_eq!(ig.adiabatic_frequency(), Some(omega));
                let s = ig.mass_factors();
                assert_eq!(s.len(), p);
                for k in 0..p {
                    close(s[k], want[k], 1e-14, &format!("P = {p}, s_{k}"));
                }
            }
        }
    }
}

/// Y-6: Ω must be positive and finite: 0, −0, −1, NaN, +∞, −∞ are
/// `IntegratorError::InvalidValue`; a tiny and a huge positive Ω are accepted.
#[test]
fn bad_omega_rejected() {
    for omega in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for th in [Thermostat::Off, Thermostat::Trpmd { lambda: 0.5 }] {
            let r = RingPolymerIntegrator::new(0.1, 1.0, 4, th)
                .unwrap()
                .with_adiabatic_frequency(omega);
            assert!(
                matches!(r, Err(IntegratorError::InvalidValue { .. })),
                "Ω = {omega}: {r:?}"
            );
        }
    }
    for omega in [1e-6, 1e6] {
        let ig = RingPolymerIntegrator::new(0.1, 1.0, 4, Thermostat::Off)
            .unwrap()
            .with_adiabatic_frequency(omega)
            .unwrap();
        assert_eq!(ig.adiabatic_frequency(), Some(omega));
    }
}

/// Y-6, I-2: with Ω every internal mode has dynamical frequency Ω, so free propagation over one
/// period t = 2π/Ω returns every internal normal mode (positions and momenta) to its start, while
/// the centroid, with its physical mass (s_0 = 1), drifts freely: every bead position becomes
/// q_j(0) + t·p̄_i/m_i (p̄_i = (1/P) Σ_j p_{j,i}) and every bead momentum returns to p_j(0).
/// Agreement 1e-12 relative (floor 1); P = 3, 4, 5, 8; both transforms; distinct masses
/// (1, 2.5, 7: light enough that the rounding of sin(Ω·2π/Ω) ≈ 1e-16, times m_i s_k Ω |q̃|, stays
/// below 1e-12); two Ω. (With physical masses the internal modes have incommensurate periods and
/// do not return.)
#[test]
fn one_period_returns_internal_modes_and_drifts_centroid() {
    let mut sm = SplitMix64(61);
    for p in [3, 4, 5, 8] {
        for transform in [Transform::Matrix, Transform::Fft] {
            for omega in [3.0, 17.5] {
                let beta = 1.7;
                let ig =
                    RingPolymerIntegrator::with_transform(0.1, beta, p, Thermostat::Off, transform)
                        .unwrap()
                        .with_adiabatic_frequency(omega)
                        .unwrap();
                let s0 = RingState {
                    q: random_beads(&mut sm, p, 3, -1.0, 1.0),
                    p: random_beads(&mut sm, p, 3, -2.0, 2.0),
                    masses: LIGHT.to_vec(),
                };
                let t = 2.0 * PI / omega;
                let mut s = s0.clone();
                ig.free_propagate(&mut s, t);
                let mut want_q = Beads::zeros(p, 3);
                for (i, &m) in LIGHT.iter().enumerate() {
                    let mut pbar = [0.0; 3];
                    for j in 0..p {
                        let v = s0.p.get(j, i);
                        for d in 0..3 {
                            pbar[d] += v[d] / p as f64;
                        }
                    }
                    for j in 0..p {
                        let x = s0.q.get(j, i);
                        want_q.set(j, i, [0, 1, 2].map(|d| x[d] + t * pbar[d] / m));
                    }
                }
                let dq = max_rel_diff(&s.q, &want_q, 1.0);
                let dp = max_rel_diff(&s.p, &s0.p, 1.0);
                assert!(
                    dq <= 1e-12 && dp <= 1e-12,
                    "P = {p}, {transform:?}, Ω = {omega}: dq {dq:e}, dp {dp:e}"
                );
            }
        }
    }
}

/// Y-6, I-2: `free_propagate` with Ω equals I-2 written out per normal mode with mass m_i·s_k and
/// frequency Ω (centroid: drift with m_i) in the test's RP-4 basis, to 1e-12 relative (floor 1),
/// for t = 0.37 and t = −0.21, P = 2, 3, 4, 6, 8, both transforms. (An s_k = ω_k/Ω, Ω applied to
/// the centroid, or physical masses in I-2 all fail this.)
#[test]
fn free_propagation_uses_dynamical_masses() {
    let mut sm = SplitMix64(62);
    let (beta, omega) = (1.3, 9.0);
    for p in [2, 3, 4, 6, 8] {
        for transform in [Transform::Matrix, Transform::Fft] {
            let ig =
                RingPolymerIntegrator::with_transform(0.1, beta, p, Thermostat::Off, transform)
                    .unwrap()
                    .with_adiabatic_frequency(omega)
                    .unwrap();
            for t in [0.37, -0.21] {
                let s0 = RingState {
                    q: random_beads(&mut sm, p, 3, -1.0, 1.0),
                    p: random_beads(&mut sm, p, 3, -2.0, 2.0),
                    masses: MASSES.to_vec(),
                };
                let (wq, wp) = ref_free_propagate_dyn(&s0.q, &s0.p, &MASSES, beta, omega, t);
                let mut s = s0.clone();
                ig.free_propagate(&mut s, t);
                let dq = max_rel_diff(&s.q, &wq, 1.0);
                let dp = max_rel_diff(&s.p, &wp, 1.0);
                assert!(
                    dq <= 1e-12 && dp <= 1e-12,
                    "P = {p}, {transform:?}, t = {t}: dq {dq:e}, dp {dp:e}"
                );
            }
        }
    }
}

/// Fingerprints (`m7b_common::fingerprint`: FNV-1a of q, p, forces, energies and the next RNG
/// normal) of the 8-atom LJ cluster (P = 4, `m7_common::cluster_state(4, 71)`, RNG seed 72) after
/// 40 steps, then a `resample_momenta` and 10 more steps, for each thermostat and transform below,
/// recorded with the code of `main` at e6008c9 (M7a merged) on 2026-11-04.
const MAIN_FINGERPRINTS: [(Thermostat, Transform, u64); 8] = [
    (Thermostat::Off, Transform::Matrix, 0x5bdc_15f6_c9b1_9edd),
    (Thermostat::Off, Transform::Fft, 0x381f_8eb9_1305_83b9),
    (
        Thermostat::PileL { tau0: Some(0.7) },
        Transform::Matrix,
        0x8795_7d11_992a_cc28,
    ),
    (
        Thermostat::PileL { tau0: Some(0.7) },
        Transform::Fft,
        0x90d8_6ead_5b5a_dc0c,
    ),
    (
        Thermostat::PileL { tau0: None },
        Transform::Matrix,
        0x4212_3693_463c_39c8,
    ),
    (
        Thermostat::PileL { tau0: None },
        Transform::Fft,
        0x9342_0f47_03e2_b873,
    ),
    (
        Thermostat::Trpmd { lambda: 0.5 },
        Transform::Matrix,
        0x97bc_ad8a_4de8_c84e,
    ),
    (
        Thermostat::Trpmd { lambda: 0.5 },
        Transform::Fft,
        0x9c9b_2736_227a_d222,
    ),
];

/// Y-6 ("Without Ω nothing changes"), D-3 (MILESTONE_7 M7b decision 3: "Without Ω every M7a rule
/// and result is unchanged, bitwise"): an integrator built without Ω produces, for every
/// thermostat and transform, the trajectory (positions, momenta, forces, energies), redraw and RNG
/// position recorded with `main` at e6008c9, bitwise.
#[test]
fn without_omega_bitwise_as_main() {
    for (th, transform, want) in MAIN_FINGERPRINTS {
        let ig = RingPolymerIntegrator::with_transform(CLUSTER_DT, CLUSTER_BETA, 4, th, transform)
            .unwrap();
        let mut sim = RingPolymerSim::new(cluster_state(4, 71), None, ig, Box::new(lj())).unwrap();
        let mut rng = Rng::from_seed(72);
        for _ in 0..40 {
            sim.step(&mut rng).unwrap();
        }
        sim.resample_momenta(&mut rng);
        for _ in 0..10 {
            sim.step(&mut rng).unwrap();
        }
        let got = fingerprint(&sim, &rng);
        assert_eq!(got, want, "{th:?}, {transform:?}");
    }
}

/// Y-6, RP-6 ("P = 1: s = [1], Ω has no effect"): with one bead, a simulation with Ω and one
/// without, from the same state and RNG seed, are bitwise identical (state, forces, energies, RNG
/// position) over 30 steps, a `resample_momenta` and 10 more steps, for `Off`, PILE-L with τ_0 and
/// TRPMD.
#[test]
fn one_bead_omega_has_no_effect() {
    for th in [
        Thermostat::Off,
        Thermostat::PileL { tau0: Some(0.7) },
        Thermostat::Trpmd { lambda: 0.5 },
    ] {
        let build = |omega: Option<f64>| {
            let mut ig = RingPolymerIntegrator::new(CLUSTER_DT, CLUSTER_BETA, 1, th).unwrap();
            if let Some(w) = omega {
                ig = ig.with_adiabatic_frequency(w).unwrap();
            }
            RingPolymerSim::new(cluster_state(1, 73), None, ig, Box::new(lj())).unwrap()
        };
        let (mut a, mut b) = (build(Some(5.0)), build(None));
        let (mut ra, mut rb) = (Rng::from_seed(74), Rng::from_seed(74));
        for s in 1..=40 {
            if s == 31 {
                a.resample_momenta(&mut ra);
                b.resample_momenta(&mut rb);
            }
            a.step(&mut ra).unwrap();
            b.step(&mut rb).unwrap();
            assert_same_sim(&a, &b, &format!("{th:?}, step {s}"));
        }
        assert_eq!(ra.snapshot(), rb.snapshot(), "{th:?}");
    }
}
