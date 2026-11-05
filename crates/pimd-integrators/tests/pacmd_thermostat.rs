//! The O step and `set_thermostat` with PA-CMD dynamical masses (MILESTONE_7 M7b required test 2;
//! SPEC T-1, T-2, T-4 with Y-6; Y-1, D-2, D-4).
//!
//! Expectations: c_k and the T-1 update with Y-6 masses written out in the test's own RP-4 basis
//! (`m7b_common::ref_c_dyn`, `ref_o_step_dyn`) with a same-seed `Rng`; for `set_thermostat`, a
//! simulation built directly (`RingPolymerSim::from_parts`, merged M4 code) with Ω and the target
//! thermostat.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod m7_common;
mod m7b_common;

use common::{SplitMix64, max_rel_diff, random_beads, ref_matrix, to_beads, to_modes};
use m7_common::{CLUSTER_BETA, CLUSTER_DT, assert_same_sim, fork, lj};
use m7b_common::{pacmd_cluster_sim, ref_c_dyn, ref_mass_factors, ref_o_step_dyn};
use pimd_core::{Rng, Transform};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};

const MASSES: [f64; 3] = [1.0, 4.0, 1836.0];

/// The thermostats of the O-step test with their Y-6 friction rule (γ_0, scale of Ω for k ≥ 1):
/// PILE-L γ_0 = 1/τ_0 or 0, γ_k = 2Ω (T-2); TRPMD γ_0 = 0, γ_k = 2λΩ (T-4).
fn cases() -> Vec<(Thermostat, f64, f64)> {
    vec![
        (Thermostat::PileL { tau0: Some(0.8) }, 1.0 / 0.8, 2.0),
        (Thermostat::PileL { tau0: None }, 0.0, 2.0),
        (Thermostat::Trpmd { lambda: 0.5 }, 0.0, 1.0),
        (Thermostat::Trpmd { lambda: 1.0 }, 0.0, 2.0),
        (Thermostat::Trpmd { lambda: 2.0 }, 0.0, 4.0),
    ]
}

/// T-1, T-2, T-4 with Y-6, D-2, D-4: one O step (`thermostat_step`) with Ω on a state with known
/// normal-mode momenta p̃ (set in the test's RP-4 basis): every mode becomes
/// c_k p̃_k + sqrt((1 − c_k²)·m_i·s_k·P/β)·ξ with s_k = (ω_k/Ω)², c_k = exp(−γ_k dt), γ_k = 2Ω
/// (PILE-L) or 2λΩ (TRPMD) for k ≥ 1, γ_0 = 1/τ_0 or 0, ξ drawn from a same-seed `Rng` in the
/// order k, atom i, d; exactly P·N·3 draws. Agreement 1e-12 relative (floor 1); P = 2, 3, 4, 6, 8;
/// both transforms. Then, mode by mode, c_k is recovered as (p̃'_k − amp_k ξ)/p̃_k and compared
/// with exp(−γ_k dt) (rules out friction from ω_k, a half-step exp(−γ dt/2), or Ω on the
/// centroid), to 1e-12; and a second step on p̃ = 0 must give p̃'_k = sqrt((1 − c_k²)·m_i·s_k·P/β)·ξ
/// to 1e-12 of the bead-momentum scale √(m_i P/β) (rules out the physical m_i in the noise).
#[test]
fn o_step_uses_dynamical_frequency_and_mass() {
    let mut sm = SplitMix64(81);
    let (beta, dt, omega) = (1.3, 0.07, 11.0);
    for p in [2, 3, 4, 6, 8] {
        let mat = ref_matrix(p);
        let s = ref_mass_factors(p, beta, omega);
        let pf = p as f64;
        for transform in [Transform::Matrix, Transform::Fft] {
            for (th, gamma0, scale) in cases() {
                let ig = RingPolymerIntegrator::with_transform(dt, beta, p, th, transform)
                    .unwrap()
                    .with_adiabatic_frequency(omega)
                    .unwrap();
                let pt0 = random_beads(&mut sm, p, MASSES.len(), 0.5, 3.0);
                let s0 = RingState {
                    q: random_beads(&mut sm, p, MASSES.len(), -1.0, 1.0),
                    p: to_beads(&mat, &pt0),
                    masses: MASSES.to_vec(),
                };
                let seed = 1000 + p as u64;
                let what = format!("P = {p}, {transform:?}, {th:?}");
                let c = ref_c_dyn(p, omega, dt, gamma0, scale);

                let mut st = s0.clone();
                let mut rng = Rng::from_seed(seed);
                ig.thermostat_step(&mut st, &mut rng);
                let mut rng_ref = Rng::from_seed(seed);
                let want = ref_o_step_dyn(&s0.p, &MASSES, beta, &c, &s, &mut rng_ref);
                let d = max_rel_diff(&st.p, &want, 1.0);
                assert!(d <= 1e-12, "{what}: rel diff {d:e}");
                assert_eq!(st.q, s0.q, "{what}: the O step touched positions");
                assert_eq!(rng.normal(), rng_ref.normal(), "{what}: not P·N·3 draws");

                // c_k recovered mode by mode.
                let pt1 = to_modes(&mat, &st.p);
                let mut xi = Rng::from_seed(seed);
                for k in 0..p {
                    for (i, &m) in MASSES.iter().enumerate() {
                        let amp = ((1.0 - c[k] * c[k]) * m * s[k] * pf / beta).sqrt();
                        let (a0, a1) = (pt0.get(k, i), pt1.get(k, i));
                        for dd in 0..3 {
                            let got = (a1[dd] - amp * xi.normal()) / a0[dd];
                            assert!(
                                (got - c[k]).abs() <= 1e-12,
                                "{what}, k = {k}, atom {i}: c_k = {got}, want {}",
                                c[k]
                            );
                        }
                    }
                }

                // Noise amplitude recovered from p̃ = 0.
                let mut z = s0.clone();
                z.p = pimd_core::Beads::zeros(p, MASSES.len());
                let mut rng = Rng::from_seed(seed + 50);
                ig.thermostat_step(&mut z, &mut rng);
                let zt = to_modes(&mat, &z.p);
                let mut xi = Rng::from_seed(seed + 50);
                for k in 0..p {
                    for (i, &m) in MASSES.iter().enumerate() {
                        let want = ((1.0 - c[k] * c[k]) * m * s[k] * pf / beta).sqrt();
                        // Absolute floor: the bead-momentum scale √(m_i P/β) (transform rounding).
                        let floor = (m * pf / beta).sqrt();
                        let a = zt.get(k, i);
                        for dd in 0..3 {
                            let x = xi.normal();
                            assert!(
                                (a[dd] - want * x).abs() <= 1e-12 * floor,
                                "{what}, k = {k}, atom {i}: p̃' = {}, want {want}·ξ = {}",
                                a[dd],
                                want * x
                            );
                        }
                    }
                }
            }
        }
    }
}

/// T-4 with Y-6: under TRPMD with Ω the centroid normal-mode momentum is untouched (c_0 = 1, zero
/// noise) while the k = 0 normals are still drawn (P·N·3 per step, T-4).
#[test]
fn trpmd_with_omega_leaves_the_centroid() {
    let (beta, dt, omega, p) = (2.0, 0.1, 7.0, 6);
    let mat = ref_matrix(p);
    let ig = RingPolymerIntegrator::new(dt, beta, p, Thermostat::Trpmd { lambda: 0.5 })
        .unwrap()
        .with_adiabatic_frequency(omega)
        .unwrap();
    let mut sm = SplitMix64(82);
    let pt0 = random_beads(&mut sm, p, MASSES.len(), 0.5, 3.0);
    let mut st = RingState {
        q: random_beads(&mut sm, p, MASSES.len(), -1.0, 1.0),
        p: to_beads(&mat, &pt0),
        masses: MASSES.to_vec(),
    };
    let mut rng = Rng::from_seed(9);
    ig.thermostat_step(&mut st, &mut rng);
    let pt1 = to_modes(&mat, &st.p);
    for i in 0..MASSES.len() {
        let (a0, a1) = (pt0.get(0, i), pt1.get(0, i));
        for d in 0..3 {
            assert!(
                (a1[d] - a0[d]).abs() <= 1e-12 * a0[d].abs().max(1.0),
                "atom {i}: centroid momentum changed"
            );
        }
    }
    let mut want = Rng::from_seed(9);
    for _ in 0..p * MASSES.len() * 3 {
        want.normal();
    }
    assert_eq!(rng.snapshot(), want.snapshot(), "not P·N·3 draws");
}

const THERMOSTATS: [Thermostat; 5] = [
    Thermostat::Off,
    Thermostat::PileL { tau0: Some(0.7) },
    Thermostat::PileL { tau0: None },
    Thermostat::Trpmd { lambda: 0.5 },
    Thermostat::Trpmd { lambda: 1.0 },
];

/// Y-1, Y-6 (MILESTONE_7 M7b decision 2: "`set_thermostat` keeps it"), I-3, D-2: for every pair
/// (X, Y) of thermostats, a cluster sim (P = 4) built with Ω = 4 and X, stepped 7 times, switched
/// to Y and stepped 25 more times equals, bitwise (state, forces, energies, RNG position), a sim
/// built with Ω = 4 and Y from the state at the switch (no evaluation) and stepped 25 times with a
/// restored copy of the RNG; after the switch `integrator().adiabatic_frequency()` is still Ω and
/// the mass factors are unchanged. Both transforms.
#[test]
fn set_thermostat_keeps_omega() {
    let (p, omega) = (4, 4.0);
    for transform in [Transform::Matrix, Transform::Fft] {
        for (a, &from) in THERMOSTATS.iter().enumerate() {
            for (b, &to) in THERMOSTATS.iter().enumerate() {
                let mut sim = pacmd_cluster_sim(p, 91, from, transform, omega);
                let s_before = sim.integrator().mass_factors().to_vec();
                let mut rng = Rng::from_seed(200 + (a * 10 + b) as u64);
                for _ in 0..7 {
                    sim.step(&mut rng).unwrap();
                }
                let ig = RingPolymerIntegrator::with_transform(
                    CLUSTER_DT,
                    CLUSTER_BETA,
                    p,
                    to,
                    transform,
                )
                .unwrap()
                .with_adiabatic_frequency(omega)
                .unwrap();
                let mut want = RingPolymerSim::from_parts(
                    sim.state().clone(),
                    None,
                    ig,
                    Box::new(lj()),
                    sim.forces().clone(),
                    sim.energies().to_vec(),
                )
                .unwrap();
                let mut rng_want = fork(&rng);
                sim.set_thermostat(to).unwrap();
                assert_eq!(sim.integrator().adiabatic_frequency(), Some(omega));
                assert_eq!(sim.integrator().mass_factors(), s_before.as_slice());
                for s in 1..=25 {
                    sim.step(&mut rng).unwrap();
                    want.step(&mut rng_want).unwrap();
                    assert_same_sim(
                        &sim,
                        &want,
                        &format!("{transform:?} {from:?} -> {to:?}, step {s}"),
                    );
                }
                assert_eq!(rng.snapshot(), rng_want.snapshot(), "{from:?} -> {to:?}");
            }
        }
    }
}
