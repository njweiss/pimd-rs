//! Tests for `RingPolymerSim::set_thermostat` and `RingPolymerSim::resample_momenta` (SPEC Y-1,
//! Y-2, I-3, D-2, D-4; MILESTONE_7 required tests 2 and 3).
//!
//! Expectations: a simulation built directly with the target thermostat from the same state
//! (`RingPolymerSim::from_parts`, merged M4 code, no evaluation) and a restored RNG; the Y-2 draw
//! written out in the test (√(m_i·P/β)·ξ, atom i outer, bead j, dimension d inner) from a forked
//! RNG; evaluation counts from a counting `Potential`.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod m7_common;

use m7_common::{
    CLUSTER_BETA, CLUSTER_DT, CLUSTER_MASSES, assert_same_sim, cluster_sim, counted_cluster_sim,
    fork, lj,
};
use pimd_core::{Beads, Rng, Transform};
use pimd_integrators::{
    IntegratorError, RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat,
};

const THERMOSTATS: [Thermostat; 5] = [
    Thermostat::Off,
    Thermostat::PileL { tau0: Some(0.7) },
    Thermostat::PileL { tau0: None },
    Thermostat::Trpmd { lambda: 0.5 },
    Thermostat::Trpmd { lambda: 1.7 },
];

/// A simulation built with `th` (and `transform`) directly from `sim`'s current state, forces and
/// energies, without evaluating.
fn rebuilt(sim: &RingPolymerSim, th: Thermostat, transform: Transform, p: usize) -> RingPolymerSim {
    let ig =
        RingPolymerIntegrator::with_transform(CLUSTER_DT, CLUSTER_BETA, p, th, transform).unwrap();
    RingPolymerSim::from_parts(
        sim.state().clone(),
        None,
        ig,
        Box::new(lj()),
        sim.forces().clone(),
        sim.energies().to_vec(),
    )
    .unwrap()
}

/// Y-1, I-3, D-2: for every pair (X, Y) of thermostats (Off, PileL with and without τ_0, Trpmd at
/// two λ), a cluster sim (P = 4) built with X, stepped 7 times, switched to Y and stepped 25 more
/// times equals, bitwise (state, forces, energies, RNG position), a sim built with Y from the
/// state at the switch and stepped 25 times with a restored copy of the RNG. Both transforms
/// (the switch keeps the transform, dt, β and P).
#[test]
fn switched_sim_equals_sim_built_with_target() {
    for transform in [Transform::Matrix, Transform::Fft] {
        for (a, &from) in THERMOSTATS.iter().enumerate() {
            for (b, &to) in THERMOSTATS.iter().enumerate() {
                let p = 4;
                let ig = RingPolymerIntegrator::with_transform(
                    CLUSTER_DT,
                    CLUSTER_BETA,
                    p,
                    from,
                    transform,
                )
                .unwrap();
                let mut sim =
                    RingPolymerSim::new(m7_common::cluster_state(p, 21), None, ig, Box::new(lj()))
                        .unwrap();
                let mut rng = Rng::from_seed(100 + (a * 10 + b) as u64);
                for _ in 0..7 {
                    sim.step(&mut rng).unwrap();
                }
                let mut want = rebuilt(&sim, to, transform, p);
                let mut rng_want = fork(&rng);
                sim.set_thermostat(to).unwrap();
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

/// Y-1, I-3, D-4: `set_thermostat` performs no `evaluate` (counting potential), draws nothing (the
/// caller's RNG is not even passed), and leaves state, forces and energies bitwise unchanged;
/// the next step performs exactly one evaluation (I-3: the forces at the end of one phase are
/// reused by the next).
#[test]
fn switch_does_not_evaluate_or_change_state() {
    for &to in &THERMOSTATS {
        let (mut sim, calls) = counted_cluster_sim(4, 22, Thermostat::PileL { tau0: Some(0.7) });
        let mut rng = Rng::from_seed(7);
        sim.step(&mut rng).unwrap();
        let (st, f, e) = (
            sim.state().clone(),
            sim.forces().clone(),
            sim.energies().to_vec(),
        );
        let n0 = calls.get();
        let r0 = rng.snapshot();
        sim.set_thermostat(to).unwrap();
        assert_eq!(calls.get(), n0, "{to:?}: set_thermostat evaluated");
        assert_eq!(sim.state(), &st);
        assert_eq!(sim.forces(), &f);
        assert_eq!(sim.energies(), e.as_slice());
        assert_eq!(rng.snapshot(), r0);
        sim.step(&mut rng).unwrap();
        assert_eq!(calls.get(), n0 + 1, "{to:?}: one evaluation per step");
    }
}

/// Y-1, T-4, T-2: a rejected thermostat (λ ≤ 0, NaN, ±∞; τ_0 ≤ 0, NaN, +∞) is
/// `IntegratorError::InvalidValue`, and the simulation is unchanged: it keeps its thermostat, so
/// it then steps bitwise like an identical sim that was never asked to switch (PILE-L τ_0 = 0.7,
/// 10 steps, same seed), and nothing was evaluated by the failed call.
#[test]
fn rejected_thermostat_leaves_sim_unchanged() {
    let bad = [
        Thermostat::Trpmd { lambda: 0.0 },
        Thermostat::Trpmd { lambda: -1.0 },
        Thermostat::Trpmd { lambda: f64::NAN },
        Thermostat::Trpmd {
            lambda: f64::INFINITY,
        },
        Thermostat::Trpmd {
            lambda: f64::NEG_INFINITY,
        },
        Thermostat::PileL { tau0: Some(0.0) },
        Thermostat::PileL { tau0: Some(-2.0) },
        Thermostat::PileL {
            tau0: Some(f64::NAN),
        },
        Thermostat::PileL {
            tau0: Some(f64::INFINITY),
        },
    ];
    for th in bad {
        let start = Thermostat::PileL { tau0: Some(0.7) };
        let (mut a, calls) = counted_cluster_sim(4, 23, start);
        let mut b = cluster_sim(4, 23, start);
        let n0 = calls.get();
        let r = a.set_thermostat(th);
        assert!(
            matches!(r, Err(IntegratorError::InvalidValue { .. })),
            "{th:?}: {r:?}"
        );
        assert_eq!(calls.get(), n0);
        assert_same_sim(&a, &b, &format!("{th:?} right after the failed switch"));
        let (mut ra, mut rb) = (Rng::from_seed(9), Rng::from_seed(9));
        for s in 1..=10 {
            a.step(&mut ra).unwrap();
            b.step(&mut rb).unwrap();
            assert_same_sim(&a, &b, &format!("{th:?}, step {s} after the failed switch"));
        }
        assert_eq!(ra.snapshot(), rb.snapshot());
    }
}

/// Y-2, D-2, D-4: `resample_momenta` sets every bead momentum to √(m_i·P/β)·ξ (evaluated as
/// `((m_i·P)/β).sqrt()·ξ`, left to right as Y-2 and the initial draw of MILESTONE_4 write it),
/// with ξ drawn from a forked RNG atom i outermost, then bead j, then dimension d, bitwise;
/// exactly P·N·3 draws (the streams agree afterwards); positions, forces, energies and masses
/// untouched bitwise and no evaluation. Distinct masses per atom (8-atom cluster), P = 1, 3, 4.
#[test]
fn resample_is_y2_draw_in_order() {
    for p in [1, 3, 4] {
        let (mut sim, calls) = counted_cluster_sim(p, 24, Thermostat::PileL { tau0: Some(0.7) });
        let mut rng = Rng::from_seed(31 + p as u64);
        for _ in 0..3 {
            sim.step(&mut rng).unwrap();
        }
        let (q0, f0, e0) = (
            sim.state().q.clone(),
            sim.forces().clone(),
            sim.energies().to_vec(),
        );
        let n0 = calls.get();
        let mut xi = fork(&rng);
        sim.resample_momenta(&mut rng);

        let n = CLUSTER_MASSES.len();
        let mut want = Beads::zeros(p, n);
        for (i, &m) in CLUSTER_MASSES.iter().enumerate() {
            let sd = (m * p as f64 / CLUSTER_BETA).sqrt();
            for j in 0..p {
                let v = [sd * xi.normal(), sd * xi.normal(), sd * xi.normal()];
                want.set(j, i, v);
            }
        }
        assert_eq!(sim.state().p, want, "P = {p}: momenta are not the Y-2 draw");
        assert_eq!(rng.snapshot(), xi.snapshot(), "P = {p}: not P·N·3 draws");
        assert_eq!(sim.state().q, q0);
        assert_eq!(sim.state().masses, CLUSTER_MASSES.to_vec());
        assert_eq!(sim.forces(), &f0);
        assert_eq!(sim.energies(), e0.as_slice());
        assert_eq!(calls.get(), n0, "resample_momenta evaluated");
    }
}

/// Y-2 (statistics of the draw, independent of the order): over 4 000 redraws of one atom
/// (m = 3, P = 4, β = 2), the sample variance of every bead momentum component is m·P/β = 6
/// within 5σ (σ of a sample variance of n normals: var·√(2/(n − 1))), and the bead-bead
/// covariance is zero within 5σ (σ = var/√n). Catches a variance m/β (= 1.5) or a redraw of the
/// centroid only.
#[test]
fn resample_variance_is_m_p_over_beta() {
    let (p, m, beta) = (4usize, 3.0, 2.0);
    let ig = RingPolymerIntegrator::new(0.01, beta, p, Thermostat::Off).unwrap();
    let state = RingState {
        q: Beads::zeros(p, 1),
        p: Beads::zeros(p, 1),
        masses: vec![m],
    };
    let pot = pimd_potentials::Harmonic::new(vec![m], 1.0).unwrap();
    let mut sim = RingPolymerSim::new(state, None, ig, Box::new(pot)).unwrap();
    let mut rng = Rng::from_seed(77);
    let n = 4000;
    let mut s2 = vec![0.0; p * 3];
    let mut cross = 0.0;
    for _ in 0..n {
        sim.resample_momenta(&mut rng);
        let b = &sim.state().p;
        for j in 0..p {
            let v = b.get(j, 0);
            for d in 0..3 {
                s2[3 * j + d] += v[d] * v[d];
            }
        }
        cross += b.get(0, 0)[0] * b.get(1, 0)[0];
    }
    let var = m * p as f64 / beta;
    let nf = n as f64;
    let sig = var * (2.0 / (nf - 1.0)).sqrt();
    for (k, s) in s2.iter().enumerate() {
        let got = s / nf;
        assert!(
            (got - var).abs() <= 5.0 * sig,
            "component {k}: variance {got} vs {var} ± {sig}"
        );
    }
    let c = cross / nf;
    assert!(c.abs() <= 5.0 * var / nf.sqrt(), "bead covariance {c}");
}
