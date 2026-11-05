//! The harmonic centroid under PA-CMD (MILESTONE_7 M7b required test 5; SPEC Y-6, Y-7, I-1, I-2,
//! T-2, T-4).
//!
//! Expectation: the exact discrete velocity-Verlet map of a harmonic oscillator, written out in
//! the test, applied to the centroid (q̄(0), p̄(0)). Y-6 leaves the centroid's mass physical
//! (s_0 = 1) and Y-7 keeps it unthermostatted during segments (γ_0 = 0), so the centroid of a
//! harmonic ring polymer still decouples from the internal modes and follows that map, whatever
//! the internal modes do.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod m7_common;

use common::max_rel_diff;
use m7_common::centroid_of;
use pimd_core::{Beads, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::Harmonic;

/// The velocity-Verlet map for a harmonic oscillator of frequency ω, mass m, step dt, per
/// component: p ← p − (dt/2) m ω² q; q ← q + dt p/m; p ← p − (dt/2) m ω² q.
fn vv_harmonic(q: &mut [f64; 3], p: &mut [f64; 3], m: f64, w: f64, dt: f64) {
    let k = m * w * w;
    for d in 0..3 {
        p[d] -= 0.5 * dt * k * q[d];
        q[d] += dt * p[d] / m;
        p[d] -= 0.5 * dt * k * q[d];
    }
}

/// Two atoms (m = 1 and 3) in a trap ω = 1.3, P = 8, β = 4, dt = 0.05, random bead positions,
/// thermal momenta; integrator with thermostat `th` and adiabatic frequency `omega`.
fn harmonic_sim(th: Thermostat, omega: f64) -> (RingPolymerSim, Vec<f64>, f64, f64) {
    let (p, beta, dt, w) = (8, 4.0, 0.05, 1.3);
    let masses = vec![1.0, 3.0];
    let mut sm = common::SplitMix64(152);
    let q = common::random_beads(&mut sm, p, 2, -1.0, 1.0);
    let mut rng = Rng::from_seed(152);
    let mom = common::thermal_momenta(&mut rng, p, &masses, beta);
    let state = RingState {
        q,
        p: mom,
        masses: masses.clone(),
    };
    let ig = RingPolymerIntegrator::new(dt, beta, p, th)
        .unwrap()
        .with_adiabatic_frequency(omega)
        .unwrap();
    let pot = Harmonic::new(masses.clone(), w).unwrap();
    (
        RingPolymerSim::new(state, None, ig, Box::new(pot)).unwrap(),
        masses,
        w,
        dt,
    )
}

/// Steps `sim` 1 000 times and checks the centroid against the VV map for ω applied to
/// (q̄(0), p̄(0)), p̄ = (1/P) Σ_j p_j, to 1e-12 relative (floor 1) after every step.
fn check_centroid_follows_vv(mut sim: RingPolymerSim, masses: &[f64], w: f64, dt: f64, what: &str) {
    let mut qc = sim.state().q.centroid();
    let mut pc = centroid_of(&sim.state().p);
    let mut rng = Rng::from_seed(153);
    let flat = |v: &[[f64; 3]]| {
        Beads::from_vec(1, v.len(), v.iter().flatten().copied().collect()).unwrap()
    };
    for s in 1..=1000 {
        sim.step(&mut rng).unwrap();
        for (i, &m) in masses.iter().enumerate() {
            vv_harmonic(&mut qc[i], &mut pc[i], m, w, dt);
        }
        let dq = max_rel_diff(&flat(&sim.state().q.centroid()), &flat(&qc), 1.0);
        let dp = max_rel_diff(&flat(&centroid_of(&sim.state().p)), &flat(&pc), 1.0);
        assert!(
            dq <= 1e-12 && dp <= 1e-12,
            "{what}, step {s}: centroid dq {dq:e}, dp {dp:e}"
        );
    }
}

/// Y-6, Y-7, T-4 with Y-6 (γ_0 = 0: internal-mode friction 2λΩ and noise do not reach the
/// centroid), I-1, I-2: with Ω = 6 (Ω·dt = 0.3) and 20, under `Trpmd { lambda: 1.0 }` (the M7b test
/// value) and `{ lambda: 0.5 }`, the centroid follows the deterministic VV map for ω to 1e-12 over
/// 1 000 steps. Also under `Off` and `PileL { tau0: None }` with Ω (γ_0 = 0 too). (Ω applied to
/// the centroid's mass or frequency breaks this.)
#[test]
fn harmonic_centroid_is_velocity_verlet_with_omega() {
    for omega in [6.0, 20.0] {
        for th in [
            Thermostat::Trpmd { lambda: 1.0 },
            Thermostat::Trpmd { lambda: 0.5 },
            Thermostat::Off,
            Thermostat::PileL { tau0: None },
        ] {
            let (sim, masses, w, dt) = harmonic_sim(th, omega);
            check_centroid_follows_vv(sim, &masses, w, dt, &format!("Ω = {omega}, {th:?}"));
        }
    }
}
