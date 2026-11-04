//! Ring-polymer dynamics limits (MILESTONE_7 required tests 4 and 5; SPEC I-1, I-2, I-4, T-3,
//! T-4, Y-1).
//!
//! Expectations: `VelocityVerlet` (merged M3, I-4) for P = 1, and the exact discrete
//! velocity-Verlet map of a harmonic oscillator, written out in the test, for the centroid of a
//! harmonic ring polymer.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod m7_common;

use common::max_rel_diff;
use m7_common::{CLUSTER_BETA, CLUSTER_DT, centroid_of, cluster_state, lj};
use pimd_core::{Beads, Rng};
use pimd_integrators::{
    RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat, VelocityVerlet,
};
use pimd_potentials::{Harmonic, Potential};

/// I-1, I-2, I-4, T-3 (MILESTONE_7 "P = 1": RPMD at P = 1 is classical NVE MD):
/// `RingPolymerSim` with P = 1 and `Thermostat::Off`, and `VelocityVerlet`, from the same state
/// of the 8-atom LJ cluster, agree in positions and momenta to 1e-10 relative (floor 1) after
/// every one of 200 steps. (With one bead the free propagation is the drift q + t p/m, so BAOAB
/// without O is velocity Verlet up to rounding of the two half drifts.)
#[test]
fn one_bead_rpmd_is_velocity_verlet() {
    let s0 = cluster_state(1, 51);
    let ig = RingPolymerIntegrator::new(CLUSTER_DT, CLUSTER_BETA, 1, Thermostat::Off).unwrap();
    let mut sim = RingPolymerSim::new(s0.clone(), None, ig, Box::new(lj())).unwrap();
    let mut pot = lj();
    let mut forces = pot.evaluate(&s0.q, None, None).unwrap().forces;
    let mut st = s0;
    let vv = VelocityVerlet::new(CLUSTER_DT).unwrap();
    let mut rng = Rng::from_seed(1);
    for s in 1..=200 {
        sim.step(&mut rng).unwrap();
        vv.step(&mut st, &mut forces, &mut pot, None).unwrap();
        let dq = max_rel_diff(&sim.state().q, &st.q, 1.0);
        let dp = max_rel_diff(&sim.state().p, &st.p, 1.0);
        assert!(dq <= 1e-10 && dp <= 1e-10, "step {s}: dq {dq:e}, dp {dp:e}");
    }
}

/// The velocity-Verlet map for a harmonic oscillator of frequency ω, mass m, step dt, applied to
/// (q, p) per component: p ← p − (dt/2) m ω² q; q ← q + dt p/m; p ← p − (dt/2) m ω² q.
fn vv_harmonic(q: &mut [f64; 3], p: &mut [f64; 3], m: f64, w: f64, dt: f64) {
    let k = m * w * w;
    for d in 0..3 {
        p[d] -= 0.5 * dt * k * q[d];
        q[d] += dt * p[d] / m;
        p[d] -= 0.5 * dt * k * q[d];
    }
}

/// The harmonic-trap test system: two atoms (m = 1 and 3), ω = 1.3, P = 8, β = 4, dt = 0.05,
/// random bead positions and thermal momenta.
fn harmonic_sim(th: Thermostat) -> (RingPolymerSim, Vec<f64>, f64, f64) {
    let (p, beta, dt, w) = (8, 4.0, 0.05, 1.3);
    let masses = vec![1.0, 3.0];
    let mut sm = common::SplitMix64(52);
    let q = common::random_beads(&mut sm, p, 2, -1.0, 1.0);
    let mut rng = Rng::from_seed(52);
    let mom = common::thermal_momenta(&mut rng, p, &masses, beta);
    let state = RingState {
        q,
        p: mom,
        masses: masses.clone(),
    };
    let ig = RingPolymerIntegrator::new(dt, beta, p, th).unwrap();
    let pot = Harmonic::new(masses.clone(), w).unwrap();
    (
        RingPolymerSim::new(state, None, ig, Box::new(pot)).unwrap(),
        masses,
        w,
        dt,
    )
}

/// Steps `sim` 1 000 times and checks the centroid against the VV map for ω applied to
/// (q̄(0), p̄(0)), p̄ = (1/P) Σ_j p_j (the centroid force of the trap is −m ω² q̄), to 1e-12
/// relative (floor 1) after every step.
fn check_centroid_follows_vv(mut sim: RingPolymerSim, masses: &[f64], w: f64, dt: f64) {
    let mut qc = sim.state().q.centroid();
    let mut pc = centroid_of(&sim.state().p);
    let mut rng = Rng::from_seed(53);
    for s in 1..=1000 {
        sim.step(&mut rng).unwrap();
        for (i, &m) in masses.iter().enumerate() {
            vv_harmonic(&mut qc[i], &mut pc[i], m, w, dt);
        }
        let got_q = sim.state().q.centroid();
        let got_p = centroid_of(&sim.state().p);
        let flat = |v: &[[f64; 3]]| {
            Beads::from_vec(1, v.len(), v.iter().flatten().copied().collect()).unwrap()
        };
        let dq = max_rel_diff(&flat(&got_q), &flat(&qc), 1.0);
        let dp = max_rel_diff(&flat(&got_p), &flat(&pc), 1.0);
        assert!(
            dq <= 1e-12 && dp <= 1e-12,
            "step {s}: centroid dq {dq:e}, dp {dp:e}"
        );
    }
}

/// I-1, I-2, T-3: for `Harmonic` the centroid decouples from the internal modes; under
/// `Thermostat::Off` (P = 8) it follows the exact discrete velocity-Verlet map for ω applied to
/// (q̄(0), p̄(0)), to 1e-12 over 1 000 steps.
#[test]
fn harmonic_centroid_is_velocity_verlet_under_off() {
    let (sim, masses, w, dt) = harmonic_sim(Thermostat::Off);
    check_centroid_follows_vv(sim, &masses, w, dt);
}

/// T-4 (γ_0 = 0: internal-mode noise does not reach the centroid), I-1, I-2: the same holds
/// under `Thermostat::Trpmd { lambda: 0.5 }` and `{ lambda: 1.0 }`: the centroid follows the
/// deterministic VV map to 1e-12 over 1 000 steps although the internal modes are thermostatted.
#[test]
fn harmonic_centroid_is_velocity_verlet_under_trpmd() {
    for lambda in [0.5, 1.0] {
        let (sim, masses, w, dt) = harmonic_sim(Thermostat::Trpmd { lambda });
        check_centroid_follows_vv(sim, &masses, w, dt);
    }
}
