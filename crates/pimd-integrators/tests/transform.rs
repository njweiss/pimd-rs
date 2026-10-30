//! Tests for `RingPolymerIntegrator::with_transform` (SPEC RP-4, T-1, I-1, I-2, D-3;
//! MILESTONE_4 "Normal-mode transform"): the FFT setting gives the same dynamics and the same
//! thermostat draws as the matrix setting, to rounding; `Matrix` is bitwise `new`.
//!
//! References are the SPEC-formula helpers in `common` (RP-4 matrix, I-2 propagator, T-1 O step),
//! and the `Matrix` integrator for the bitwise checks. Tolerances: the FFT and matrix transforms
//! agree to ~1e-15 (pimd-core `normal_modes_fft`); a harmonic trajectory is not chaotic, so the
//! difference grows at most linearly over 50 steps; 1e-11 relative (floor 1e-3 of the typical
//! magnitude) leaves a wide margin and still catches any wrong column or factor (O(1) errors).

mod common;

use common::{
    SplitMix64, max_rel_diff, random_beads, ref_free_propagate, ref_o_step, thermal_momenta,
};
use pimd_core::normal_modes::Transform;
use pimd_core::{CoreError, Rng};
use pimd_integrators::{
    IntegratorError, RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat,
};
use pimd_potentials::Harmonic;

const PS: [usize; 7] = [1, 2, 3, 4, 8, 16, 32];
const BETA: f64 = 3.0;
const DT: f64 = 0.05;
const TAU0: f64 = 2.0;
const OMEGA: f64 = 1.3;

fn masses(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + 0.5 * i as f64).collect()
}

fn state(p: usize, n: usize, seed: u64) -> RingState {
    let m = masses(n);
    let mut sm = SplitMix64(seed);
    let mut rng = Rng::from_seed(seed);
    RingState {
        q: random_beads(&mut sm, p, n, -1.0, 1.0),
        p: thermal_momenta(&mut rng, p, &m, BETA),
        masses: m,
    }
}

fn pile() -> Thermostat {
    Thermostat::PileL { tau0: Some(TAU0) }
}

fn sim(p: usize, transform: Transform, st: RingState) -> RingPolymerSim {
    let n = st.masses.len();
    let ig = RingPolymerIntegrator::with_transform(DT, BETA, p, pile(), transform).unwrap();
    let pot = Harmonic::new(masses(n), OMEGA).unwrap();
    RingPolymerSim::new(st, None, ig, Box::new(pot)).unwrap()
}

/// RP-4: the integrator's `modes()` carry the requested setting; `new` gives `Matrix`.
#[test]
fn modes_report_setting() {
    for p in PS {
        for t in [Transform::Matrix, Transform::Fft] {
            let ig = RingPolymerIntegrator::with_transform(DT, BETA, p, pile(), t).unwrap();
            assert_eq!(ig.modes().transform(), t, "P = {p}");
            assert_eq!(ig.modes().n_beads(), p);
        }
        let ig = RingPolymerIntegrator::new(DT, BETA, p, pile()).unwrap();
        assert_eq!(ig.modes().transform(), Transform::Matrix);
    }
}

/// I-1, D-3: `with_transform(.., Matrix)` is bitwise `new`: 30 PILE-L steps on a harmonic trap
/// from the same state and seed give identical states, forces and energies.
#[test]
fn matrix_setting_is_bitwise_new() {
    for p in [1usize, 3, 8] {
        let st = state(p, 4, 11 + p as u64);
        let mut a = sim(p, Transform::Matrix, st.clone());
        let ig = RingPolymerIntegrator::new(DT, BETA, p, pile()).unwrap();
        let pot = Harmonic::new(masses(4), OMEGA).unwrap();
        let mut b = RingPolymerSim::new(st, None, ig, Box::new(pot)).unwrap();
        let (mut ra, mut rb) = (Rng::from_seed(7), Rng::from_seed(7));
        for _ in 0..30 {
            a.step(&mut ra).unwrap();
            b.step(&mut rb).unwrap();
        }
        assert_eq!(a.state(), b.state(), "P = {p}");
        assert_eq!(a.forces(), b.forces());
        assert_eq!(a.energies(), b.energies());
    }
}

/// I-2, RP-4: `free_propagate` under `Fft` equals the SPEC-formula propagator to 1e-12.
#[test]
fn fft_free_propagation_matches_reference() {
    for p in PS {
        let mut st = state(p, 3, 100 + p as u64);
        let (q_ref, p_ref) = ref_free_propagate(&st.q, &st.p, &st.masses, BETA, 0.7);
        let ig =
            RingPolymerIntegrator::with_transform(DT, BETA, p, pile(), Transform::Fft).unwrap();
        ig.free_propagate(&mut st, 0.7);
        let eq = max_rel_diff(&st.q, &q_ref, 1e-3);
        let ep = max_rel_diff(&st.p, &p_ref, 1e-3);
        assert!(eq <= 1e-12 && ep <= 1e-12, "P = {p}: q {eq:e}, p {ep:e}");
    }
}

/// T-1, RP-4: under `Fft` the O step uses the same draws, in the same (k, i, d) order, as the
/// SPEC O step: the result equals `ref_o_step` with an identically seeded generator to 1e-12,
/// and both generators are left at the same position (the next draws agree bitwise).
#[test]
fn fft_thermostat_same_draws() {
    for p in PS {
        let mut st = state(p, 3, 200 + p as u64);
        let mut r_ref = Rng::from_seed(42);
        let want = ref_o_step(&st.p, &st.masses, BETA, DT, Some(TAU0), &mut r_ref);
        let ig =
            RingPolymerIntegrator::with_transform(DT, BETA, p, pile(), Transform::Fft).unwrap();
        let mut r = Rng::from_seed(42);
        ig.thermostat_step(&mut st, &mut r);
        let e = max_rel_diff(&st.p, &want, 1e-3);
        assert!(e <= 1e-12, "P = {p}: O step relative error {e:e}");
        assert_eq!(
            r.normal().to_bits(),
            r_ref.normal().to_bits(),
            "P = {p}: generator positions differ after the O step"
        );
    }
}

/// T-3: `Thermostat::Off` under `Fft` is the identity and draws nothing.
#[test]
fn fft_thermostat_off_draws_nothing() {
    let mut st = state(4, 2, 5);
    let before = st.clone();
    let ig = RingPolymerIntegrator::with_transform(DT, BETA, 4, Thermostat::Off, Transform::Fft)
        .unwrap();
    let mut r = Rng::from_seed(9);
    ig.thermostat_step(&mut st, &mut r);
    assert_eq!(st, before);
    assert_eq!(r.normal().to_bits(), Rng::from_seed(9).normal().to_bits());
}

/// I-1, T-1, RP-4: 50 BAOAB + PILE-L steps on a harmonic trap: the `Fft` and `Matrix` settings
/// give the same trajectory to 1e-11 (same draws, transforms equal to rounding) and leave the
/// generator at the same position.
#[test]
fn fft_trajectory_matches_matrix() {
    for p in PS {
        let st = state(p, 4, 300 + p as u64);
        let mut a = sim(p, Transform::Matrix, st.clone());
        let mut b = sim(p, Transform::Fft, st);
        let (mut ra, mut rb) = (Rng::from_seed(77), Rng::from_seed(77));
        for _ in 0..50 {
            a.step(&mut ra).unwrap();
            b.step(&mut rb).unwrap();
        }
        let eq = max_rel_diff(&b.state().q, &a.state().q, 1e-3);
        let ep = max_rel_diff(&b.state().p, &a.state().p, 1e-3);
        assert!(eq <= 1e-11 && ep <= 1e-11, "P = {p}: q {eq:e}, p {ep:e}");
        assert_eq!(ra.normal().to_bits(), rb.normal().to_bits(), "P = {p}");
    }
}

/// D-3 (per setting): two `Fft` simulations from the same state and seed are bitwise identical.
#[test]
fn fft_trajectory_deterministic() {
    let st = state(8, 4, 400);
    let mut a = sim(8, Transform::Fft, st.clone());
    let mut b = sim(8, Transform::Fft, st);
    let (mut ra, mut rb) = (Rng::from_seed(1), Rng::from_seed(1));
    for _ in 0..30 {
        a.step(&mut ra).unwrap();
        b.step(&mut rb).unwrap();
    }
    assert_eq!(a.state(), b.state());
}

/// I-5, RP-1: `with_transform` validates like `new` for both settings: bad dt or β →
/// `InvalidValue`; P = 0 → `Core(InvalidBeadCount(0))`.
#[test]
fn with_transform_errors() {
    for t in [Transform::Matrix, Transform::Fft] {
        for dt in [0.0, -1.0, f64::NAN] {
            assert!(matches!(
                RingPolymerIntegrator::with_transform(dt, BETA, 4, pile(), t),
                Err(IntegratorError::InvalidValue { .. })
            ));
        }
        assert!(matches!(
            RingPolymerIntegrator::with_transform(DT, -1.0, 4, pile(), t),
            Err(IntegratorError::InvalidValue { .. })
        ));
        assert!(matches!(
            RingPolymerIntegrator::with_transform(DT, BETA, 0, pile(), t),
            Err(IntegratorError::Core(CoreError::InvalidBeadCount(0)))
        ));
    }
}
