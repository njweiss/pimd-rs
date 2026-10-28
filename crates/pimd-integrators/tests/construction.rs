//! Construction and error paths of `RingPolymerIntegrator` and `RingPolymerSim`
//! (SPEC I-5, T-2, RP-4, I-3, P-1).

mod common;

use common::{SplitMix64, harmonic_ef, random_beads, ref_freqs, ref_matrix};
use pimd_core::{Beads, Cell, CoreError, Rng};
use pimd_integrators::{
    IntegratorError, RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat,
};
use pimd_potentials::{Evaluation, Harmonic, Potential, PotentialError};

const BAD: [f64; 5] = [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

fn is_invalid_value(r: Result<RingPolymerIntegrator, IntegratorError>) -> bool {
    matches!(r, Err(IntegratorError::InvalidValue { .. }))
}

/// I-5: dt must be positive and finite; otherwise `IntegratorError::InvalidValue`.
#[test]
fn rejects_bad_dt() {
    for dt in BAD {
        for th in [Thermostat::Off, Thermostat::PileL { tau0: Some(1.0) }] {
            let r = RingPolymerIntegrator::new(dt, 1.0, 4, th);
            assert!(is_invalid_value(r), "dt = {dt} accepted");
        }
    }
}

/// MILESTONE_2 `new` contract (β > 0) and `IntegratorError::InvalidValue` ("dt, β, τ₀"):
/// non-positive or non-finite β is `InvalidValue`.
#[test]
fn rejects_bad_beta() {
    for beta in BAD {
        let r = RingPolymerIntegrator::new(0.1, beta, 4, Thermostat::Off);
        assert!(is_invalid_value(r), "beta = {beta} accepted");
    }
}

/// T-2 (τ_0 configured ⇒ γ_0 = 1/τ_0) and the brief's "tau0 > 0 when Some": τ_0 must be
/// positive and finite when given; otherwise `InvalidValue`.
#[test]
fn rejects_bad_tau0() {
    for tau0 in BAD {
        let r = RingPolymerIntegrator::new(0.1, 1.0, 4, Thermostat::PileL { tau0: Some(tau0) });
        assert!(is_invalid_value(r), "tau0 = {tau0} accepted");
    }
}

/// RP-1: P = 0 is rejected as `IntegratorError::Core(CoreError::InvalidBeadCount(0))`, the only
/// error source for it being `NormalModes::new`.
#[test]
fn rejects_zero_beads() {
    let r = RingPolymerIntegrator::new(0.1, 1.0, 0, Thermostat::Off);
    assert!(
        matches!(
            r,
            Err(IntegratorError::Core(CoreError::InvalidBeadCount(0)))
        ),
        "{r:?}"
    );
}

/// RP-4, RP-2: `modes()` is the RP-4 transform at the given P and β (matrix and frequencies
/// equal the test's own RP-4 construction to 1e-14), for both thermostat kinds.
#[test]
fn modes_are_rp4() {
    for (p, beta) in [(1, 0.7), (4, 2.0), (7, 10.0)] {
        for th in [Thermostat::Off, Thermostat::PileL { tau0: None }] {
            let ig = RingPolymerIntegrator::new(0.05, beta, p, th).unwrap();
            let nm = ig.modes();
            assert_eq!(nm.n_beads(), p);
            for (a, b) in nm.matrix().iter().zip(ref_matrix(p)) {
                assert!((a - b).abs() <= 1e-14, "P = {p}: matrix");
            }
            for (a, b) in nm.frequencies().iter().zip(ref_freqs(p, beta)) {
                assert!((a - b).abs() <= 1e-14 * b.max(1.0), "P = {p}: frequencies");
            }
        }
    }
}

fn harmonic_state(p: usize, masses: &[f64]) -> RingState {
    let mut rng = SplitMix64(5);
    RingState {
        q: random_beads(&mut rng, p, masses.len(), -1.0, 1.0),
        p: random_beads(&mut rng, p, masses.len(), -1.0, 1.0),
        masses: masses.to_vec(),
    }
}

fn sim(
    state: RingState,
    p: usize,
    pot: Box<dyn Potential>,
) -> Result<RingPolymerSim, IntegratorError> {
    let ig =
        RingPolymerIntegrator::new(0.1, 1.0, p, Thermostat::PileL { tau0: Some(1.0) }).unwrap();
    RingPolymerSim::new(state, None, ig, pot)
}

/// I-3: `new` performs the initialising evaluate: `forces()` and `energies()` equal −∇V and V
/// of `Harmonic` at the initial positions (written out in the test, 1e-14), and `state()` is
/// the given state.
#[test]
fn new_evaluates_initial_state() {
    let masses = [1.0, 3.0];
    let s0 = harmonic_state(3, &masses);
    let pot = Harmonic::new(masses.to_vec(), 0.8).unwrap();
    let sim = sim(s0.clone(), 3, Box::new(pot)).unwrap();
    assert_eq!(sim.state(), &s0);
    let (e, f) = harmonic_ef(&s0.q, &masses, 0.8);
    assert_eq!(sim.energies().len(), 3);
    for (a, b) in sim.energies().iter().zip(&e) {
        assert!((a - b).abs() <= 1e-14 * b.abs().max(1.0));
    }
    assert!(common::max_rel_diff(sim.forces(), &f, 1.0) <= 1e-14);
}

/// RingPolymerSim::new contract: q, p, masses and the integrator's P must agree; otherwise
/// `IntegratorError::ShapeMismatch` (fields not pinned).
#[test]
fn new_rejects_shape_mismatch() {
    let masses = [1.0, 3.0];
    let pot = || Box::new(Harmonic::new(masses.to_vec(), 1.0).unwrap()) as Box<dyn Potential>;
    let good = harmonic_state(3, &masses);
    let is_shape = |r: Result<RingPolymerSim, IntegratorError>| {
        matches!(r, Err(IntegratorError::ShapeMismatch { .. }))
    };
    // Integrator P = 4, state P = 3.
    assert!(
        is_shape(sim(good.clone(), 4, pot())),
        "bead count vs integrator"
    );
    // p has a different bead count from q.
    let mut s = good.clone();
    s.p = Beads::zeros(2, 2);
    assert!(is_shape(sim(s, 3, pot())), "p bead count");
    // p has a different atom count from q.
    let mut s = good.clone();
    s.p = Beads::zeros(3, 1);
    assert!(is_shape(sim(s, 3, pot())), "p atom count");
    // masses.len() != N.
    let mut s = good.clone();
    s.masses = vec![1.0, 3.0, 2.0];
    assert!(is_shape(sim(s, 3, pot())), "masses length");
}

/// P-1, I-3: an error from the initialising evaluate is returned as
/// `IntegratorError::Potential` (here `Harmonic` built for 2 atoms, state with 3).
#[test]
fn new_propagates_potential_error() {
    let s = harmonic_state(2, &[1.0, 2.0, 3.0]);
    let pot = Harmonic::new(vec![1.0, 2.0], 1.0).unwrap();
    let r = sim(s, 2, Box::new(pot));
    assert!(
        matches!(
            r,
            Err(IntegratorError::Potential(PotentialError::ShapeMismatch {
                expected: 2,
                actual: 3
            }))
        ),
        "{:?}",
        r.err()
    );
}

/// A potential that succeeds `ok_calls` times, then returns `Unsupported`.
struct FailAfter {
    inner: Harmonic,
    ok_calls: usize,
}

impl Potential for FailAfter {
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        if self.ok_calls == 0 {
            return Err(PotentialError::Unsupported);
        }
        self.ok_calls -= 1;
        self.inner.evaluate(positions, cell, subset)
    }
}

/// I-3, P-1: an error from the per-step evaluate is returned by `step` as
/// `IntegratorError::Potential`.
#[test]
fn step_propagates_potential_error() {
    let masses = [1.0, 2.0];
    let pot = FailAfter {
        inner: Harmonic::new(masses.to_vec(), 1.0).unwrap(),
        ok_calls: 2,
    };
    let mut sim = sim(harmonic_state(2, &masses), 2, Box::new(pot)).unwrap();
    let mut rng = Rng::from_seed(1);
    assert!(sim.step(&mut rng).is_ok());
    let r = sim.step(&mut rng);
    assert!(
        matches!(
            r,
            Err(IntegratorError::Potential(PotentialError::Unsupported))
        ),
        "{r:?}"
    );
}
