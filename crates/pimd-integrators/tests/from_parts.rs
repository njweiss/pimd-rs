//! `RingPolymerSim::from_parts` (MILESTONE_5 "Required tests: from_parts"; Q18 restart, SPEC I-3,
//! S-2, P-1, D-3). Built from the forces and energies that `new` computed, it continues exactly
//! like `new` (bitwise, over a run); it validates masses and shapes like `new`; it never calls
//! `evaluate` (a counting potential in the test).

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{SplitMix64, random_beads, thermal_momenta};
use pimd_core::{Beads, Cell, Rng};
use pimd_integrators::{
    IntegratorError, RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat,
};
use pimd_potentials::{Evaluation, LennardJones, LjParams, Potential, PotentialError};

/// Wraps a potential and counts `evaluate` calls.
struct Counting {
    inner: Box<dyn Potential>,
    calls: Arc<AtomicUsize>,
}

impl Potential for Counting {
    fn evaluate(
        &mut self,
        q: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.evaluate(q, cell, subset)
    }
}

const P: usize = 4;
const N: usize = 5;
const BETA: f64 = 1000.0;
const DT: f64 = 20.0;

/// LJ (P-3) in a periodic box with a small irregular cluster, masses differing per atom.
fn setup() -> (RingState, Cell, LjParams) {
    let mut g = SplitMix64(7);
    let cell = Cell::new([31.0, 32.0, 33.0]).unwrap();
    // A chain of atoms 6.2 bohr apart along x (also across the boundary), beads jittered.
    let mut q = Beads::zeros(P, N);
    let jitter = random_beads(&mut g, P, N, -0.05, 0.05);
    for i in 0..N {
        let base = [
            3.0 + 6.2 * i as f64,
            8.0 + 0.3 * i as f64,
            9.0 - 0.2 * i as f64,
        ];
        for j in 0..P {
            let d = jitter.get(j, i);
            q.set(j, i, [base[0] + d[0], base[1] + d[1], base[2] + d[2]]);
        }
    }
    let masses = vec![36000.0, 30000.0, 1837.0, 20000.0, 50000.0];
    let mut rng = Rng::from_seed(99);
    let p = thermal_momenta(&mut rng, P, &masses, BETA);
    let params = LjParams {
        epsilon: 1.2e-4,
        sigma: 5.2,
        cutoff: 9.0,
        skin: 0.8,
        shift: true,
    };
    (RingState { q, p, masses }, cell, params)
}

fn integrator() -> RingPolymerIntegrator {
    RingPolymerIntegrator::new(DT, BETA, P, Thermostat::PileL { tau0: Some(500.0) }).unwrap()
}

fn counting(params: LjParams) -> (Box<dyn Potential>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let pot = Counting {
        inner: Box::new(LennardJones::new(params).unwrap()),
        calls: calls.clone(),
    };
    (Box::new(pot), calls)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Q18, I-3: `from_parts` with the forces and energies `new` computed performs no `evaluate`,
/// exposes exactly those forces and energies, and then 40 steps (same RNG stream) give bitwise the
/// same state, forces and energies as the `new` simulation, with one `evaluate` per step each.
#[test]
fn from_parts_equals_new_bitwise_without_evaluating() {
    let (state, cell, params) = setup();
    let (pot_a, calls_a) = counting(params);
    let mut a = RingPolymerSim::new(state.clone(), Some(cell), integrator(), pot_a).unwrap();
    assert_eq!(
        calls_a.load(Ordering::SeqCst),
        1,
        "new evaluates once (I-3)"
    );
    let (f0, e0) = (a.forces().clone(), a.energies().to_vec());

    let (pot_b, calls_b) = counting(params);
    let mut b = RingPolymerSim::from_parts(
        state.clone(),
        Some(cell),
        integrator(),
        pot_b,
        f0.clone(),
        e0.clone(),
    )
    .unwrap();
    assert_eq!(
        calls_b.load(Ordering::SeqCst),
        0,
        "from_parts must not evaluate"
    );
    assert_eq!(b.state(), &state);
    assert_eq!(bits(b.forces().as_slice()), bits(f0.as_slice()));
    assert_eq!(bits(b.energies()), bits(&e0));

    let (mut ra, mut rb) = (Rng::from_seed(5), Rng::from_seed(5));
    for s in 0..40 {
        a.step(&mut ra).unwrap();
        b.step(&mut rb).unwrap();
        assert_eq!(
            bits(a.state().q.as_slice()),
            bits(b.state().q.as_slice()),
            "q, step {s}"
        );
        assert_eq!(
            bits(a.state().p.as_slice()),
            bits(b.state().p.as_slice()),
            "p, step {s}"
        );
        assert_eq!(bits(a.forces().as_slice()), bits(b.forces().as_slice()));
        assert_eq!(bits(a.energies()), bits(b.energies()));
    }
    assert_eq!(calls_a.load(Ordering::SeqCst), 41);
    assert_eq!(calls_b.load(Ordering::SeqCst), 40);
}

/// Q18: the given forces are used as they are (restored, not recomputed): with every force scaled
/// by 2, the first step's momenta follow the I-1 B step with those forces, so they differ from the
/// `new` simulation's, while positions after one step (A O A use the half-kicked momenta) differ
/// too; gas phase (cell `None`) is accepted.
#[test]
fn given_forces_are_used() {
    let (state, _, params) = setup();
    let (pot_a, _) = counting(params);
    let mut a = RingPolymerSim::new(state.clone(), None, integrator(), pot_a).unwrap();
    let mut f2 = a.forces().clone();
    for x in f2.as_mut_slice() {
        *x *= 2.0;
    }
    let (pot_b, calls_b) = counting(params);
    let mut b = RingPolymerSim::from_parts(
        state,
        None,
        integrator(),
        pot_b,
        f2.clone(),
        a.energies().to_vec(),
    )
    .unwrap();
    assert_eq!(bits(b.forces().as_slice()), bits(f2.as_slice()));
    let (mut ra, mut rb) = (Rng::from_seed(1), Rng::from_seed(1));
    a.step(&mut ra).unwrap();
    b.step(&mut rb).unwrap();
    assert_ne!(
        bits(a.state().q.as_slice()),
        bits(b.state().q.as_slice()),
        "doubled restored forces had no effect: forces were re-evaluated"
    );
    assert_eq!(calls_b.load(Ordering::SeqCst), 1);
}

fn expect_shape(r: Result<RingPolymerSim, IntegratorError>, what: &str) {
    match r {
        Err(IntegratorError::ShapeMismatch { .. }) => {}
        Err(e) => panic!("{what}: expected ShapeMismatch, got {e:?}"),
        Ok(_) => panic!("{what}: accepted"),
    }
}

/// P-1 (as `new`): forces of the wrong bead or atom count, energies of the wrong length, a state
/// whose q, p, masses disagree, or a P differing from the integrator's are `ShapeMismatch`; no
/// `evaluate` is called.
#[test]
fn shapes_validated() {
    let (state, cell, params) = setup();
    let f = Beads::zeros(P, N);
    let e = vec![0.0; P];
    let mut bad_p = state.clone();
    bad_p.p = Beads::zeros(P, N + 1);
    let mut bad_m = state.clone();
    bad_m.masses.pop();
    let mut bad_beads = state.clone();
    bad_beads.q = Beads::zeros(P + 1, N);
    bad_beads.p = Beads::zeros(P + 1, N);
    let cases: Vec<(&str, RingState, Beads, Vec<f64>)> = vec![
        (
            "forces P+1",
            state.clone(),
            Beads::zeros(P + 1, N),
            e.clone(),
        ),
        (
            "forces N-1",
            state.clone(),
            Beads::zeros(P, N - 1),
            e.clone(),
        ),
        ("energies P-1", state.clone(), f.clone(), vec![0.0; P - 1]),
        ("energies P+1", state.clone(), f.clone(), vec![0.0; P + 1]),
        ("p atoms", bad_p, f.clone(), e.clone()),
        ("masses", bad_m, f.clone(), e.clone()),
        (
            "integrator P",
            bad_beads,
            Beads::zeros(P + 1, N),
            vec![0.0; P + 1],
        ),
    ];
    for (what, st, f, e) in cases {
        let (pot, calls) = counting(params);
        expect_shape(
            RingPolymerSim::from_parts(st, Some(cell), integrator(), pot, f, e),
            what,
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0, "{what}: evaluate called");
    }
}

/// S-2 (as `new`): a zero, negative, NaN or infinite mass is `InvalidValue`; no `evaluate`.
#[test]
fn masses_validated() {
    let (state, cell, params) = setup();
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut st = state.clone();
        st.masses[2] = bad;
        let (pot, calls) = counting(params);
        let r = RingPolymerSim::from_parts(
            st,
            Some(cell),
            integrator(),
            pot,
            Beads::zeros(P, N),
            vec![0.0; P],
        );
        assert!(
            matches!(r, Err(IntegratorError::InvalidValue { .. })),
            "mass {bad}: {:?}",
            r.err()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

/// Q23 (M5 review): a NaN or infinite given force or energy (a corrupt checkpoint that is
/// otherwise well formed) is `InvalidValue { what: "given forces" | "given energies" }`; no
/// `evaluate`.
#[test]
fn non_finite_parts_rejected() {
    let (state, cell, params) = setup();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut f = Beads::zeros(P, N);
        f.set(P - 1, N - 1, [0.0, 0.0, bad]);
        let mut e = vec![0.0; P];
        e[P - 1] = bad;
        let cases = [
            ("given forces", f, vec![0.0; P]),
            ("given energies", Beads::zeros(P, N), e),
        ];
        for (want, f, e) in cases {
            let (pot, calls) = counting(params);
            match RingPolymerSim::from_parts(state.clone(), Some(cell), integrator(), pot, f, e) {
                Err(IntegratorError::InvalidValue { what, .. }) => assert_eq!(what, want),
                Err(other) => panic!("{want} {bad}: expected InvalidValue, got {other:?}"),
                Ok(_) => panic!("{want} {bad}: accepted"),
            }
            assert_eq!(calls.load(Ordering::SeqCst), 0, "{want}: evaluate called");
        }
    }
}
