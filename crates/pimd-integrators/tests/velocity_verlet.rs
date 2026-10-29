//! Tests for `VelocityVerlet` (SPEC I-3, I-4, I-5, C-5, P-1) with the merged `Harmonic` and
//! `DoubleWell` potentials. The expected trajectories come from the test's own velocity Verlet
//! with hand-written forces (`common::harmonic_ef`, `common::double_well_ef`).

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::{SplitMix64, double_well_ef, harmonic_ef, kick, max_rel_diff, random_beads};
use pimd_core::{Beads, Cell};
use pimd_integrators::{IntegratorError, RingState, VelocityVerlet};
use pimd_potentials::{DoubleWell, Evaluation, Harmonic, Potential, PotentialError};

const MASSES: [f64; 3] = [1.0, 2.5, 7.0];
const OMEGA: f64 = 0.9;
const DW_A: f64 = 0.125;
const DW_B: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
enum Pot {
    Harmonic,
    DoubleWell,
}

fn make_pot(kind: Pot) -> Box<dyn Potential> {
    match kind {
        Pot::Harmonic => Box::new(Harmonic::new(MASSES.to_vec(), OMEGA).unwrap()),
        Pot::DoubleWell => Box::new(DoubleWell::new(DW_A, DW_B).unwrap()),
    }
}

fn ref_ef(kind: Pot, q: &Beads) -> (Vec<f64>, Beads) {
    match kind {
        Pot::Harmonic => harmonic_ef(q, &MASSES, OMEGA),
        Pot::DoubleWell => double_well_ef(q, DW_A, DW_B),
    }
}

fn start(seed: u64) -> RingState {
    let mut rng = SplitMix64(seed);
    RingState {
        q: random_beads(&mut rng, 1, 3, -2.0, 2.0),
        p: random_beads(&mut rng, 1, 3, -1.5, 1.5),
        masses: MASSES.to_vec(),
    }
}

/// The test's velocity Verlet (I-4): p += dt/2 F; q += dt p/m; F = F(q); p += dt/2 F.
/// Returns (q, p, F, V) after the step.
fn ref_step(kind: Pot, q: &Beads, p: &Beads, f: &Beads, dt: f64) -> (Beads, Beads, Beads, f64) {
    let ph = kick(p, f, 0.5 * dt);
    let mut qn = q.clone();
    for (i, &m) in MASSES.iter().enumerate() {
        let (x, v) = (q.get(0, i), ph.get(0, i));
        qn.set(0, i, [0, 1, 2].map(|d| x[d] + dt * v[d] / m));
    }
    let (e, fnew) = ref_ef(kind, &qn);
    let pn = kick(&ph, &fnew, 0.5 * dt);
    (qn, pn, fnew, e[0])
}

/// I-4: 100 steps (dt = 0.05 ħ/Eh) for `Harmonic` (ω = 0.9) and `DoubleWell` (a = 1/8, b = 1),
/// N = 3 with masses (1, 2.5, 7), match the test's velocity Verlet after every step: positions,
/// momenta and exit forces to 1e-12 relative (floor 1), and the returned energy equals V at the
/// new positions to 1e-12. Pins the dt/2 kicks, the full-dt drift with p/m, and that the exit
/// forces are those at the new positions.
#[test]
fn matches_reference_velocity_verlet() {
    let dt = 0.05;
    for kind in [Pot::Harmonic, Pot::DoubleWell] {
        let mut state = start(1);
        let mut pot = make_pot(kind);
        let mut forces = pot.evaluate(&state.q, None, None).unwrap().forces;
        let (mut q, mut p) = (state.q.clone(), state.p.clone());
        let mut f = ref_ef(kind, &q).1;
        let vv = VelocityVerlet::new(dt).unwrap();
        for step in 0..100 {
            let e = vv
                .step(&mut state, &mut forces, pot.as_mut(), None)
                .unwrap();
            let (qn, pn, fnew, en) = ref_step(kind, &q, &p, &f, dt);
            (q, p, f) = (qn, pn, fnew);
            let tag = format!("{kind:?} step {step}");
            assert!(max_rel_diff(&state.q, &q, 1.0) <= 1e-12, "{tag}: q");
            assert!(max_rel_diff(&state.p, &p, 1.0) <= 1e-12, "{tag}: p");
            assert!(
                max_rel_diff(&forces, &f, 1.0) <= 1e-12,
                "{tag}: exit forces"
            );
            assert!(
                (e - en).abs() <= 1e-12 * en.abs().max(1.0),
                "{tag}: E {e} vs {en}"
            );
        }
    }
}

/// I-3, I-4: the forces passed in are the ones used for the first half kick (not recomputed).
/// With arbitrary entry forces G (not F(q)), one step gives q' = q + dt (p + dt/2 G)/m and
/// p' = p + dt/2 G + dt/2 F(q'), forces = F(q') on exit; to 1e-14 relative (floor 1).
#[test]
fn uses_entry_forces() {
    let dt = 0.1;
    let mut rng = SplitMix64(2);
    let g = random_beads(&mut rng, 1, 3, -3.0, 3.0);
    for kind in [Pot::Harmonic, Pot::DoubleWell] {
        let mut state = start(3);
        let (q, p) = (state.q.clone(), state.p.clone());
        let mut forces = g.clone();
        let mut pot = make_pot(kind);
        let e = VelocityVerlet::new(dt)
            .unwrap()
            .step(&mut state, &mut forces, pot.as_mut(), None)
            .unwrap();
        let (qn, pn, fnew, en) = ref_step(kind, &q, &p, &g, dt);
        assert!(max_rel_diff(&state.q, &qn, 1.0) <= 1e-14, "{kind:?}: q");
        assert!(max_rel_diff(&state.p, &pn, 1.0) <= 1e-14, "{kind:?}: p");
        assert!(
            max_rel_diff(&forces, &fnew, 1.0) <= 1e-14,
            "{kind:?}: forces"
        );
        assert!((e - en).abs() <= 1e-14 * en.abs().max(1.0));
    }
}

/// Positions and cell of every `evaluate` call.
type Log = Rc<RefCell<Vec<(Beads, Option<Cell>)>>>;

/// Wraps a potential, recording the positions and cell of every `evaluate` call.
struct Recording {
    inner: Harmonic,
    log: Log,
}

impl Potential for Recording {
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        self.log
            .borrow_mut()
            .push((positions.clone(), cell.copied()));
        self.inner.evaluate(positions, cell, subset)
    }
}

/// I-3, I-4: exactly one `evaluate` per step, at the step's new positions, with the cell passed
/// through unchanged (here `Some`; also `None`).
#[test]
fn one_evaluation_per_step_at_new_positions() {
    let cell = Cell::new([20.0, 21.0, 22.0]).unwrap();
    for c in [Some(cell), None] {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut pot = Recording {
            inner: Harmonic::new(MASSES.to_vec(), OMEGA).unwrap(),
            log: log.clone(),
        };
        let mut state = start(4);
        let mut forces = harmonic_ef(&state.q, &MASSES, OMEGA).1;
        let vv = VelocityVerlet::new(0.05).unwrap();
        for step in 0..10 {
            vv.step(&mut state, &mut forces, &mut pot, c.as_ref())
                .unwrap();
            let log = log.borrow();
            assert_eq!(log.len(), step + 1, "evaluations after {} steps", step + 1);
            let (q, lc) = log.last().unwrap();
            assert_eq!(q, &state.q, "step {step}: evaluated at the new positions");
            assert_eq!(lc, &c, "step {step}: cell passed through");
        }
    }
}

/// I-4: velocity Verlet is time reversible. 200 steps forward (dt = 0.05, `DoubleWell`), negate the
/// momenta, 200 steps, negate again: back to the start to 1e-10 (floor 1; round-off accumulates).
#[test]
fn time_reversible() {
    let mut state = start(5);
    let (q0, p0) = (state.q.clone(), state.p.clone());
    let mut pot = make_pot(Pot::DoubleWell);
    let mut forces = pot.evaluate(&state.q, None, None).unwrap().forces;
    let vv = VelocityVerlet::new(0.05).unwrap();
    for _ in 0..200 {
        vv.step(&mut state, &mut forces, pot.as_mut(), None)
            .unwrap();
    }
    assert!(
        max_rel_diff(&state.q, &q0, 1.0) > 0.1,
        "trajectory did not move"
    );
    state.p.as_mut_slice().iter_mut().for_each(|x| *x = -*x);
    for _ in 0..200 {
        vv.step(&mut state, &mut forces, pot.as_mut(), None)
            .unwrap();
    }
    state.p.as_mut_slice().iter_mut().for_each(|x| *x = -*x);
    assert!(max_rel_diff(&state.q, &q0, 1.0) <= 1e-10, "q not recovered");
    assert!(max_rel_diff(&state.p, &p0, 1.0) <= 1e-10, "p not recovered");
}

/// C-5, I-4: positions are never wrapped. Free flight (`Harmonic` with ω = 0, so F = 0) in a cell
/// of edge 5: after n steps q = q0 + n dt p/m exactly as in the test's arithmetic (1e-12
/// relative), far outside [0, L) (atom 2 moves −12 bohr, atom 0 +30 bohr).
#[test]
fn positions_are_not_wrapped() {
    let cell = Cell::new([5.0; 3]).unwrap();
    let mut state = RingState {
        q: Beads::from_vec(1, 3, vec![4.9, 1.0, 1.0, 2.0, 0.1, 2.0, 0.2, 0.3, 0.4]).unwrap(),
        p: Beads::from_vec(1, 3, vec![1.5, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.4]).unwrap(),
        masses: MASSES.to_vec(),
    };
    let q0 = state.q.clone();
    let p0 = state.p.clone();
    let mut pot = Harmonic::new(MASSES.to_vec(), 0.0).unwrap();
    let mut forces = Beads::zeros(1, 3);
    let (dt, n) = (0.1, 200);
    let vv = VelocityVerlet::new(dt).unwrap();
    for _ in 0..n {
        vv.step(&mut state, &mut forces, &mut pot, Some(&cell))
            .unwrap();
    }
    let mut want = q0.clone();
    for (i, &m) in MASSES.iter().enumerate() {
        let (x, p) = (q0.get(0, i), p0.get(0, i));
        want.set(0, i, [0, 1, 2].map(|d| x[d] + n as f64 * dt * p[d] / m));
    }
    assert!(max_rel_diff(&state.q, &want, 1.0) <= 1e-12, "{:?}", state.q);
    assert!(state.q.get(0, 0)[0] > 30.0 && state.q.get(0, 2)[2] < -3.5);
    assert_eq!(state.p, p0);
}

/// I-5: dt must be positive and finite; otherwise `IntegratorError::InvalidValue`.
#[test]
fn new_rejects_bad_dt() {
    for dt in [0.0, -0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(
                VelocityVerlet::new(dt),
                Err(IntegratorError::InvalidValue { .. })
            ),
            "dt = {dt} accepted"
        );
    }
    assert!(VelocityVerlet::new(1e-6).is_ok());
}

/// I-4 (P = 1 only), P-1: `step` errors. A state with P = 2 gives
/// `ShapeMismatch { expected: 1, actual: 2, .. }` (the existing variant for a bead-count mismatch).
/// A momentum, force or mass count that disagrees with the positions gives `ShapeMismatch`. A
/// failing evaluation (here `Harmonic` built for 2 atoms, given 3: `PotentialError::ShapeMismatch`,
/// P-1) is passed through as `IntegratorError::Potential`.
#[test]
fn step_errors() {
    let vv = VelocityVerlet::new(0.05).unwrap();
    let mut pot = Harmonic::new(MASSES.to_vec(), OMEGA).unwrap();
    let mut rng = SplitMix64(6);
    let mut two = RingState {
        q: random_beads(&mut rng, 2, 3, -1.0, 1.0),
        p: random_beads(&mut rng, 2, 3, -1.0, 1.0),
        masses: MASSES.to_vec(),
    };
    let mut f2 = Beads::zeros(2, 3);
    let r = vv.step(&mut two, &mut f2, &mut pot, None);
    assert!(
        matches!(
            r,
            Err(IntegratorError::ShapeMismatch {
                expected: 1,
                actual: 2,
                ..
            })
        ),
        "P = 2: {r:?}"
    );

    let good = start(7);
    let f = Beads::zeros(1, 3);
    let mut bad_p = good.clone();
    bad_p.p = Beads::zeros(1, 4);
    let mut bad_m = good.clone();
    bad_m.masses = vec![1.0, 2.0];
    for (what, mut s, mut ff) in [
        ("momenta", bad_p, f.clone()),
        ("masses", bad_m, f.clone()),
        ("forces", good.clone(), Beads::zeros(1, 2)),
        ("force beads", good.clone(), Beads::zeros(2, 3)),
    ] {
        let r = vv.step(&mut s, &mut ff, &mut pot, None);
        assert!(
            matches!(r, Err(IntegratorError::ShapeMismatch { .. })),
            "{what}: {r:?}"
        );
    }

    let mut s = good.clone();
    let mut ff = f.clone();
    let mut wrong = Harmonic::new(vec![1.0, 2.0], OMEGA).unwrap();
    let r = vv.step(&mut s, &mut ff, &mut wrong, None);
    assert!(
        matches!(
            r,
            Err(IntegratorError::Potential(
                PotentialError::ShapeMismatch { .. }
            ))
        ),
        "{r:?}"
    );
}
