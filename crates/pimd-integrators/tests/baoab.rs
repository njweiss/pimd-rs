//! Tests for `RingPolymerSim::step` (SPEC I-1, I-2, I-3, T-1, T-3, RP-3, RP-6, S-3).
//!
//! The reference step is assembled in the test from its own pieces: B kicks with forces written
//! out for `Harmonic` and `DoubleWell` (P-2), the I-2 per-mode propagator and the T-1 O step in
//! the test's RP-4 basis (`common`). Nothing from the integrator is used to build expectations.

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::{
    SplitMix64, double_well_ef, harmonic_ef, kick, max_rel_diff, random_beads, ref_free_propagate,
    ref_o_step, ring_hamiltonian, thermal_momenta,
};
use pimd_core::{Beads, Cell, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::{DoubleWell, Evaluation, Harmonic, Potential, PotentialError};

const OMEGA: f64 = 0.9;
const DW_A: f64 = 0.3;
const DW_B: f64 = 0.8;

#[derive(Clone, Copy, Debug)]
enum Pot {
    Harmonic,
    DoubleWell,
}

fn make_pot(kind: Pot, masses: &[f64]) -> Box<dyn Potential> {
    match kind {
        Pot::Harmonic => Box::new(Harmonic::new(masses.to_vec(), OMEGA).unwrap()),
        Pot::DoubleWell => Box::new(DoubleWell::new(DW_A, DW_B).unwrap()),
    }
}

fn ref_ef(kind: Pot, q: &Beads, masses: &[f64]) -> (Vec<f64>, Beads) {
    match kind {
        Pot::Harmonic => harmonic_ef(q, masses, OMEGA),
        Pot::DoubleWell => double_well_ef(q, DW_A, DW_B),
    }
}

/// One I-1 step written out: B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2), forces at the new
/// positions evaluated once, after the second A (I-3).
#[allow(clippy::too_many_arguments)]
fn ref_step(
    kind: Pot,
    q: &Beads,
    p: &Beads,
    f: &Beads,
    masses: &[f64],
    beta: f64,
    dt: f64,
    th: Thermostat,
    rng: &mut Rng,
) -> (Beads, Beads, Beads, Vec<f64>) {
    let p1 = kick(p, f, 0.5 * dt);
    let (q2, p2) = ref_free_propagate(q, &p1, masses, beta, 0.5 * dt);
    let p3 = match th {
        Thermostat::Off => p2,
        Thermostat::PileL { tau0 } => ref_o_step(&p2, masses, beta, dt, tau0, rng),
    };
    let (q4, p4) = ref_free_propagate(&q2, &p3, masses, beta, 0.5 * dt);
    let (e, f4) = ref_ef(kind, &q4, masses);
    let p5 = kick(&p4, &f4, 0.5 * dt);
    (q4, p5, f4, e)
}

/// I-1, I-2, I-3, T-1: ten steps of `RingPolymerSim::step` equal the hand-assembled BAOAB
/// sequence (same-seed `Rng`), for `Harmonic` and `DoubleWell`, P ∈ {1, 3, 4}, PILE-L with and
/// without τ_0 and `Off`. Positions, momenta, forces and energies agree to 1e-11 relative
/// (floor 1) after every step. This pins the order (O between the two A's, not before the first
/// B or after the last), the dt/2 in A and B, and that the forces of step n feed step n + 1.
#[test]
fn steps_match_hand_assembled_baoab() {
    let masses = [1.0, 2.5, 6.0];
    let (beta, dt) = (1.4, 0.15);
    let mut srng = SplitMix64(21);
    for kind in [Pot::Harmonic, Pot::DoubleWell] {
        for p in [1, 3, 4] {
            for th in [
                Thermostat::PileL { tau0: Some(0.7) },
                Thermostat::PileL { tau0: None },
                Thermostat::Off,
            ] {
                let q0 = random_beads(&mut srng, p, 3, -1.2, 1.2);
                let p0 = random_beads(&mut srng, p, 3, -2.0, 2.0);
                let state = RingState {
                    q: q0.clone(),
                    p: p0.clone(),
                    masses: masses.to_vec(),
                };
                let ig = RingPolymerIntegrator::new(dt, beta, p, th).unwrap();
                let mut sim =
                    RingPolymerSim::new(state, None, ig, make_pot(kind, &masses)).unwrap();
                let (mut q, mut pm) = (q0, p0);
                let (_, mut f) = ref_ef(kind, &q, &masses);
                let mut rng = Rng::from_seed(5);
                let mut rng_ref = Rng::from_seed(5);
                for n in 0..10 {
                    sim.step(&mut rng).unwrap();
                    let (q1, p1, f1, e1) =
                        ref_step(kind, &q, &pm, &f, &masses, beta, dt, th, &mut rng_ref);
                    (q, pm, f) = (q1, p1, f1);
                    let ctx = format!("{kind:?}, P = {p}, {th:?}, step {n}");
                    assert!(max_rel_diff(&sim.state().q, &q, 1.0) <= 1e-11, "{ctx}: q");
                    assert!(max_rel_diff(&sim.state().p, &pm, 1.0) <= 1e-11, "{ctx}: p");
                    assert!(
                        max_rel_diff(sim.forces(), &f, 1.0) <= 1e-11,
                        "{ctx}: forces"
                    );
                    for (a, b) in sim.energies().iter().zip(&e1) {
                        assert!((a - b).abs() <= 1e-11 * b.abs().max(1.0), "{ctx}: energies");
                    }
                }
            }
        }
    }
}

/// Wraps a potential, counting `evaluate` calls and recording the positions passed.
struct Counting {
    inner: Harmonic,
    log: Rc<RefCell<Vec<Beads>>>,
}

impl Potential for Counting {
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        self.log.borrow_mut().push(positions.clone());
        self.inner.evaluate(positions, cell, subset)
    }
}

/// I-3: exactly one `evaluate` at construction (at the initial positions) and exactly one per
/// step, at the step's final positions (after the second A), for PILE-L and Off.
#[test]
fn one_evaluation_per_step() {
    let masses = [1.0, 2.0];
    for th in [Thermostat::PileL { tau0: Some(1.0) }, Thermostat::Off] {
        let mut srng = SplitMix64(31);
        let state = RingState {
            q: random_beads(&mut srng, 4, 2, -1.0, 1.0),
            p: random_beads(&mut srng, 4, 2, -1.0, 1.0),
            masses: masses.to_vec(),
        };
        let log = Rc::new(RefCell::new(Vec::new()));
        let pot = Counting {
            inner: Harmonic::new(masses.to_vec(), 1.0).unwrap(),
            log: Rc::clone(&log),
        };
        let ig = RingPolymerIntegrator::new(0.1, 2.0, 4, th).unwrap();
        let mut sim = RingPolymerSim::new(state.clone(), None, ig, Box::new(pot)).unwrap();
        assert_eq!(log.borrow().len(), 1, "{th:?}: construction");
        assert_eq!(log.borrow()[0], state.q, "{th:?}: construction positions");
        let mut rng = Rng::from_seed(3);
        for n in 1..=25 {
            sim.step(&mut rng).unwrap();
            let calls = log.borrow();
            assert_eq!(calls.len(), 1 + n, "{th:?}: after {n} steps");
            assert_eq!(
                calls[n],
                sim.state().q,
                "{th:?}: step {n} evaluated elsewhere"
            );
        }
    }
}

/// T-3, RP-3: with `Off` the dynamics conserve H_P with O(dt²) error. `Harmonic` (ω = 1,
/// masses 1 and 3), P = 8, β = 4, N = 2, from a fixed thermal-like start. Over a fixed duration
/// T = 20 the max |H_P(t) − H_P(0)| at dt = 0.05 divided by that at dt = 0.025 lies in [3, 5]
/// (dt² scaling gives 4). H_P is computed in the test from RP-3 (`ring_hamiltonian`). The
/// deviation at dt = 0.05 is also required to be small (< 1e-2 of H_P(0)) but non-zero.
#[test]
fn nve_energy_error_scales_as_dt_squared() {
    let masses = [1.0, 3.0];
    let (beta, omega, p) = (4.0, 1.0, 8);
    let mut rng = Rng::from_seed(41);
    let mut srng = SplitMix64(41);
    let q0 = random_beads(&mut srng, p, 2, -0.8, 0.8);
    let p0 = thermal_momenta(&mut rng, p, &masses, beta);
    let max_dev = |dt: f64| {
        let state = RingState {
            q: q0.clone(),
            p: p0.clone(),
            masses: masses.to_vec(),
        };
        let ig = RingPolymerIntegrator::new(dt, beta, p, Thermostat::Off).unwrap();
        let pot = Harmonic::new(masses.to_vec(), omega).unwrap();
        let mut sim = RingPolymerSim::new(state, None, ig, Box::new(pot)).unwrap();
        let h = |s: &RingPolymerSim| {
            ring_hamiltonian(&s.state().q, &s.state().p, &masses, beta, s.energies())
        };
        let h0 = h(&sim);
        let steps = (20.0 / dt).round() as usize;
        let mut dev: f64 = 0.0;
        let mut r = Rng::from_seed(0);
        for _ in 0..steps {
            sim.step(&mut r).unwrap();
            dev = dev.max((h(&sim) - h0).abs());
        }
        (dev, h0)
    };
    let (d1, h0) = max_dev(0.05);
    let (d2, _) = max_dev(0.025);
    let ratio = d1 / d2;
    assert!(
        d1 > 0.0 && d1 < 1e-2 * h0.abs(),
        "max |dH| = {d1:e}, H0 = {h0}"
    );
    assert!(
        (3.0..=5.0).contains(&ratio),
        "dH(dt)/dH(dt/2) = {ratio} ({d1:e} / {d2:e})"
    );
}

/// Velocity Verlet written in the test (I-4 form): p += dt/2 F; q += dt p/m; F = F(q); p += dt/2 F.
fn velocity_verlet(
    kind: Pot,
    q0: &Beads,
    p0: &Beads,
    masses: &[f64],
    dt: f64,
    steps: usize,
) -> Vec<(Beads, Beads)> {
    let (mut q, mut p) = (q0.clone(), p0.clone());
    let (_, mut f) = ref_ef(kind, &q, masses);
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        p = kick(&p, &f, 0.5 * dt);
        for (i, &m) in masses.iter().enumerate() {
            let (x, v) = (q.get(0, i), p.get(0, i));
            q.set(
                0,
                i,
                [
                    x[0] + dt * v[0] / m,
                    x[1] + dt * v[1] / m,
                    x[2] + dt * v[2] / m,
                ],
            );
        }
        f = ref_ef(kind, &q, masses).1;
        p = kick(&p, &f, 0.5 * dt);
        out.push((q.clone(), p.clone()));
    }
    out
}

/// RP-6, I-1, I-2: with P = 1 and `Off`, A(dt/2) O A(dt/2) is a drift of dt (ω_0 = 0, O the
/// identity), so BAOAB is velocity Verlet. 200 steps match the test's own velocity Verlet to
/// 1e-12 relative (floor 1), for `Harmonic` and `DoubleWell`, N = 2.
#[test]
fn p1_off_equals_velocity_verlet() {
    let masses = [1.0, 5.0];
    let dt = 0.07;
    for kind in [Pot::Harmonic, Pot::DoubleWell] {
        let q0 = Beads::from_vec(1, 2, vec![0.9, -0.4, 1.3, -1.1, 0.2, 0.6]).unwrap();
        let p0 = Beads::from_vec(1, 2, vec![0.3, 0.8, -0.5, 1.5, -2.0, 0.7]).unwrap();
        let state = RingState {
            q: q0.clone(),
            p: p0.clone(),
            masses: masses.to_vec(),
        };
        // β enters nothing at P = 1 with Off; any valid value.
        let ig = RingPolymerIntegrator::new(dt, 3.0, 1, Thermostat::Off).unwrap();
        let mut sim = RingPolymerSim::new(state, None, ig, make_pot(kind, &masses)).unwrap();
        let want = velocity_verlet(kind, &q0, &p0, &masses, dt, 200);
        let mut rng = Rng::from_seed(0);
        for (n, (q, p)) in want.iter().enumerate() {
            sim.step(&mut rng).unwrap();
            let dq = max_rel_diff(&sim.state().q, q, 1.0);
            let dp = max_rel_diff(&sim.state().p, p, 1.0);
            assert!(
                dq <= 1e-12 && dp <= 1e-12,
                "{kind:?}, step {n}: dq {dq:e}, dp {dp:e}"
            );
        }
    }
}
