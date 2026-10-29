//! Ring-polymer state, BAOAB integrator with PILE-L (SPEC I-1 … I-5, T-1 … T-3).

use pimd_core::{Beads, Cell, NormalModes, Rng};
use pimd_potentials::{Evaluation, Potential};

use crate::error::IntegratorError;

/// Ring-polymer phase-space point in the bead representation (RP-3, S-1).
#[derive(Debug, Clone, PartialEq)]
pub struct RingState {
    /// Bead positions, shape `[P, N, 3]`, in bohr.
    pub q: Beads,
    /// Bead momenta, shape `[P, N, 3]`, in m_e·bohr per ħ/Eh (atomic units). Their variance in
    /// equilibrium is m_i·P/β per component (RP-3).
    pub p: Beads,
    /// One mass per atom, length N, in m_e (S-2).
    pub masses: Vec<f64>,
}

/// Thermostat applied in the O step (T-1, T-2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Thermostat {
    /// No thermostat: the O step is the identity (T-3).
    Off,
    /// PILE-L (T-1). `tau0` is the centroid time constant in ħ/Eh, > 0 when `Some`;
    /// `None` leaves the centroid unthermostatted (γ_0 = 0, T-2).
    PileL {
        /// Centroid time constant τ_0 in ħ/Eh, or `None` for γ_0 = 0.
        tau0: Option<f64>,
    },
}

/// BAOAB ring-polymer integrator with exact free propagation and optional PILE-L (I-1, I-2, T-1).
#[derive(Debug, Clone)]
pub struct RingPolymerIntegrator {
    /// Time step, ħ/Eh.
    dt: f64,
    /// Inverse temperature, Eh⁻¹.
    beta: f64,
    modes: NormalModes,
    thermostat: Thermostat,
    /// c_k = exp(−γ_k dt) per mode, aligned with the `modes` columns (T-1, T-2).
    c_k: Vec<f64>,
}

/// Returns `v` if it is positive and finite, else `IntegratorError::InvalidValue` named `what`.
fn positive_finite(what: &'static str, v: f64) -> Result<f64, IntegratorError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(IntegratorError::InvalidValue { what, value: v })
    }
}

impl RingPolymerIntegrator {
    /// Integrator with time step `dt` (ħ/Eh), inverse temperature `beta` (Eh⁻¹) and `n_beads` = P.
    ///
    /// Errors: `IntegratorError::InvalidValue` unless dt (I-5) and β are positive and finite, and
    /// τ_0 is positive and finite when given; `IntegratorError::Core` for P = 0.
    pub fn new(
        dt: f64,
        beta: f64,
        n_beads: usize,
        thermostat: Thermostat,
    ) -> Result<Self, IntegratorError> {
        positive_finite("dt (hbar/Eh)", dt)?;
        positive_finite("beta (1/Eh)", beta)?;
        let gamma0 = match thermostat {
            Thermostat::PileL { tau0: Some(tau0) } => {
                1.0 / positive_finite("tau0 (hbar/Eh)", tau0)?
            }
            _ => 0.0,
        };
        let modes = NormalModes::new(n_beads, beta)?;
        // T-2: γ_0 = 1/τ_0 (or 0), γ_k = 2 ω_k for ω_k > 0; T-1: c_k = exp(−γ_k dt).
        let c_k = modes
            .frequencies()
            .iter()
            .enumerate()
            .map(|(k, &w)| {
                let gamma = if k == 0 { gamma0 } else { 2.0 * w };
                (-gamma * dt).exp()
            })
            .collect();
        Ok(RingPolymerIntegrator {
            dt,
            beta,
            modes,
            thermostat,
            c_k,
        })
    }

    /// Exact free ring-polymer propagation of `state` for time `t` (ħ/Eh, any sign), per normal
    /// mode and component (I-2), using `state.masses`. No potential forces.
    ///
    /// `state.masses` must be positive and finite (S-2); this is not checked here
    /// (`RingPolymerSim::new` checks it).
    ///
    /// Panics if the bead count of `state` differs from P or `state.masses.len()` from N.
    pub fn free_propagate(&self, state: &mut RingState, t: f64) {
        self.assert_shape(state);
        let mut qt = self.modes.to_normal(&state.q);
        let mut pt = self.modes.to_normal(&state.p);
        self.rotate_modes(&mut qt, &mut pt, &state.masses, t);
        state.q = self.modes.to_beads(&qt);
        state.p = self.modes.to_beads(&pt);
    }

    /// One O step of length dt (T-1): PILE-L on the normal-mode momenta, or the identity for
    /// `Thermostat::Off`. Draws one normal per (k, i, d), k outermost, then atom i, then d.
    /// `Thermostat::Off` draws nothing (T-3).
    ///
    /// `state.masses` must be positive and finite (S-2); this is not checked here
    /// (`RingPolymerSim::new` checks it).
    ///
    /// Panics if the bead count of `state` differs from P or `state.masses.len()` from N.
    pub fn thermostat_step(&self, state: &mut RingState, rng: &mut Rng) {
        self.assert_shape(state);
        if self.thermostat == Thermostat::Off {
            return;
        }
        let mut pt = self.modes.to_normal(&state.p);
        self.thermostat_modes(&mut pt, &state.masses, rng);
        state.p = self.modes.to_beads(&pt);
    }

    /// The normal-mode transform used by this integrator (RP-4).
    pub fn modes(&self) -> &NormalModes {
        &self.modes
    }

    fn assert_shape(&self, state: &RingState) {
        let p = self.modes.n_beads();
        assert!(
            state.q.n_beads() == p && state.p.n_beads() == p,
            "integrator built for P = {p} beads, state has {} (q) and {} (p)",
            state.q.n_beads(),
            state.p.n_beads()
        );
        let n = state.q.n_atoms();
        assert!(
            state.p.n_atoms() == n && state.masses.len() == n,
            "state has {n} atoms in q, {} in p, {} masses",
            state.p.n_atoms(),
            state.masses.len()
        );
    }

    /// I-2 in normal-mode coordinates `qt`, `pt` (`[P, N, 3]`, mode-major) for time `t`.
    fn rotate_modes(&self, qt: &mut Beads, pt: &mut Beads, masses: &[f64], t: f64) {
        let n3 = masses.len() * 3;
        let (qs, ps) = (qt.as_mut_slice(), pt.as_mut_slice());
        for (k, &w) in self.modes.frequencies().iter().enumerate() {
            let mode = k * n3..(k + 1) * n3;
            let pairs = qs[mode.clone()]
                .chunks_exact_mut(3)
                .zip(ps[mode].chunks_exact_mut(3));
            if w > 0.0 {
                let (s, c) = (w * t).sin_cos();
                for ((q, p), &m) in pairs.zip(masses) {
                    for (q, p) in q.iter_mut().zip(p) {
                        let (q0, p0) = (*q, *p);
                        *q = c * q0 + s / (m * w) * p0;
                        *p = -m * w * s * q0 + c * p0;
                    }
                }
            } else {
                for ((q, p), &m) in pairs.zip(masses) {
                    for (q, p) in q.iter_mut().zip(p.iter()) {
                        *q += t * p / m;
                    }
                }
            }
        }
    }

    /// T-1 on normal-mode momenta `pt`; draws k outermost, then atom i, then d. No-op for `Off`.
    fn thermostat_modes(&self, pt: &mut Beads, masses: &[f64], rng: &mut Rng) {
        if self.thermostat == Thermostat::Off {
            return;
        }
        let pf = self.modes.n_beads() as f64;
        let n3 = masses.len() * 3;
        let ps = pt.as_mut_slice();
        for (k, &c) in self.c_k.iter().enumerate() {
            for (p, &m) in ps[k * n3..(k + 1) * n3].chunks_exact_mut(3).zip(masses) {
                let amp = ((1.0 - c * c) * m * pf / self.beta).sqrt();
                for x in p {
                    *x = c * *x + amp * rng.normal();
                }
            }
        }
    }
}

/// `IntegratorError::ShapeMismatch` unless `ev` has forces of shape `[p, n, 3]` and `p`
/// energies (P-1).
fn check_evaluation(ev: &Evaluation, p: usize, n: usize) -> Result<(), IntegratorError> {
    for (what, expected, actual) in [
        ("bead count of evaluated forces", p, ev.forces.n_beads()),
        ("atom count of evaluated forces", n, ev.forces.n_atoms()),
        ("number of evaluated energies", p, ev.energies.len()),
    ] {
        if expected != actual {
            return Err(IntegratorError::ShapeMismatch {
                what,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

/// p ← p + h·F, element-wise (the B step of I-1 with h = dt/2).
fn kick(p: &mut Beads, f: &Beads, h: f64) {
    for (p, f) in p.as_mut_slice().iter_mut().zip(f.as_slice()) {
        *p += h * f;
    }
}

/// A ring-polymer simulation: state, current forces and energies, integrator and potential.
pub struct RingPolymerSim {
    state: RingState,
    forces: Beads,
    energies: Vec<f64>,
    integrator: RingPolymerIntegrator,
    potential: Box<dyn Potential>,
    cell: Option<Cell>,
}

impl RingPolymerSim {
    /// Builds the simulation and performs the initialising `evaluate` (I-3).
    ///
    /// Errors: `IntegratorError::ShapeMismatch` if `state.q`, `state.p`, `state.masses` and the
    /// integrator's P disagree, or if the evaluation's forces are not `[P, N, 3]` or its energies
    /// not length P (P-1); `IntegratorError::InvalidValue` if a mass is not positive and finite
    /// (S-2); `IntegratorError::Potential` if the evaluation fails.
    pub fn new(
        state: RingState,
        cell: Option<Cell>,
        integrator: RingPolymerIntegrator,
        mut potential: Box<dyn Potential>,
    ) -> Result<Self, IntegratorError> {
        let (p, n) = (state.q.n_beads(), state.q.n_atoms());
        for (what, expected, actual) in [
            (
                "bead count of q vs integrator",
                integrator.modes.n_beads(),
                p,
            ),
            ("bead count of p", p, state.p.n_beads()),
            ("atom count of p", n, state.p.n_atoms()),
            ("number of masses", n, state.masses.len()),
        ] {
            if expected != actual {
                return Err(IntegratorError::ShapeMismatch {
                    what,
                    expected,
                    actual,
                });
            }
        }
        for &m in &state.masses {
            positive_finite("mass (m_e)", m)?;
        }
        let ev = potential.evaluate(&state.q, cell.as_ref(), None)?;
        check_evaluation(&ev, p, n)?;
        Ok(RingPolymerSim {
            state,
            forces: ev.forces,
            energies: ev.energies,
            integrator,
            potential,
            cell,
        })
    }

    /// One step B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2) (I-1), with exactly one `evaluate` (I-3).
    /// A O A run in normal-mode coordinates (one transform each way per step).
    ///
    /// Errors: `IntegratorError::Potential` if the evaluation fails, or
    /// `IntegratorError::ShapeMismatch` if its forces are not `[P, N, 3]` or its energies not
    /// length P (P-1). The simulation is then left unchanged, but the caller's `rng` has already
    /// advanced: P·N·3 normal draws under PILE-L, none under `Off` (D-4).
    pub fn step(&mut self, rng: &mut Rng) -> Result<(), IntegratorError> {
        let ig = &self.integrator;
        let half = 0.5 * ig.dt;
        let masses = &self.state.masses;
        let mut p = self.state.p.clone();
        kick(&mut p, &self.forces, half);
        let mut qt = ig.modes.to_normal(&self.state.q);
        let mut pt = ig.modes.to_normal(&p);
        ig.rotate_modes(&mut qt, &mut pt, masses, half);
        ig.thermostat_modes(&mut pt, masses, rng);
        ig.rotate_modes(&mut qt, &mut pt, masses, half);
        let q = ig.modes.to_beads(&qt);
        let mut p = ig.modes.to_beads(&pt);
        let ev = self.potential.evaluate(&q, self.cell.as_ref(), None)?;
        check_evaluation(&ev, q.n_beads(), q.n_atoms())?;
        kick(&mut p, &ev.forces, half);
        self.state.q = q;
        self.state.p = p;
        self.forces = ev.forces;
        self.energies = ev.energies;
        Ok(())
    }

    /// The current state (bead representation).
    pub fn state(&self) -> &RingState {
        &self.state
    }

    /// Forces −∇V at the current positions, shape `[P, N, 3]`, Eh/bohr (S-3).
    pub fn forces(&self) -> &Beads {
        &self.forces
    }

    /// V(q_j) per bead at the current positions, length P, Eh (S-4).
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }
}
