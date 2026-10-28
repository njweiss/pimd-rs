//! Ring-polymer state, BAOAB integrator with PILE-L (SPEC I-1 … I-5, T-1 … T-3).

use pimd_core::{Beads, Cell, NormalModes, Rng};
use pimd_potentials::Potential;

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
    #[allow(dead_code)] // Phase A stub: used once implemented.
    dt: f64,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    beta: f64,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    modes: NormalModes,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    thermostat: Thermostat,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    gamma_k: Vec<f64>,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    c_k: Vec<f64>,
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
        let _ = (dt, beta, n_beads, thermostat);
        unimplemented!()
    }

    /// Exact free ring-polymer propagation of `state` for time `t` (ħ/Eh, any sign), per normal
    /// mode and component (I-2), using `state.masses`. No potential forces.
    ///
    /// Panics if the bead count of `state` differs from P or `state.masses.len()` from N.
    pub fn free_propagate(&self, state: &mut RingState, t: f64) {
        let _ = (state, t);
        unimplemented!()
    }

    /// One O step of length dt (T-1): PILE-L on the normal-mode momenta, or the identity for
    /// `Thermostat::Off`. Draws one normal per (k, i, d), k outermost, then atom i, then d.
    ///
    /// Panics if the bead count of `state` differs from P or `state.masses.len()` from N.
    pub fn thermostat_step(&self, state: &mut RingState, rng: &mut Rng) {
        let _ = (state, rng);
        unimplemented!()
    }

    /// The normal-mode transform used by this integrator (RP-4).
    pub fn modes(&self) -> &NormalModes {
        unimplemented!()
    }
}

/// A ring-polymer simulation: state, current forces and energies, integrator and potential.
pub struct RingPolymerSim {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    state: RingState,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    forces: Beads,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    energies: Vec<f64>,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    integrator: RingPolymerIntegrator,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    potential: Box<dyn Potential>,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    cell: Option<Cell>,
}

impl RingPolymerSim {
    /// Builds the simulation and performs the initialising `evaluate` (I-3).
    ///
    /// Errors: `IntegratorError::ShapeMismatch` if `state.q`, `state.p`, `state.masses` and the
    /// integrator's P disagree; `IntegratorError::Potential` if the evaluation fails.
    pub fn new(
        state: RingState,
        cell: Option<Cell>,
        integrator: RingPolymerIntegrator,
        potential: Box<dyn Potential>,
    ) -> Result<Self, IntegratorError> {
        let _ = (state, cell, integrator, potential);
        unimplemented!()
    }

    /// One step B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2) (I-1), with exactly one `evaluate` (I-3).
    ///
    /// Errors: `IntegratorError::Potential` if the evaluation fails.
    pub fn step(&mut self, rng: &mut Rng) -> Result<(), IntegratorError> {
        let _ = rng;
        unimplemented!()
    }

    /// The current state (bead representation).
    pub fn state(&self) -> &RingState {
        unimplemented!()
    }

    /// Forces −∇V at the current positions, shape `[P, N, 3]`, Eh/bohr (S-3).
    pub fn forces(&self) -> &Beads {
        unimplemented!()
    }

    /// V(q_j) per bead at the current positions, length P, Eh (S-4).
    pub fn energies(&self) -> &[f64] {
        unimplemented!()
    }
}
