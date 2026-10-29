//! Classical velocity Verlet for a single replica (SPEC I-4, I-5).

use pimd_core::{Beads, Cell};
use pimd_potentials::Potential;

use crate::error::IntegratorError;
use crate::ring_polymer::RingState;

/// Standard velocity Verlet with time step `dt` in ħ/Eh (I-4), for states with P = 1.
#[derive(Debug, Clone)]
pub struct VelocityVerlet {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    dt: f64,
}

impl VelocityVerlet {
    /// Errors with `IntegratorError::InvalidValue` unless `dt` (ħ/Eh) is positive and finite (I-5).
    pub fn new(dt: f64) -> Result<VelocityVerlet, IntegratorError> {
        let _ = dt;
        unimplemented!()
    }

    /// One step: p ← p + (dt/2) F; q ← q + dt p/m; F ← F(q) (one `evaluate`); p ← p + (dt/2) F.
    ///
    /// `forces` holds F at the current positions on entry and F at the new positions on exit.
    /// Returns the potential energy at the new positions. Requires state with exactly one bead.
    /// Positions are never wrapped (C-5).
    ///
    /// Errors: `IntegratorError::ShapeMismatch` if the state has P ≠ 1, or `state.p`, `forces`,
    /// `state.masses` disagree with `state.q`; `IntegratorError::Potential` if the evaluation fails.
    pub fn step(
        &self,
        state: &mut RingState,
        forces: &mut Beads,
        potential: &mut dyn Potential,
        cell: Option<&Cell>,
    ) -> Result<f64, IntegratorError> {
        let _ = (state, forces, potential, cell);
        unimplemented!()
    }
}
