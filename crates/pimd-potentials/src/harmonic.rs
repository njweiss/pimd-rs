//! Isotropic harmonic potential (SPEC P-2).

use pimd_core::{Beads, Cell};

use crate::error::PotentialError;
use crate::potential::{Evaluation, Potential};

/// V = Σ_i ½ m_i ω² |q_i|² per bead (P-2); force −m_i ω² q_i.
#[derive(Debug, Clone)]
pub struct Harmonic {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    masses: Vec<f64>,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    omega: f64,
}

impl Harmonic {
    /// `masses` in m_e, one per atom, each positive and finite (S-2); `omega` in Eh, ≥ 0 and finite.
    ///
    /// Errors with `PotentialError::InvalidParameter` otherwise.
    pub fn new(masses: Vec<f64>, omega: f64) -> Result<Harmonic, PotentialError> {
        let _ = (masses, omega);
        unimplemented!()
    }
}

impl Potential for Harmonic {
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        let _ = (positions, cell, subset);
        unimplemented!()
    }
}
