//! Separable quartic double well (SPEC P-2).

use pimd_core::{Beads, Cell};

use crate::error::PotentialError;
use crate::potential::{Evaluation, Potential};

/// V = Σ_i Σ_d (a x_{i,d}⁴ − b x_{i,d}²) per bead (P-2); force −(4a x³ − 2b x) per component.
///
/// `a` in Eh/bohr⁴, `b` in Eh/bohr². Accepts any number of atoms.
#[derive(Debug, Clone)]
pub struct DoubleWell {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    a: f64,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    b: f64,
}

impl DoubleWell {
    /// `a` and `b` positive and finite; errors with `PotentialError::InvalidParameter` otherwise.
    pub fn new(a: f64, b: f64) -> Result<DoubleWell, PotentialError> {
        let _ = (a, b);
        unimplemented!()
    }
}

impl Potential for DoubleWell {
    /// `cell` is ignored: the double well is an external field, not periodic (P-2).
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
