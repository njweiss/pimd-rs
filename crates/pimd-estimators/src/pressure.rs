//! Classical kinetic energy and pressure (SPEC E-8), P = 1.

use pimd_core::{Beads, Cell};

use crate::error::EstimatorError;

/// K = Σ_i |p_i|²/(2 m_i), Eh, for momenta `p` (`[1, N, 3]`, atomic units) and `masses` (m_e).
///
/// Panics if `masses.len()` differs from N (a programming error, as for the other energy
/// estimators, Q4).
pub fn kinetic_classical(masses: &[f64], p: &Beads) -> f64 {
    let _ = (masses, p);
    unimplemented!()
}

/// p = (2K + tr W)/(3V) in Eh/bohr³ (E-8), with K from `kinetic_classical`, the virial `virial`
/// (Eh, W = Σ_{i<j} r_ij ⊗ f_ij) and V the cell volume (bohr³).
///
/// Errors with `EstimatorError::ShapeMismatch` if `p` has P ≠ 1 or `masses.len()` differs from N.
pub fn pressure_classical(
    masses: &[f64],
    p: &Beads,
    virial: [[f64; 3]; 3],
    cell: &Cell,
) -> Result<f64, EstimatorError> {
    let _ = (masses, p, virial, cell);
    unimplemented!()
}
