//! Classical kinetic energy and pressure (SPEC E-8), P = 1.

use pimd_core::{Beads, Cell};

use crate::error::EstimatorError;

/// K = Σ_i |p_i|²/(2 m_i), Eh, for momenta `p` (`[1, N, 3]`, atomic units) and `masses` (m_e).
///
/// Panics if `masses.len()` differs from N or `p` has P ≠ 1 (a programming error, as for the
/// other energy estimators, Q4, Q10).
pub fn kinetic_classical(masses: &[f64], p: &Beads) -> f64 {
    assert_eq!(p.n_beads(), 1, "kinetic_classical needs P = 1");
    assert_eq!(masses.len(), p.n_atoms(), "number of masses vs atoms");
    masses
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let v = p.get(0, i);
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / (2.0 * m)
        })
        .sum()
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
    for (what, expected, actual) in [
        ("bead count of momenta (P = 1)", 1, p.n_beads()),
        ("number of masses", p.n_atoms(), masses.len()),
    ] {
        if expected != actual {
            return Err(EstimatorError::ShapeMismatch {
                what,
                expected,
                actual,
            });
        }
    }
    let k = kinetic_classical(masses, p);
    let tr_w = virial[0][0] + virial[1][1] + virial[2][2];
    Ok((2.0 * k + tr_w) / (3.0 * cell.volume()))
}
