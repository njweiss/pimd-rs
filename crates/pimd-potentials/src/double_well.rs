//! Separable quartic double well (SPEC P-2).

use pimd_core::{Beads, Cell};

use crate::error::PotentialError;
use crate::potential::{Evaluation, Potential};

/// V = Σ_i Σ_d (a x_{i,d}⁴ − b x_{i,d}²) per bead (P-2); force −(4a x³ − 2b x) per component.
///
/// `a` in Eh/bohr⁴, `b` in Eh/bohr². Accepts any number of atoms.
#[derive(Debug, Clone)]
pub struct DoubleWell {
    a: f64,
    b: f64,
}

impl DoubleWell {
    /// `a` and `b` positive and finite; errors with `PotentialError::InvalidParameter` otherwise.
    pub fn new(a: f64, b: f64) -> Result<DoubleWell, PotentialError> {
        for (what, v) in [("a (Eh/bohr^4)", a), ("b (Eh/bohr^2)", b)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(PotentialError::InvalidParameter { what, value: v });
            }
        }
        Ok(DoubleWell { a, b })
    }
}

impl Potential for DoubleWell {
    /// `cell` is ignored: the double well is an external field, not periodic (P-2).
    fn evaluate(
        &mut self,
        positions: &Beads,
        _cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        if subset.is_some() {
            return Err(PotentialError::Unsupported);
        }
        let (p, n3) = (positions.n_beads(), positions.n_atoms() * 3);
        let mut energies = Vec::with_capacity(p);
        let mut forces = Beads::zeros(p, positions.n_atoms());
        let (xs, fs) = (positions.as_slice(), forces.as_mut_slice());
        for j in 0..p {
            let bead = j * n3..(j + 1) * n3;
            let mut e = 0.0;
            for (&x, f) in xs[bead.clone()].iter().zip(&mut fs[bead]) {
                let x2 = x * x;
                e += self.a * x2 * x2 - self.b * x2;
                *f = -(4.0 * self.a * x2 * x - 2.0 * self.b * x);
            }
            energies.push(e);
        }
        Ok(Evaluation {
            energies,
            forces,
            virial: None,
        })
    }
}
