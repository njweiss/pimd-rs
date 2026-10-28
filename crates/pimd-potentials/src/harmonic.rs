//! Isotropic harmonic potential (SPEC P-2).

use pimd_core::{Beads, Cell};

use crate::error::PotentialError;
use crate::potential::{Evaluation, Potential};

/// V = Σ_i ½ m_i ω² |q_i|² per bead (P-2); force −m_i ω² q_i.
#[derive(Debug, Clone)]
pub struct Harmonic {
    masses: Vec<f64>,
    omega: f64,
}

impl Harmonic {
    /// `masses` in m_e, one per atom, each positive and finite (S-2); `omega` in Eh, ≥ 0 and finite.
    ///
    /// Errors with `PotentialError::InvalidParameter` otherwise.
    pub fn new(masses: Vec<f64>, omega: f64) -> Result<Harmonic, PotentialError> {
        for &m in &masses {
            if !(m.is_finite() && m > 0.0) {
                return Err(PotentialError::InvalidParameter {
                    what: "mass (m_e)",
                    value: m,
                });
            }
        }
        if !(omega.is_finite() && omega >= 0.0) {
            return Err(PotentialError::InvalidParameter {
                what: "omega (Eh)",
                value: omega,
            });
        }
        Ok(Harmonic { masses, omega })
    }
}

impl Potential for Harmonic {
    /// `cell` is ignored: the harmonic trap is an external field, not periodic (P-2).
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        let _ = cell;
        if subset.is_some() {
            return Err(PotentialError::Unsupported);
        }
        let n = self.masses.len();
        if positions.n_atoms() != n {
            return Err(PotentialError::ShapeMismatch {
                expected: n,
                actual: positions.n_atoms(),
            });
        }
        let p = positions.n_beads();
        let w2 = self.omega * self.omega;
        let mut energies = Vec::with_capacity(p);
        let mut forces = Beads::zeros(p, n);
        for j in 0..p {
            let mut e = 0.0;
            for (i, &m) in self.masses.iter().enumerate() {
                let q = positions.get(j, i);
                let k = m * w2;
                e += 0.5 * k * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
                forces.set(j, i, q.map(|x| -k * x));
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
