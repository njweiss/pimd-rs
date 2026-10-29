//! Lennard-Jones pair potential with per-bead Verlet lists (SPEC P-3, P-5, N-3, C-4, D-3).

use pimd_core::{Beads, Cell, NeighborList};
use rayon::prelude::*;

use crate::error::PotentialError;
use crate::potential::{Evaluation, Potential};

/// Lennard-Jones parameters (P-3): `epsilon` in Eh, `sigma`, `cutoff` r_c and neighbor-list `skin`
/// in bohr. With `shift`, V(r_c) is subtracted from the pair energy for r < r_c (forces unchanged).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LjParams {
    /// Well depth ε, Eh, > 0.
    pub epsilon: f64,
    /// Length scale σ, bohr, > 0.
    pub sigma: f64,
    /// Cutoff r_c, bohr, > 0: pairs with r ≥ r_c contribute nothing.
    pub cutoff: f64,
    /// Verlet skin, bohr, ≥ 0 (N-1, N-2).
    pub skin: f64,
    /// Subtract V(r_c) inside r_c (P-3).
    pub shift: bool,
}

/// V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] for r < r_c, optionally shifted, no tail correction (P-3).
///
/// Keeps one `NeighborList` per bead (N-3), rebuilt inside `evaluate` when `needs_rebuild` is true
/// or the atom count changed. Positions passed are unwrapped (C-5); the minimum image is applied
/// internally when a cell is given. Accepts any number of atoms.
#[derive(Debug, Clone)]
pub struct LennardJones {
    params: LjParams,
    lists: Vec<NeighborList>,
    /// Empty list with this cutoff and skin, cloned when the bead count changes.
    template: NeighborList,
}

impl LennardJones {
    /// Errors with `PotentialError::InvalidParameter` unless ε, σ and r_c are positive and finite
    /// and the skin is non-negative and finite.
    pub fn new(params: LjParams) -> Result<LennardJones, PotentialError> {
        for (what, v) in [
            ("LJ epsilon (Eh)", params.epsilon),
            ("LJ sigma (bohr)", params.sigma),
            ("LJ cutoff (bohr)", params.cutoff),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(PotentialError::InvalidParameter { what, value: v });
            }
        }
        let template = NeighborList::new(params.cutoff, params.skin).map_err(|_| {
            PotentialError::InvalidParameter {
                what: "LJ skin (bohr)",
                value: params.skin,
            }
        })?;
        Ok(LennardJones {
            params,
            lists: Vec::new(),
            template,
        })
    }

    /// Atom `i`'s share of one bead: (½ Σ_j V(r_ij), Σ_j f_ij, ½ Σ_j r_ij ⊗ f_ij) over its full
    /// neighbor list in ascending j (D-3), with r_ij = r_i − r_j (minimum image when `cell` is
    /// given) and only pairs with r_ij < r_c (P-3).
    fn atom_terms(
        &self,
        i: usize,
        x: &[[f64; 3]],
        neighbors: &[usize],
        cell: Option<&Cell>,
        v_shift: f64,
    ) -> (f64, [f64; 3], [[f64; 3]; 3]) {
        let LjParams {
            epsilon,
            sigma,
            cutoff,
            ..
        } = self.params;
        let (rc2, sig2) = (cutoff * cutoff, sigma * sigma);
        let mut e = 0.0;
        let mut f = [0.0; 3];
        let mut w = [[0.0; 3]; 3];
        for &j in neighbors {
            let mut d = [x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]];
            if let Some(c) = cell {
                d = c.min_image(d);
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 >= rc2 {
                continue;
            }
            let s6 = (sig2 / r2).powi(3);
            e += 4.0 * epsilon * (s6 * s6 - s6) - v_shift;
            // F(r)/r with F = −dV/dr.
            let f_over_r = 24.0 * epsilon * (2.0 * s6 * s6 - s6) / r2;
            for a in 0..3 {
                f[a] += f_over_r * d[a];
                for b in 0..3 {
                    w[a][b] += f_over_r * d[a] * d[b];
                }
            }
        }
        for row in &mut w {
            for v in row {
                *v *= 0.5;
            }
        }
        (0.5 * e, f, w)
    }
}

impl Potential for LennardJones {
    /// Per-bead energies (Eh) and forces (Eh/bohr) for `positions` `[P, N, 3]` (bohr, unwrapped).
    /// `virial` is `Some(W)` (Eh, W = Σ_{i<j} r_ij ⊗ f_ij, E-8) for P = 1 and `None` for P > 1
    /// (P-5). Each atom's force and energy come from its own full neighbor list and per-atom values
    /// are reduced in index order, so results do not depend on the thread count (D-3).
    ///
    /// Errors: `PotentialError::Unsupported` for `subset = Some` (P-4);
    /// `PotentialError::InvalidParameter` when a cell is given and `cutoff + skin > L_min/2` (C-4,
    /// Q7).
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        if subset.is_some() {
            return Err(PotentialError::Unsupported);
        }
        let LjParams {
            epsilon,
            sigma,
            cutoff,
            skin,
            shift,
        } = self.params;
        if let Some(c) = cell {
            let l_min = c.lengths().into_iter().fold(f64::INFINITY, f64::min);
            if cutoff + skin > 0.5 * l_min {
                return Err(PotentialError::InvalidParameter {
                    what: "cutoff + skin (bohr), must be <= L_min/2",
                    value: cutoff + skin,
                });
            }
        }
        let v_shift = if shift {
            let s6 = (sigma / cutoff).powi(6);
            4.0 * epsilon * (s6 * s6 - s6)
        } else {
            0.0
        };
        let (p, n) = (positions.n_beads(), positions.n_atoms());
        let fresh = self.lists.len() != p;
        if fresh {
            self.lists = vec![self.template.clone(); p];
        }
        let mut energies = Vec::with_capacity(p);
        let mut forces = Beads::zeros(p, n);
        let mut virial = [[0.0; 3]; 3];
        for j in 0..p {
            let x: Vec<[f64; 3]> = (0..n).map(|i| positions.get(j, i)).collect();
            if fresh || self.lists[j].needs_rebuild(&x) {
                // C-4 was checked above, so the build cannot fail.
                self.lists[j]
                    .build(&x, cell)
                    .map_err(|_| PotentialError::InvalidParameter {
                        what: "cutoff + skin (bohr), must be <= L_min/2",
                        value: cutoff + skin,
                    })?;
            }
            let list = &self.lists[j];
            // Per-atom terms in parallel, collected in index order, then reduced sequentially (D-3).
            let terms: Vec<_> = (0..n)
                .into_par_iter()
                .map(|i| self.atom_terms(i, &x, list.neighbors(i), cell, v_shift))
                .collect();
            let mut e = 0.0;
            for (i, (ei, fi, wi)) in terms.iter().enumerate() {
                e += ei;
                forces.set(j, i, *fi);
                if p == 1 {
                    for a in 0..3 {
                        for b in 0..3 {
                            virial[a][b] += wi[a][b];
                        }
                    }
                }
            }
            energies.push(e);
        }
        Ok(Evaluation {
            energies,
            forces,
            virial: (p == 1).then_some(virial),
        })
    }
}
