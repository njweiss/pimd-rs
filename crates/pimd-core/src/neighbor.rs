//! Verlet neighbor list (SPEC N-1, N-2, C-4).

use crate::cell::Cell;
use crate::error::{CoreError, positive_finite};

/// Verlet list of full pairs (N-1): `neighbors(i)` holds every j ≠ i with r_ij < `cutoff + skin`
/// (strict; minimum image when built with a cell), sorted ascending. Remembers the positions it was built
/// from for the rebuild criterion (N-2).
///
/// Lengths in bohr. Positions are `[N]` 3-vectors in bohr, unwrapped (C-5).
#[derive(Debug, Clone)]
pub struct NeighborList {
    cutoff: f64,
    skin: f64,
    pairs: Vec<Vec<usize>>,
    reference: Vec<[f64; 3]>,
}

impl NeighborList {
    /// Empty list for `cutoff` (bohr, > 0, finite) and `skin` (bohr, ≥ 0, finite).
    ///
    /// Errors with `CoreError::InvalidValue` otherwise.
    pub fn new(cutoff: f64, skin: f64) -> Result<NeighborList, CoreError> {
        positive_finite("neighbor-list cutoff (bohr)", cutoff)?;
        if !(skin.is_finite() && skin >= 0.0) {
            return Err(CoreError::InvalidValue {
                what: "neighbor-list skin (bohr)",
                value: skin,
            });
        }
        Ok(NeighborList {
            cutoff,
            skin,
            pairs: Vec::new(),
            reference: Vec::new(),
        })
    }

    /// Rebuilds the list from `positions` (N 3-vectors, bohr, unwrapped) and stores them as the
    /// reference for `needs_rebuild`. With `Some(cell)` distances use the minimum image (C-2) and
    /// `cutoff + skin ≤ L_min/2` is required (C-4); `None` means open boundaries.
    ///
    /// Errors with `CoreError::InvalidValue` if C-4 is violated.
    pub fn build(&mut self, positions: &[[f64; 3]], cell: Option<&Cell>) -> Result<(), CoreError> {
        let rl = self.cutoff + self.skin;
        if let Some(c) = cell {
            let l_min = c.lengths().into_iter().fold(f64::INFINITY, f64::min);
            if rl > 0.5 * l_min {
                return Err(CoreError::InvalidValue {
                    what: "cutoff + skin (bohr), must be <= L_min/2",
                    value: rl,
                });
            }
        }
        let rl2 = rl * rl;
        // O(N²) scan; j ascending, so every list is sorted (N-1).
        self.pairs = positions
            .iter()
            .enumerate()
            .map(|(i, &ri)| {
                positions
                    .iter()
                    .enumerate()
                    .filter(|&(j, &rj)| {
                        let mut d = [rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]];
                        if let Some(c) = cell {
                            d = c.min_image(d);
                        }
                        j != i && d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < rl2
                    })
                    .map(|(j, _)| j)
                    .collect()
            })
            .collect();
        self.reference = positions.to_vec();
        Ok(())
    }

    /// True when any atom has moved more than `skin/2` (Euclidean norm, bohr) from the positions of
    /// the last build, measured in unwrapped coordinates, or when `positions.len()` differs from
    /// the last build's atom count (N-2, Q10).
    pub fn needs_rebuild(&self, positions: &[[f64; 3]]) -> bool {
        if positions.len() != self.reference.len() {
            return true;
        }
        let half2 = 0.25 * self.skin * self.skin;
        positions.iter().zip(&self.reference).any(|(r, r0)| {
            let d = [r[0] - r0[0], r[1] - r0[1], r[2] - r0[2]];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > half2
        })
    }

    /// Neighbors of atom `i`, sorted ascending, never containing `i` (N-1).
    ///
    /// Panics if `i >= n_atoms()`, including before the first build (when `n_atoms()` is 0).
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.pairs[i]
    }

    /// Number of atoms in the last build.
    pub fn n_atoms(&self) -> usize {
        self.reference.len()
    }
}
