//! Verlet neighbor list (SPEC N-1, N-2, C-4).

use crate::cell::Cell;
use crate::error::CoreError;

/// Verlet list of full pairs (N-1): `neighbors(i)` holds every j ≠ i within `cutoff + skin` of i
/// (minimum image when built with a cell), sorted ascending. Remembers the positions it was built
/// from for the rebuild criterion (N-2).
///
/// Lengths in bohr. Positions are `[N]` 3-vectors in bohr, unwrapped (C-5).
#[derive(Debug, Clone)]
pub struct NeighborList {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    cutoff: f64,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    skin: f64,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    pairs: Vec<Vec<usize>>,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    reference: Vec<[f64; 3]>,
}

impl NeighborList {
    /// Empty list for `cutoff` (bohr, > 0, finite) and `skin` (bohr, ≥ 0, finite).
    ///
    /// Errors with `CoreError::InvalidValue` otherwise.
    pub fn new(cutoff: f64, skin: f64) -> Result<NeighborList, CoreError> {
        let _ = (cutoff, skin);
        unimplemented!()
    }

    /// Rebuilds the list from `positions` (N 3-vectors, bohr, unwrapped) and stores them as the
    /// reference for `needs_rebuild`. With `Some(cell)` distances use the minimum image (C-2) and
    /// `cutoff + skin ≤ L_min/2` is required (C-4); `None` means open boundaries.
    ///
    /// Errors with `CoreError::InvalidValue` if C-4 is violated.
    pub fn build(&mut self, positions: &[[f64; 3]], cell: Option<&Cell>) -> Result<(), CoreError> {
        let _ = (positions, cell);
        unimplemented!()
    }

    /// True when any atom has moved more than `skin/2` (Euclidean norm, bohr) from the positions of
    /// the last build, measured in unwrapped coordinates (N-2).
    pub fn needs_rebuild(&self, positions: &[[f64; 3]]) -> bool {
        let _ = positions;
        unimplemented!()
    }

    /// Neighbors of atom `i`, sorted ascending, never containing `i` (N-1).
    pub fn neighbors(&self, i: usize) -> &[usize] {
        let _ = i;
        unimplemented!()
    }

    /// Number of atoms in the last build.
    pub fn n_atoms(&self) -> usize {
        unimplemented!()
    }
}
