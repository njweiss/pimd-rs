//! Lennard-Jones pair potential with per-bead Verlet lists (SPEC P-3, P-5, N-3, C-4, D-3).

use pimd_core::{Beads, Cell, NeighborList};

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
    #[allow(dead_code)] // Phase A stub: used once implemented.
    params: LjParams,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    lists: Vec<NeighborList>,
}

impl LennardJones {
    /// Errors with `PotentialError::InvalidParameter` unless ε, σ and r_c are positive and finite
    /// and the skin is non-negative and finite.
    pub fn new(params: LjParams) -> Result<LennardJones, PotentialError> {
        let _ = params;
        unimplemented!()
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
        let _ = (positions, cell, subset);
        unimplemented!()
    }
}
