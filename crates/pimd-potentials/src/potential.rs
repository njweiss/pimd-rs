//! The model-agnostic `Potential` trait (SPEC P-1).

use pimd_core::{Beads, Cell};

use crate::error::PotentialError;

/// Result of one `Potential::evaluate` call (P-1).
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// Per-bead energy V(q_j), length P, in Eh (S-4).
    pub energies: Vec<f64>,
    /// Forces −∇V, shape `[P, N, 3]`, in Eh/bohr (S-3).
    pub forces: Beads,
    /// Virial tensor in Eh, when the potential provides one (P-5).
    pub virial: Option<[[f64; 3]; 3]>,
}

/// A potential energy surface evaluated bead by bead (P-1).
pub trait Potential {
    /// Energies and forces for every bead of `positions` (`[P, N, 3]`, bohr).
    ///
    /// `cell` is `None` for open boundaries. `subset` restricts the atoms evaluated;
    /// native potentials return `PotentialError::Unsupported` when it is `Some` (P-4).
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError>;
}
