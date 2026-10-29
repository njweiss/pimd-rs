//! Radial distribution function (SPEC E-7).

use pimd_core::{Beads, Cell};

use crate::error::EstimatorError;

/// Histogram of minimum-image pair distances (i < j, every bead) on `n_bins` equal bins on
/// [0, r_max] (bohr), accumulated over samples (E-7). Requires a cell and r_max ≤ L_min/2.
#[derive(Debug, Clone)]
pub struct Rdf {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    n_bins: usize,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    r_max: f64,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    counts: Vec<f64>,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    n_atoms: usize,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    cell: Cell,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    n_samples: usize,
}

impl Rdf {
    /// Empty RDF with `n_bins` ≥ 1 bins on [0, `r_max`] (bohr, > 0, ≤ L_min/2) for `n_atoms` atoms
    /// in `cell`.
    ///
    /// Errors with `EstimatorError::InvalidValue` for a bad bin count or `r_max`.
    pub fn new(
        n_bins: usize,
        r_max: f64,
        cell: Cell,
        n_atoms: usize,
    ) -> Result<Rdf, EstimatorError> {
        let _ = (n_bins, r_max, cell, n_atoms);
        unimplemented!()
    }

    /// Adds one configuration: all beads of `q` (`[P, N, 3]`, bohr, unwrapped), pairs i < j within
    /// each bead, minimum image (C-2).
    ///
    /// Errors with `EstimatorError::ShapeMismatch` if `q.n_atoms()` differs from `n_atoms`.
    pub fn accumulate(&mut self, q: &Beads) -> Result<(), EstimatorError> {
        let _ = q;
        unimplemented!()
    }

    /// Number of configurations accumulated.
    pub fn n_samples(&self) -> usize {
        unimplemented!()
    }

    /// Bin centres (r_b + r_{b+1})/2, bohr, length `n_bins`.
    pub fn bin_centers(&self) -> Vec<f64> {
        unimplemented!()
    }

    /// g_b = 2 C_b V / (P N (N − 1) V_shell(b)) (E-7), averaged over the accumulated samples.
    pub fn g(&self) -> Vec<f64> {
        unimplemented!()
    }
}
