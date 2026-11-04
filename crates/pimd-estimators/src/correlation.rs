//! Centroid time-correlation functions and Green–Kubo diffusion (SPEC Y-3, Y-4, Y-5).

use pimd_core::Beads;

use crate::error::EstimatorError;

/// Centroid velocities v̄_i = (1/P) Σ_j p_{j,i}/m_i, length N, bohr per ħ/Eh (Y-3).
///
/// Panics if `masses.len()` differs from the atom count of `p`.
pub fn centroid_velocity(p: &Beads, masses: &[f64]) -> Vec<[f64; 3]> {
    let _ = (p, masses);
    unimplemented!("M7: Y-3")
}

/// Running Green–Kubo integral D(τ_n) = (1/3) ∫ C_vv by the trapezoid rule (Y-5); same length as
/// `c_vv`, D(0) = 0. `dt_lag` is the lag spacing in ħ/Eh. Units bohr²/(ħ/Eh) for C_vv in
/// (bohr/(ħ/Eh))².
pub fn green_kubo(c_vv: &[f64], dt_lag: f64) -> Vec<f64> {
    let _ = (c_vv, dt_lag);
    unimplemented!("M7: Y-5")
}

/// Segment-averaged autocorrelation of a per-atom 3-vector (Y-4).
#[derive(Debug, Clone, PartialEq)]
pub struct CentroidCorrelation {
    n_lags: usize,
    n_atoms: usize,
    samples: Vec<[f64; 3]>,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    n_segments: usize,
}

impl CentroidCorrelation {
    /// Empty accumulator for lags 0 … `n_lags` − 1 of `n_atoms` atoms.
    ///
    /// Errors: `EstimatorError::InvalidValue` for n_lags = 0 or n_atoms = 0.
    pub fn new(n_lags: usize, n_atoms: usize) -> Result<CentroidCorrelation, EstimatorError> {
        let _ = (n_lags, n_atoms);
        unimplemented!("M7: Y-4")
    }

    /// Appends one sample of the current segment.
    ///
    /// Errors: `EstimatorError::ShapeMismatch` if `x.len()` ≠ n_atoms.
    pub fn push(&mut self, x: &[[f64; 3]]) -> Result<(), EstimatorError> {
        let _ = x;
        unimplemented!("M7: Y-4")
    }

    /// Closes the current segment: adds its C_seg (Y-4) and clears the samples.
    ///
    /// Errors: `EstimatorError::InvalidValue` if it has fewer than n_lags samples (nothing
    /// changes).
    pub fn end_segment(&mut self) -> Result<(), EstimatorError> {
        unimplemented!("M7: Y-4")
    }

    /// Number of closed segments.
    pub fn n_segments(&self) -> usize {
        unimplemented!("M7: Y-4")
    }

    /// Mean over segments, length n_lags; NaN before the first segment.
    pub fn mean(&self) -> Vec<f64> {
        unimplemented!("M7: Y-4")
    }

    /// σ of the mean (Y-4), length n_lags; NaN with fewer than two segments.
    pub fn sem(&self) -> Vec<f64> {
        unimplemented!("M7: Y-4")
    }

    /// Accumulators only (n_lags, n_atoms, Σ, Σ², n_segments), little-endian; valid only between
    /// segments (no samples pending).
    ///
    /// Errors: `EstimatorError::InvalidValue` if samples are pending.
    pub fn to_bytes(&self) -> Result<Vec<u8>, EstimatorError> {
        unimplemented!("M7: Y-4")
    }

    /// Inverse of `to_bytes`.
    ///
    /// Errors: `EstimatorError::InvalidValue` for a malformed or truncated buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<CentroidCorrelation, EstimatorError> {
        let _ = bytes;
        unimplemented!("M7: Y-4")
    }
}
