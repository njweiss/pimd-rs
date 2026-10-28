//! Ring-polymer normal-mode transform (SPEC RP-4, RP-5).

use crate::beads::Beads;
use crate::error::CoreError;

/// Orthogonal normal-mode transform of a P-bead ring polymer (RP-4).
///
/// `matrix()` is C, row j = bead, column k = mode, in the fixed RP-4 column order;
/// `frequencies()` are the free ring-polymer frequencies ω_k (Eh, ħ = 1) aligned with the columns.
#[derive(Debug, Clone)]
pub struct NormalModes {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    n_beads: usize,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    omega_p: f64,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    matrix: Vec<f64>,
    #[allow(dead_code)] // Phase A stub: used once implemented.
    freqs: Vec<f64>,
}

impl NormalModes {
    /// Transform for `n_beads` = P ≥ 1 beads at inverse temperature `beta` (Eh⁻¹, > 0, finite).
    ///
    /// Errors: `CoreError::InvalidBeadCount` for P = 0, `CoreError::InvalidValue` for bad β.
    pub fn new(n_beads: usize, beta: f64) -> Result<NormalModes, CoreError> {
        let _ = (n_beads, beta);
        unimplemented!()
    }

    /// Number of beads P.
    pub fn n_beads(&self) -> usize {
        unimplemented!()
    }

    /// ω_P = P/β, in Eh (RP-2).
    pub fn omega_p(&self) -> f64 {
        unimplemented!()
    }

    /// Mode frequencies ω_k, length P, aligned with the columns of `matrix()` (RP-4).
    pub fn frequencies(&self) -> &[f64] {
        unimplemented!()
    }

    /// C, row-major P × P: C[j][k] at `j*P + k` (RP-4).
    pub fn matrix(&self) -> &[f64] {
        unimplemented!()
    }

    /// Bead to normal-mode coordinates: q̃_k = Σ_j C_jk q_j, per atom and component.
    ///
    /// Same shape `[P, N, 3]` as the input, mode-major (mode k in place of bead j).
    pub fn to_normal(&self, beads: &Beads) -> Beads {
        let _ = beads;
        unimplemented!()
    }

    /// Normal-mode to bead coordinates, the inverse of `to_normal`: q_j = Σ_k C_jk q̃_k.
    pub fn to_beads(&self, modes: &Beads) -> Beads {
        let _ = modes;
        unimplemented!()
    }
}
