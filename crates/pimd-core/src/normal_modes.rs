//! Ring-polymer normal-mode transform (SPEC RP-4, RP-5).

use std::f64::consts::PI;

use crate::beads::Beads;
use crate::error::CoreError;

/// Orthogonal normal-mode transform of a P-bead ring polymer (RP-4).
///
/// `matrix()` is C, row j = bead, column k = mode, in the fixed RP-4 column order;
/// `frequencies()` are the free ring-polymer frequencies ω_k (Eh, ħ = 1) aligned with the columns.
#[derive(Debug, Clone)]
pub struct NormalModes {
    n_beads: usize,
    omega_p: f64,
    matrix: Vec<f64>,
    freqs: Vec<f64>,
}

impl NormalModes {
    /// Transform for `n_beads` = P ≥ 1 beads at inverse temperature `beta` (Eh⁻¹, > 0, finite).
    ///
    /// Errors: `CoreError::InvalidBeadCount` for P = 0, `CoreError::InvalidValue` for bad β.
    pub fn new(n_beads: usize, beta: f64) -> Result<NormalModes, CoreError> {
        if n_beads == 0 {
            return Err(CoreError::InvalidBeadCount(n_beads));
        }
        if !(beta.is_finite() && beta > 0.0) {
            return Err(CoreError::InvalidValue {
                what: "beta (1/Eh)",
                value: beta,
            });
        }
        let p = n_beads;
        let pf = p as f64;
        let omega_p = pf / beta;
        let mut matrix = vec![0.0; p * p];
        let mut freqs = vec![0.0; p];
        let norm0 = 1.0 / pf.sqrt();
        let norm = (2.0 / pf).sqrt();
        for j in 0..p {
            let jf = j as f64;
            matrix[j * p] = norm0;
            for m in 1..=(p - 1) / 2 {
                let arg = 2.0 * PI * (m as f64) * jf / pf;
                matrix[j * p + 2 * m - 1] = norm * arg.cos();
                matrix[j * p + 2 * m] = norm * arg.sin();
            }
            if p.is_multiple_of(2) {
                let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
                matrix[j * p + p - 1] = sign * norm0;
            }
        }
        for m in 1..=(p - 1) / 2 {
            let w = 2.0 * omega_p * ((m as f64) * PI / pf).sin();
            freqs[2 * m - 1] = w;
            freqs[2 * m] = w;
        }
        if p.is_multiple_of(2) {
            freqs[p - 1] = 2.0 * omega_p;
        }
        Ok(NormalModes {
            n_beads,
            omega_p,
            matrix,
            freqs,
        })
    }

    /// Number of beads P.
    pub fn n_beads(&self) -> usize {
        self.n_beads
    }

    /// ω_P = P/β, in Eh (RP-2).
    pub fn omega_p(&self) -> f64 {
        self.omega_p
    }

    /// Mode frequencies ω_k, length P, aligned with the columns of `matrix()` (RP-4).
    pub fn frequencies(&self) -> &[f64] {
        &self.freqs
    }

    /// C, row-major P × P: C[j][k] at `j*P + k` (RP-4).
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Bead to normal-mode coordinates: q̃_k = Σ_j C_jk q_j, per atom and component.
    ///
    /// Same shape `[P, N, 3]` as the input, mode-major (mode k in place of bead j).
    ///
    /// Panics if `beads.n_beads()` differs from P.
    pub fn to_normal(&self, beads: &Beads) -> Beads {
        self.transform(beads, false)
    }

    /// Normal-mode to bead coordinates, the inverse of `to_normal`: q_j = Σ_k C_jk q̃_k.
    ///
    /// Panics if `modes.n_beads()` differs from P.
    pub fn to_beads(&self, modes: &Beads) -> Beads {
        self.transform(modes, true)
    }

    /// out_a = Σ_b M_ab x_b with M = Cᵀ (`inverse = false`) or C (`inverse = true`),
    /// summed over b in ascending order (deterministic, D-3).
    fn transform(&self, x: &Beads, inverse: bool) -> Beads {
        let p = self.n_beads;
        assert_eq!(
            x.n_beads(),
            p,
            "normal-mode transform built for P = {p} beads, input has {}",
            x.n_beads()
        );
        let n3 = x.n_atoms() * 3;
        let src = x.as_slice();
        let mut out = Beads::zeros(p, x.n_atoms());
        let dst = out.as_mut_slice();
        for a in 0..p {
            let row = &mut dst[a * n3..(a + 1) * n3];
            for b in 0..p {
                let c = if inverse {
                    self.matrix[a * p + b]
                } else {
                    self.matrix[b * p + a]
                };
                for (o, &v) in row.iter_mut().zip(&src[b * n3..(b + 1) * n3]) {
                    *o += c * v;
                }
            }
        }
        out
    }
}
