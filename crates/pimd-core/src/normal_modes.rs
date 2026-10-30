//! Ring-polymer normal-mode transform (SPEC RP-4, RP-5).

use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlannerScalar};

use crate::beads::Beads;
use crate::error::{CoreError, positive_finite};

/// How `NormalModes::to_normal` / `to_beads` are computed (RP-4). Both give the same RP-4
/// columns and normalisation; they agree to rounding, not bitwise (D-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transform {
    /// Dense P × P matrix product with C (the M1 transform).
    #[default]
    Matrix,
    /// Real FFT of length P, mapped onto the RP-4 columns.
    Fft,
}

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
    /// Length-P forward and inverse FFT plans for `Transform::Fft`, `None` for `Matrix`.
    fft: Option<FftPlans>,
}

/// Forward and inverse length-P plans from `FftPlannerScalar`, so the algorithm and its results
/// do not depend on CPU features (D-3, MILESTONE_4).
#[derive(Clone)]
struct FftPlans {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl fmt::Debug for FftPlans {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FftPlans")
    }
}

impl NormalModes {
    /// Transform for `n_beads` = P ≥ 1 beads at inverse temperature `beta` (Eh⁻¹, > 0, finite).
    ///
    /// Errors: `CoreError::InvalidBeadCount` for P = 0, `CoreError::InvalidValue` for bad β.
    pub fn new(n_beads: usize, beta: f64) -> Result<NormalModes, CoreError> {
        if n_beads == 0 {
            return Err(CoreError::InvalidBeadCount(n_beads));
        }
        positive_finite("beta (1/Eh)", beta)?;
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
            fft: None,
        })
    }

    /// As `new`, with the transform algorithm `transform` (RP-4). `new(p, beta)` is
    /// `with_transform(p, beta, Transform::Matrix)`. `frequencies()` and `matrix()` do not depend
    /// on `transform`.
    ///
    /// Errors: as `new`.
    pub fn with_transform(
        n_beads: usize,
        beta: f64,
        transform: Transform,
    ) -> Result<NormalModes, CoreError> {
        let mut nm = NormalModes::new(n_beads, beta)?;
        if transform == Transform::Fft {
            let mut planner = FftPlannerScalar::new();
            nm.fft = Some(FftPlans {
                forward: planner.plan_fft_forward(n_beads),
                inverse: planner.plan_fft_inverse(n_beads),
            });
        }
        Ok(nm)
    }

    /// The transform algorithm used by `to_normal` / `to_beads`.
    pub fn transform(&self) -> Transform {
        if self.fft.is_some() {
            Transform::Fft
        } else {
            Transform::Matrix
        }
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
        match &self.fft {
            None => self.apply_matrix(beads, false),
            Some(plans) => self.fft_to_normal(plans, beads),
        }
    }

    /// Normal-mode to bead coordinates, the inverse of `to_normal`: q_j = Σ_k C_jk q̃_k.
    ///
    /// Panics if `modes.n_beads()` differs from P.
    pub fn to_beads(&self, modes: &Beads) -> Beads {
        match &self.fft {
            None => self.apply_matrix(modes, true),
            Some(plans) => self.fft_to_beads(plans, modes),
        }
    }

    /// out_a = Σ_b M_ab x_b with M = Cᵀ (`inverse = false`) or C (`inverse = true`),
    /// summed over b in ascending order (deterministic, D-3).
    fn apply_matrix(&self, x: &Beads, inverse: bool) -> Beads {
        let p = self.n_beads;
        self.assert_beads(x);
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

    /// Panics unless `x` has P beads (RP-4).
    fn assert_beads(&self, x: &Beads) {
        assert_eq!(
            x.n_beads(),
            self.n_beads,
            "normal-mode transform built for P = {} beads, input has {}",
            self.n_beads,
            x.n_beads()
        );
    }

    /// `to_normal` by FFT. With X_m = Σ_j q_j e^{−2πi m j/P} per atom and component:
    /// q̃_0 = X_0/√P, q̃_{2m−1} = √(2/P) Re X_m, q̃_{2m} = −√(2/P) Im X_m, and for even P
    /// q̃_{P−1} = X_{P/2}/√P (RP-4 columns and normalisation).
    fn fft_to_normal(&self, plans: &FftPlans, x: &Beads) -> Beads {
        self.assert_beads(x);
        let p = self.n_beads;
        let n3 = x.n_atoms() * 3;
        let src = x.as_slice();
        // Transpose to one contiguous length-P series per (atom, component), followed by the FFT
        // scratch space so one allocation serves both.
        let scratch_len = plans.forward.get_inplace_scratch_len();
        let mut buf = Vec::with_capacity(n3 * p + scratch_len);
        for e in 0..n3 {
            buf.extend((0..p).map(|j| Complex::new(src[j * n3 + e], 0.0)));
        }
        buf.resize(n3 * p + scratch_len, Complex::new(0.0, 0.0));
        let (buf, scratch) = buf.split_at_mut(n3 * p);
        plans.forward.process_with_scratch(buf, scratch);
        let pf = p as f64;
        let (norm0, norm) = (1.0 / pf.sqrt(), (2.0 / pf).sqrt());
        let mut out = Beads::zeros(p, x.n_atoms());
        let dst = out.as_mut_slice();
        for (e, series) in buf.chunks_exact(p).enumerate() {
            dst[e] = norm0 * series[0].re;
            for m in 1..=(p - 1) / 2 {
                dst[(2 * m - 1) * n3 + e] = norm * series[m].re;
                dst[2 * m * n3 + e] = -norm * series[m].im;
            }
            if p.is_multiple_of(2) {
                dst[(p - 1) * n3 + e] = norm0 * series[p / 2].re;
            }
        }
        out
    }

    /// `to_beads` by inverse FFT: q_j = Re Σ_m Z_m e^{+2πi m j/P} with the Hermitian spectrum
    /// Z_0 = q̃_0/√P, Z_m = conj(Z_{P−m}) = (q̃_{2m−1} − i q̃_{2m})/√(2P), and for even P
    /// Z_{P/2} = q̃_{P−1}/√P, which reproduces q_j = Σ_k C_jk q̃_k (RP-4).
    fn fft_to_beads(&self, plans: &FftPlans, modes: &Beads) -> Beads {
        self.assert_beads(modes);
        let p = self.n_beads;
        let n3 = modes.n_atoms() * 3;
        let src = modes.as_slice();
        let pf = p as f64;
        let (norm0, half) = (1.0 / pf.sqrt(), 1.0 / (2.0 * pf).sqrt());
        let scratch_len = plans.inverse.get_inplace_scratch_len();
        let mut buf = vec![Complex::new(0.0, 0.0); n3 * p + scratch_len];
        let (buf, scratch) = buf.split_at_mut(n3 * p);
        for (e, series) in buf.chunks_exact_mut(p).enumerate() {
            series[0] = Complex::new(norm0 * src[e], 0.0);
            for m in 1..=(p - 1) / 2 {
                let z = Complex::new(
                    half * src[(2 * m - 1) * n3 + e],
                    -half * src[2 * m * n3 + e],
                );
                series[m] = z;
                series[p - m] = z.conj();
            }
            if p.is_multiple_of(2) {
                series[p / 2] = Complex::new(norm0 * src[(p - 1) * n3 + e], 0.0);
            }
        }
        plans.inverse.process_with_scratch(buf, scratch);
        let mut out = Beads::zeros(p, modes.n_atoms());
        let dst = out.as_mut_slice();
        for (e, series) in buf.chunks_exact(p).enumerate() {
            for (j, z) in series.iter().enumerate() {
                dst[j * n3 + e] = z.re;
            }
        }
        out
    }
}
