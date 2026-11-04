//! Centroid time-correlation functions and Green–Kubo diffusion (SPEC Y-3, Y-4, Y-5).

use pimd_core::Beads;

use crate::error::EstimatorError;

/// Centroid velocities v̄_i = (1/P) Σ_j p_{j,i}/m_i, length N, bohr per ħ/Eh (Y-3).
///
/// Panics if `masses.len()` differs from the atom count of `p`.
pub fn centroid_velocity(p: &Beads, masses: &[f64]) -> Vec<[f64; 3]> {
    let n = p.n_atoms();
    assert_eq!(
        masses.len(),
        n,
        "centroid_velocity: {} masses for {n} atoms",
        masses.len()
    );
    // Bead-major traversal; each atom sums p_{j,i}/m_i over beads j = 0..P−1 in order.
    let mut sums = vec![[0.0; 3]; n];
    if n == 0 {
        return sums;
    }
    for slab in p.as_slice().chunks_exact(n * 3) {
        for ((sum, pj), &m) in sums.iter_mut().zip(slab.chunks_exact(3)).zip(masses) {
            for (s, x) in sum.iter_mut().zip(pj) {
                *s += x / m;
            }
        }
    }
    let pf = p.n_beads() as f64;
    sums.into_iter().map(|sum| sum.map(|s| s / pf)).collect()
}

/// Running Green–Kubo integral D(τ_n) = (1/3) ∫ C_vv by the trapezoid rule (Y-5); same length as
/// `c_vv`, D(0) = 0. `dt_lag` is the lag spacing in ħ/Eh. Units bohr²/(ħ/Eh) for C_vv in
/// (bohr/(ħ/Eh))².
pub fn green_kubo(c_vv: &[f64], dt_lag: f64) -> Vec<f64> {
    let mut d = Vec::with_capacity(c_vv.len());
    let mut acc = 0.0;
    for (k, &c) in c_vv.iter().enumerate() {
        if k > 0 {
            acc += 0.5 * (c_vv[k - 1] + c) * dt_lag / 3.0;
        }
        d.push(acc);
    }
    d
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
        if n_lags == 0 {
            return Err(EstimatorError::InvalidValue {
                what: "n_lags",
                value: 0.0,
            });
        }
        if n_atoms == 0 {
            return Err(EstimatorError::InvalidValue {
                what: "n_atoms",
                value: 0.0,
            });
        }
        Ok(CentroidCorrelation {
            n_lags,
            n_atoms,
            samples: Vec::new(),
            sum: vec![0.0; n_lags],
            sum_sq: vec![0.0; n_lags],
            n_segments: 0,
        })
    }

    /// Appends one sample of the current segment.
    ///
    /// Errors: `EstimatorError::ShapeMismatch` if `x.len()` ≠ n_atoms.
    pub fn push(&mut self, x: &[[f64; 3]]) -> Result<(), EstimatorError> {
        if x.len() != self.n_atoms {
            return Err(EstimatorError::ShapeMismatch {
                what: "atom count of a correlation sample",
                expected: self.n_atoms,
                actual: x.len(),
            });
        }
        self.samples.extend_from_slice(x);
        Ok(())
    }

    /// Closes the current segment: adds its C_seg (Y-4) and clears the samples.
    ///
    /// Errors: `EstimatorError::InvalidValue` if it has fewer than n_lags samples (nothing
    /// changes).
    pub fn end_segment(&mut self) -> Result<(), EstimatorError> {
        let n = self.n_atoms;
        let l = self.samples.len() / n;
        if l < self.n_lags {
            return Err(EstimatorError::InvalidValue {
                what: "number of samples in the segment (fewer than n_lags)",
                value: l as f64,
            });
        }
        let x = &self.samples;
        for tau in 0..self.n_lags {
            // Y-4: s outermost, then atom i, then dimension d.
            let mut acc = 0.0;
            for s in 0..l - tau {
                let (a, b) = (&x[s * n..(s + 1) * n], &x[(s + tau) * n..(s + tau + 1) * n]);
                for (xa, xb) in a.iter().zip(b) {
                    for d in 0..3 {
                        acc += xa[d] * xb[d];
                    }
                }
            }
            let c = acc / (n * (l - tau)) as f64;
            self.sum[tau] += c;
            self.sum_sq[tau] += c * c;
        }
        self.n_segments += 1;
        self.samples.clear();
        Ok(())
    }

    /// Number of closed segments.
    pub fn n_segments(&self) -> usize {
        self.n_segments
    }

    /// Mean over segments, length n_lags; NaN before the first segment.
    pub fn mean(&self) -> Vec<f64> {
        if self.n_segments == 0 {
            return vec![f64::NAN; self.n_lags];
        }
        let n = self.n_segments as f64;
        self.sum.iter().map(|s| s / n).collect()
    }

    /// σ of the mean (Y-4), length n_lags; NaN with fewer than two segments. A sample variance
    /// made slightly negative by rounding is taken as zero.
    pub fn sem(&self) -> Vec<f64> {
        if self.n_segments < 2 {
            return vec![f64::NAN; self.n_lags];
        }
        let n = self.n_segments as f64;
        self.sum
            .iter()
            .zip(&self.sum_sq)
            .map(|(&s, &s2)| {
                // Sample variance with n − 1 from Σ and Σ²; rounding can make it slightly
                // negative when the segments agree, so it is clamped at zero.
                let var = ((s2 - s * s / n) / (n - 1.0)).max(0.0);
                (var / n).sqrt()
            })
            .collect()
    }

    /// Accumulators only (n_lags, n_atoms, Σ, Σ², n_segments), little-endian; valid only between
    /// segments (no samples pending).
    ///
    /// Errors: `EstimatorError::InvalidValue` if samples are pending.
    pub fn to_bytes(&self) -> Result<Vec<u8>, EstimatorError> {
        if !self.samples.is_empty() {
            return Err(EstimatorError::InvalidValue {
                what: "pending samples at serialisation",
                value: (self.samples.len() / self.n_atoms) as f64,
            });
        }
        let mut out = Vec::with_capacity(8 * (3 + 2 * self.n_lags));
        out.extend_from_slice(&(self.n_lags as u64).to_le_bytes());
        out.extend_from_slice(&(self.n_atoms as u64).to_le_bytes());
        for v in self.sum.iter().chain(&self.sum_sq) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.n_segments as u64).to_le_bytes());
        Ok(out)
    }

    /// Inverse of `to_bytes`.
    ///
    /// Errors: `EstimatorError::InvalidValue` for a malformed or truncated buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<CentroidCorrelation, EstimatorError> {
        let malformed = |what| EstimatorError::InvalidValue {
            what,
            value: bytes.len() as f64,
        };
        let word = |k: usize| -> Option<[u8; 8]> { bytes.get(8 * k..8 * k + 8)?.try_into().ok() };
        let header = |k| {
            word(k)
                .map(u64::from_le_bytes)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(malformed("correlation buffer (truncated)"))
        };
        let n_lags = header(0)?;
        let n_atoms = header(1)?;
        if n_lags == 0 || n_atoms == 0 {
            return Err(malformed("correlation buffer (n_lags or n_atoms zero)"));
        }
        let words = n_lags
            .checked_mul(2)
            .and_then(|w| w.checked_add(3))
            .ok_or(malformed("correlation buffer (n_lags too large)"))?;
        if words.checked_mul(8) != Some(bytes.len()) {
            return Err(malformed("correlation buffer (length)"));
        }
        let f = |k| word(k).map(f64::from_le_bytes).unwrap_or(f64::NAN);
        let sum = (0..n_lags).map(|t| f(2 + t)).collect();
        let sum_sq = (0..n_lags).map(|t| f(2 + n_lags + t)).collect();
        let n_segments = header(2 + 2 * n_lags)?;
        Ok(CentroidCorrelation {
            n_lags,
            n_atoms,
            samples: Vec::new(),
            sum,
            sum_sq,
            n_segments,
        })
    }
}
