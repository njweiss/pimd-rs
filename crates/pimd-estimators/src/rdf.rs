//! Radial distribution function (SPEC E-7).

use pimd_core::{Beads, Cell};

use crate::error::EstimatorError;

/// Histogram of minimum-image pair distances (i < j, every bead) on `n_bins` equal bins on the
/// half-open interval [0, r_max) (bohr), accumulated over samples (E-7). Pairs with r ≥ r_max are
/// not counted; a distance below r_max whose bin index rounds to `n_bins` goes into the last bin.
/// Requires a cell and r_max ≤ L_min/2.
#[derive(Debug, Clone)]
pub struct Rdf {
    n_bins: usize,
    r_max: f64,
    counts: Vec<f64>,
    n_atoms: usize,
    cell: Cell,
    n_samples: usize,
}

impl Rdf {
    /// Empty RDF with `n_bins` ≥ 1 bins on [0, `r_max`) (bohr, > 0, ≤ L_min/2) for `n_atoms` atoms
    /// in `cell`.
    ///
    /// Errors with `EstimatorError::InvalidValue` for a bad bin count, a bad `r_max` or
    /// `n_atoms < 2` (Q10).
    pub fn new(
        n_bins: usize,
        r_max: f64,
        cell: Cell,
        n_atoms: usize,
    ) -> Result<Rdf, EstimatorError> {
        if n_bins == 0 {
            return Err(EstimatorError::InvalidValue {
                what: "RDF bin count",
                value: 0.0,
            });
        }
        let l_min = cell.lengths().into_iter().fold(f64::INFINITY, f64::min);
        if !(r_max.is_finite() && r_max > 0.0 && r_max <= 0.5 * l_min) {
            return Err(EstimatorError::InvalidValue {
                what: "RDF r_max (bohr), must be in (0, L_min/2]",
                value: r_max,
            });
        }
        if n_atoms < 2 {
            return Err(EstimatorError::InvalidValue {
                what: "RDF atom count (need N >= 2)",
                value: n_atoms as f64,
            });
        }
        Ok(Rdf {
            n_bins,
            r_max,
            counts: vec![0.0; n_bins],
            n_atoms,
            cell,
            n_samples: 0,
        })
    }

    /// Adds one configuration: all beads of `q` (`[P, N, 3]`, bohr, unwrapped), pairs i < j within
    /// each bead, minimum image (C-2).
    ///
    /// Errors with `EstimatorError::ShapeMismatch` if `q.n_atoms()` differs from `n_atoms` or `q`
    /// has no beads (Q12); the accumulator is unchanged on error.
    pub fn accumulate(&mut self, q: &Beads) -> Result<(), EstimatorError> {
        if q.n_beads() == 0 {
            return Err(EstimatorError::ShapeMismatch {
                what: "bead count of RDF positions (need P >= 1)",
                expected: 1,
                actual: 0,
            });
        }
        if q.n_atoms() != self.n_atoms {
            return Err(EstimatorError::ShapeMismatch {
                what: "atom count of RDF positions",
                expected: self.n_atoms,
                actual: q.n_atoms(),
            });
        }
        let inv_dr = self.n_bins as f64 / self.r_max;
        let r_max2 = self.r_max * self.r_max;
        let mut hist = vec![0usize; self.n_bins];
        for j in 0..q.n_beads() {
            for a in 0..self.n_atoms {
                let ra = q.get(j, a);
                for b in a + 1..self.n_atoms {
                    let rb = q.get(j, b);
                    let d = self
                        .cell
                        .min_image([rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]]);
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 < r_max2 {
                        // r < r_max, so only rounding can give n_bins: clamp into the last bin.
                        let bin = ((r2.sqrt() * inv_dr) as usize).min(self.n_bins - 1);
                        hist[bin] += 1;
                    }
                }
            }
        }
        // C_b / P per sample (E-7), so samples with different P average correctly.
        let pf = q.n_beads() as f64;
        for (c, &h) in self.counts.iter_mut().zip(&hist) {
            *c += h as f64 / pf;
        }
        self.n_samples += 1;
        Ok(())
    }

    /// Rebuilds an RDF from saved state (Q14): the arguments of `new` plus `counts` (as returned by
    /// `counts`, length `n_bins`) and `n_samples`. The result behaves exactly like the RDF that was
    /// saved, including for further `accumulate` calls.
    ///
    /// Errors: as `new` (`EstimatorError::InvalidValue`); `EstimatorError::ShapeMismatch` if
    /// `counts.len()` differs from `n_bins`.
    pub fn from_parts(
        n_bins: usize,
        r_max: f64,
        cell: Cell,
        n_atoms: usize,
        counts: Vec<f64>,
        n_samples: usize,
    ) -> Result<Rdf, EstimatorError> {
        let _ = (n_bins, r_max, cell, n_atoms, counts, n_samples);
        unimplemented!()
    }

    /// Accumulated Σ C_b/P per bin over all samples (E-7), length `n_bins`; saved by checkpoints
    /// together with `n_samples` (Q14).
    pub fn counts(&self) -> &[f64] {
        unimplemented!()
    }

    /// Number of configurations accumulated.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Bin centres (r_b + r_{b+1})/2, bohr, length `n_bins`.
    pub fn bin_centers(&self) -> Vec<f64> {
        let dr = self.r_max / self.n_bins as f64;
        (0..self.n_bins).map(|b| (b as f64 + 0.5) * dr).collect()
    }

    /// g_b = 2 C_b V / (P N (N − 1) V_shell(b)) (E-7), averaged over the accumulated samples; all
    /// zeros before the first sample (Q10).
    pub fn g(&self) -> Vec<f64> {
        if self.n_samples == 0 {
            return vec![0.0; self.n_bins];
        }
        let dr = self.r_max / self.n_bins as f64;
        let n = self.n_atoms as f64;
        let norm = 2.0 * self.cell.volume() / (n * (n - 1.0) * self.n_samples as f64);
        self.counts
            .iter()
            .enumerate()
            .map(|(b, &c)| {
                let (lo, hi) = (b as f64 * dr, (b + 1) as f64 * dr);
                let shell = 4.0 * std::f64::consts::PI / 3.0 * (hi.powi(3) - lo.powi(3));
                norm * c / shell
            })
            .collect()
    }
}
