//! Orthorhombic simulation cell (SPEC §4).

use crate::error::{CoreError, positive_finite};

/// Orthorhombic cell with three positive, finite edge lengths in bohr (C-1).
///
/// Passed as `Option<&Cell>`; `None` means open boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    lengths: [f64; 3],
}

impl Cell {
    /// Cell with edge lengths `[Lx, Ly, Lz]` in bohr (C-1).
    ///
    /// Errors with `CoreError::InvalidValue` unless every length is positive and finite.
    pub fn new(lengths: [f64; 3]) -> Result<Cell, CoreError> {
        for &l in &lengths {
            positive_finite("cell length (bohr)", l)?;
        }
        Ok(Cell { lengths })
    }

    /// Minimum-image displacement (C-2): component-wise, each result component in [−L/2, L/2), bohr.
    pub fn min_image(&self, d: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for k in 0..3 {
            let l = self.lengths[k];
            let half = 0.5 * l;
            let mut x = d[k] - l * (d[k] / l).round();
            // Needed even with exact arithmetic: `round` sends ties away from zero, so d = −L/2
            // gives +L/2. Rounding error can also leave x just outside the half-open interval.
            if x >= half {
                x -= l;
            } else if x < -half {
                x += l;
            }
            out[k] = x;
        }
        out
    }

    /// Position wrapped into the cell (C-3): each result component in [0, L), bohr.
    pub fn wrap(&self, r: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for k in 0..3 {
            let l = self.lengths[k];
            let x = r[k].rem_euclid(l);
            // rem_euclid can round up to exactly L for tiny negative inputs.
            out[k] = if x >= l { 0.0 } else { x };
        }
        out
    }

    /// Edge lengths `[Lx, Ly, Lz]`, in bohr.
    pub fn lengths(&self) -> [f64; 3] {
        self.lengths
    }

    /// The shortest edge length min(Lx, Ly, Lz), in bohr (L_min of C-4 and E-7).
    pub fn min_length(&self) -> f64 {
        unimplemented!()
    }

    /// Cell volume Lx·Ly·Lz, in bohr³.
    pub fn volume(&self) -> f64 {
        self.lengths[0] * self.lengths[1] * self.lengths[2]
    }
}
