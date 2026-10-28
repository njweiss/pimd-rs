//! Orthorhombic simulation cell (SPEC §4).

use crate::error::CoreError;

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
        let _ = lengths;
        unimplemented!()
    }

    /// Minimum-image displacement (C-2): component-wise, each result component in [−L/2, L/2), bohr.
    pub fn min_image(&self, d: [f64; 3]) -> [f64; 3] {
        let _ = d;
        unimplemented!()
    }

    /// Position wrapped into the cell (C-3): each result component in [0, L), bohr.
    pub fn wrap(&self, r: [f64; 3]) -> [f64; 3] {
        let _ = r;
        unimplemented!()
    }

    /// Edge lengths `[Lx, Ly, Lz]`, in bohr.
    pub fn lengths(&self) -> [f64; 3] {
        self.lengths
    }

    /// Cell volume Lx·Ly·Lz, in bohr³.
    pub fn volume(&self) -> f64 {
        self.lengths[0] * self.lengths[1] * self.lengths[2]
    }
}
