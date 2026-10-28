//! Orthorhombic simulation cell (SPEC §4).

/// Orthorhombic cell with three positive, finite edge lengths in bohr (C-1).
///
/// Passed as `Option<&Cell>`; `None` means open boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    lengths: [f64; 3],
}

impl Cell {
    /// Edge lengths `[Lx, Ly, Lz]`, in bohr.
    pub fn lengths(&self) -> [f64; 3] {
        self.lengths
    }

    /// Cell volume Lx·Ly·Lz, in bohr³.
    pub fn volume(&self) -> f64 {
        self.lengths[0] * self.lengths[1] * self.lengths[2]
    }
}
