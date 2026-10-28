//! Ring-polymer bead storage (SPEC S-1).

use crate::error::CoreError;

/// Per-bead, per-atom 3-vectors with shape `[P, N, 3]`, bead-major and row-major:
/// element (j, i, d) is at index `(j*N + i)*3 + d` (S-1).
///
/// Used for positions (bohr), momenta and forces (Eh/bohr), and normal-mode coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Beads {
    n_beads: usize,
    n_atoms: usize,
    data: Vec<f64>,
}

impl Beads {
    /// All-zero storage for `n_beads` beads (P) of `n_atoms` atoms (N).
    pub fn zeros(n_beads: usize, n_atoms: usize) -> Beads {
        Beads {
            n_beads,
            n_atoms,
            data: vec![0.0; n_beads * n_atoms * 3],
        }
    }

    /// Wraps a flat buffer laid out as in S-1. Errors unless `data.len() == n_beads * n_atoms * 3`.
    pub fn from_vec(n_beads: usize, n_atoms: usize, data: Vec<f64>) -> Result<Beads, CoreError> {
        let expected = n_beads * n_atoms * 3;
        if data.len() != expected {
            return Err(CoreError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Beads {
            n_beads,
            n_atoms,
            data,
        })
    }

    /// Number of beads P.
    pub fn n_beads(&self) -> usize {
        self.n_beads
    }

    /// Number of atoms N.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// The 3-vector of bead `j`, atom `i`. Panics if out of range.
    pub fn get(&self, j: usize, i: usize) -> [f64; 3] {
        let k = self.offset(j, i);
        [self.data[k], self.data[k + 1], self.data[k + 2]]
    }

    /// Sets the 3-vector of bead `j`, atom `i`. Panics if out of range.
    pub fn set(&mut self, j: usize, i: usize, v: [f64; 3]) {
        let k = self.offset(j, i);
        self.data[k..k + 3].copy_from_slice(&v);
    }

    /// The flat buffer, length `P * N * 3`, in S-1 order.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The flat buffer, mutable, length `P * N * 3`, in S-1 order.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Centroid of each atom, the mean over beads: one 3-vector per atom, length N.
    pub fn centroid(&self) -> Vec<[f64; 3]> {
        let p = self.n_beads as f64;
        (0..self.n_atoms)
            .map(|i| {
                let mut sum = [0.0; 3];
                for j in 0..self.n_beads {
                    let q = self.get(j, i);
                    for d in 0..3 {
                        sum[d] += q[d];
                    }
                }
                sum.map(|s| s / p)
            })
            .collect()
    }

    fn offset(&self, j: usize, i: usize) -> usize {
        assert!(
            j < self.n_beads && i < self.n_atoms,
            "bead index ({j}, {i}) out of range for shape [{}, {}, 3]",
            self.n_beads,
            self.n_atoms
        );
        (j * self.n_atoms + i) * 3
    }
}
