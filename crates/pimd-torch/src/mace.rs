//! `MacePotential` (SPEC P-6): a TorchScript MACE export (`scripts/export_mace.py`) evaluated
//! in-process, all P beads as one batched graph.

use std::path::Path;

use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, Potential, PotentialError};

/// Where the model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// libtorch on the CPU, with the intra-op thread count given to `MacePotential::load`.
    Cpu,
    /// CUDA device with this index.
    Cuda(usize),
}

/// Precision inside the model (D-1: f32 only inside the adapter; inputs and outputs are f64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// float32 inside the model; positions are cast on the way in, results back to f64.
    F32,
    /// float64 throughout (the export's own dtype).
    F64,
}

/// Atomic number of an element symbol ("H" -> 1, ...), H through I; `None` if unknown.
pub fn atomic_number(symbol: &str) -> Option<u8> {
    let _ = symbol;
    unimplemented!("M6b: atomic_number")
}

/// A MACE model loaded from a TorchScript export, for a fixed list of N atoms (P-6).
pub struct MacePotential {
    _private: (),
}

impl MacePotential {
    /// Loads a TorchScript export (export_mace.py) for N = `atomic_numbers.len()` atoms, sets the
    /// libtorch intra-op thread count to `threads`, disables the TorchScript profiling executor,
    /// and sets libtorch's deterministic-algorithms switch to `deterministic` (process-wide).
    ///
    /// Errors: missing or invalid file, or a module that is not an export -> `Backend`;
    /// an atomic number the model does not know -> `InvalidParameter { what: "atomic number" }`;
    /// `threads == 0` -> `InvalidParameter { what: "threads" }`; CUDA unavailable -> `Backend`.
    pub fn load(
        path: &Path,
        atomic_numbers: &[u8],
        device: Device,
        dtype: Dtype,
        threads: usize,
        deterministic: bool,
    ) -> Result<MacePotential, PotentialError> {
        let _ = (path, atomic_numbers, device, dtype, threads, deterministic);
        unimplemented!("M6b: MacePotential::load")
    }

    /// The model cutoff r_max in bohr.
    pub fn cutoff(&self) -> f64 {
        unimplemented!("M6b: MacePotential::cutoff")
    }
}

impl Potential for MacePotential {
    /// P-6: energies (Eh, per bead) and forces (Eh/bohr, `[P, N, 3]`) of all beads of
    /// `positions` (`[P, N, 3]`, bohr) as one batched graph; `virial` is `None`.
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        let _ = (positions, cell, subset);
        unimplemented!("M6b: MacePotential::evaluate")
    }
}
