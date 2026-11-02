//! In-process MACE backend: a TorchScript export of a MACE model evaluated through `tch`
//! (libtorch), batched over beads, on CPU or CUDA (SPEC P-6, MILESTONE_6 M6b).
//!
//! Everything is behind the feature `torch`; without it this crate is empty, so the default
//! build never needs libtorch (MILESTONE_6 "Build isolation"). Inputs and outputs are f64 in
//! Hartree atomic units (U-1); the model works in Å and eV, converted only here through
//! `pimd_core::units` (U-3); f32 exists only inside the model (D-1).

#[cfg(feature = "torch")]
mod ffi;
#[cfg(feature = "torch")]
mod mace;

#[cfg(feature = "torch")]
pub use mace::{Device, Dtype, MacePotential, atomic_number};
