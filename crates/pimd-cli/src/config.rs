//! Run configuration (MILESTONE_4 "Config format"): TOML in user units (Å, K, amu, fs), stored
//! in Hartree atomic units (U-1). The conversion happens here only, through `pimd_core::units`
//! (U-3).

use pimd_core::{Cell, Transform};
use pimd_potentials::LjParams;

use crate::error::CliError;

/// A validated run configuration in atomic units. Built by `Config::from_toml_str`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `[system]`.
    pub system: SystemConfig,
    /// `[potential]` (`kind = "lj"`): ε in Eh, σ, cutoff and skin in bohr, shift flag (P-3).
    pub potential: LjParams,
    /// `[sampling]`.
    pub sampling: SamplingConfig,
    /// `[output]`.
    pub output: OutputConfig,
}

/// `[system]`: one species in an orthorhombic periodic cell.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemConfig {
    /// Number of atoms N.
    pub n_atoms: usize,
    /// Mass of every atom, m_e (from `mass_amu`).
    pub mass: f64,
    /// Periodic cell, bohr (from `box_angstrom`, one value for a cube or three).
    pub cell: Cell,
}

/// `[sampling]`: PIMD with BAOAB + PILE-L (I-1, T-1).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    /// Temperature T, K, as given.
    pub temperature_kelvin: f64,
    /// β = 1/(k_B T), Eh⁻¹ (U-4).
    pub beta: f64,
    /// Number of beads P ≥ 1.
    pub n_beads: usize,
    /// Time step, ħ/Eh (from `dt_fs`).
    pub dt: f64,
    /// Centroid thermostat time constant τ_0, ħ/Eh (from `tau0_fs`, T-2).
    pub tau0: f64,
    /// Equilibration steps (nothing recorded).
    pub n_equil: usize,
    /// Production steps; step s = 1 … n_steps is recorded when s % stride == 0.
    pub n_steps: usize,
    /// Recording stride ≥ 1, in steps.
    pub stride: usize,
    /// Seed of the single `pimd_core::Rng` (D-2).
    pub seed: u64,
    /// Normal-mode transform algorithm (`normal_modes`, default "matrix", RP-4).
    pub normal_modes: Transform,
}

/// `[output]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputConfig {
    /// RDF settings when `rdf_bins` and `rdf_rmax_angstrom` are both given, else `None`.
    pub rdf: Option<RdfConfig>,
    /// Write `checkpoint.bin` every this many production steps (and always at the end).
    pub checkpoint_every: usize,
}

/// RDF histogram settings (E-7).
#[derive(Debug, Clone, PartialEq)]
pub struct RdfConfig {
    /// Number of bins.
    pub n_bins: usize,
    /// Upper edge r_max, bohr (from `rdf_rmax_angstrom`), ≤ L_min/2.
    pub r_max: f64,
}

impl Config {
    /// Parses and validates a TOML config, converting user units to atomic units (U-3).
    ///
    /// Errors: `CliError::Parse` for TOML syntax errors, missing fields, unknown keys and
    /// wrongly typed values; `CliError::InvalidValue` (naming the field) for out-of-range values,
    /// including `n_beads = 0` and `cutoff + skin > L_min/2` (C-4).
    pub fn from_toml_str(s: &str) -> Result<Config, CliError> {
        let _ = s;
        unimplemented!()
    }
}
