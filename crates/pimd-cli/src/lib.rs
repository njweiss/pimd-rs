//! Run driver for pimd-rs: TOML config in, trajectory observables, RDF and checkpoints out
//! (MILESTONE_4).
//!
//! Units are converted only here, at the config and output boundary, through `pimd_core::units`
//! (U-3); everything inside is Hartree atomic units (U-1).

pub mod config;
pub mod error;
pub mod run;

pub use config::{Config, OutputConfig, RdfConfig, SamplingConfig, SystemConfig};
pub use error::CliError;
pub use run::{RunSummary, run, run_from_checkpoint};
