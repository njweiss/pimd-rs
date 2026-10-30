//! Error type for `pimd-cli`.

use pimd_core::CoreError;
use pimd_estimators::EstimatorError;
use pimd_integrators::IntegratorError;
use pimd_potentials::PotentialError;
use thiserror::Error;

/// Errors returned by config parsing, the run driver and checkpoint I/O.
#[derive(Debug, Error)]
pub enum CliError {
    /// TOML syntax error, missing field, unknown key or a value of the wrong type.
    #[error("config: {0}")]
    Parse(String),
    /// A config value out of range; `what` names the field (e.g. "sampling.n_beads").
    #[error("invalid {what}: {value}")]
    InvalidValue {
        /// The offending field, as `section.key`.
        what: String,
        /// The value that was rejected.
        value: f64,
    },
    /// Corrupt checkpoint file, bad magic or version, or config-hash mismatch.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    /// File-system error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Error from `pimd-core`.
    #[error(transparent)]
    Core(#[from] CoreError),
    /// Error from a potential.
    #[error(transparent)]
    Potential(#[from] PotentialError),
    /// Error from the integrator.
    #[error(transparent)]
    Integrator(#[from] IntegratorError),
    /// Error from an estimator.
    #[error(transparent)]
    Estimator(#[from] EstimatorError),
}
