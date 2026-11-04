//! Error type for `pimd-integrators`.

use pimd_core::CoreError;
use pimd_potentials::PotentialError;
use thiserror::Error;

/// Errors returned by integrator constructors and `step`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum IntegratorError {
    /// A scalar argument was out of range or non-finite (dt, β, τ₀, λ).
    #[error("invalid {what}: {value}")]
    InvalidValue {
        /// Name of the offending argument.
        what: &'static str,
        /// The value that was rejected.
        value: f64,
    },
    /// Two parts of the input disagree in size (bead or atom counts of q, p, masses, integrator).
    #[error("shape mismatch in {what}: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Which quantity has the wrong size.
        what: &'static str,
        /// Required size.
        expected: usize,
        /// Size supplied.
        actual: usize,
    },
    /// Error from `pimd-core` (e.g. building `NormalModes`).
    #[error(transparent)]
    Core(#[from] CoreError),
    /// Error from `Potential::evaluate`.
    #[error(transparent)]
    Potential(#[from] PotentialError),
}
