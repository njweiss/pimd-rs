//! Error type for the fallible estimators (RDF E-7, pressure E-8).

use thiserror::Error;

/// Errors returned by `pimd-estimators` functions that validate their input.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EstimatorError {
    /// A scalar argument was out of range or non-finite (e.g. `r_max`, bin count).
    #[error("invalid {what}: {value}")]
    InvalidValue {
        /// Name of the offending argument.
        what: &'static str,
        /// The value that was rejected.
        value: f64,
    },
    /// Inputs disagree in size (bead or atom counts of positions, momenta, masses).
    #[error("shape mismatch in {what}: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Which quantity has the wrong size.
        what: &'static str,
        /// Required size.
        expected: usize,
        /// Size supplied.
        actual: usize,
    },
}
