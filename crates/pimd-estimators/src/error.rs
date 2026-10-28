//! Error type for `pimd-estimators`.

use thiserror::Error;

/// Errors returned by estimators.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EstimatorError {
    /// Inputs disagree in size (bead or atom counts of q, forces, masses, energies).
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
