//! Error type for `pimd-core`.

use thiserror::Error;

/// Errors returned by `pimd-core` constructors and conversions.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CoreError {
    /// A scalar argument was out of range or non-finite (temperature, β, cell length).
    #[error("invalid {what}: {value}")]
    InvalidValue {
        /// Name of the offending argument.
        what: &'static str,
        /// The value that was rejected.
        value: f64,
    },
    /// A ring polymer needs at least one bead (RP-1).
    #[error("invalid number of beads: {0} (need P >= 1)")]
    InvalidBeadCount(usize),
    /// A flat buffer did not have length `P * N * 3` (S-1).
    #[error("shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch {
        /// Required length, `P * N * 3`.
        expected: usize,
        /// Length supplied.
        actual: usize,
    },
}

/// Returns `v` if it is positive and finite, else `CoreError::InvalidValue` named `what`.
pub(crate) fn positive_finite(what: &'static str, v: f64) -> Result<f64, CoreError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(CoreError::InvalidValue { what, value: v })
    }
}
