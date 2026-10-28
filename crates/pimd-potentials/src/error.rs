//! Error type for `pimd-potentials`.

use thiserror::Error;

/// Errors returned by potential constructors and `Potential::evaluate`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PotentialError {
    /// The request is valid but this potential does not support it, e.g. `subset = Some` (P-4).
    #[error("operation not supported by this potential")]
    Unsupported,
    /// A constructor parameter was out of range or non-finite (mass, frequency, ...).
    #[error("invalid {what}: {value}")]
    InvalidParameter {
        /// Name of the offending parameter.
        what: &'static str,
        /// The value that was rejected.
        value: f64,
    },
    /// `positions.n_atoms()` differs from the number of atoms the potential was built for (P-1).
    #[error("shape mismatch: potential has {expected} atoms, positions have {actual}")]
    ShapeMismatch {
        /// Number of atoms the potential was built for.
        expected: usize,
        /// `positions.n_atoms()`.
        actual: usize,
    },
}
