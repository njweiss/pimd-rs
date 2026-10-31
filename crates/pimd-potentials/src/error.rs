//! Error type for `pimd-potentials`.

use thiserror::Error;

/// Errors returned by potential constructors and `Potential::evaluate`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PotentialError {
    /// The request is valid but this potential does not support it, e.g. `subset = Some` (P-4).
    #[error("operation not supported by this potential")]
    Unsupported,
    /// A parameter, or a parameter combined with the cell, was out of range or non-finite (mass,
    /// frequency, ..., or `cutoff + skin > L_min/2` in `evaluate`, C-4, Q7).
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
    /// An external backend failed: I/O error, protocol violation, wrong atom count, non-finite
    /// value, closed connection or timeout (K-5). The message names the failing bead.
    #[error("backend: {0}")]
    Backend(String),
}
