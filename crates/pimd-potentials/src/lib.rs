//! Potentials behind the model-agnostic `Potential` trait (SPEC §9).

pub mod double_well;
pub mod error;
pub mod harmonic;
pub mod potential;

pub use double_well::DoubleWell;
pub use error::PotentialError;
pub use harmonic::Harmonic;
pub use potential::{Evaluation, Potential};
