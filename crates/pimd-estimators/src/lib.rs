//! Ring-polymer estimators (SPEC §8).
//!
//! The energy estimators return plain `f64` and panic on mismatched input shapes, a programming
//! error (Q4); the RDF and pressure estimators return `Result<_, EstimatorError>`.

pub mod energy;
pub mod error;
pub mod pressure;
pub mod rdf;

pub use energy::{
    hamiltonian, kinetic_primitive, kinetic_virial, potential_mean, spring_energy, total_primitive,
    total_virial,
};
pub use error::EstimatorError;
pub use pressure::{kinetic_classical, pressure_classical};
pub use rdf::Rdf;
