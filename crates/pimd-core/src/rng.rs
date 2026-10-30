//! Deterministic random numbers (SPEC D-2).

use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::error::CoreError;

/// The single source of randomness (D-2): ChaCha8 seeded from a `u64`.
///
/// Draw order is part of each caller's contract (D-4).
#[derive(Debug)]
pub struct Rng {
    inner: ChaCha8Rng,
}

impl Rng {
    /// Generator seeded with `ChaCha8Rng::seed_from_u64(seed)` (D-2).
    pub fn from_seed(seed: u64) -> Rng {
        Rng {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// One standard normal draw, N(0, 1), from `rand_distr::StandardNormal` (D-2).
    pub fn normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.inner)
    }

    /// One uniform draw in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        StandardUniform.sample(&mut self.inner)
    }
}

/// A saved generator position (D-2, D-3): restoring it continues the exact same sequence.
///
/// Opaque; serialisable with `to_bytes` / `from_bytes` for checkpoints.
#[derive(Clone, PartialEq, Debug)]
pub struct RngState {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    bytes: Vec<u8>,
}

impl Rng {
    /// The current generator position. Does not advance the generator.
    pub fn snapshot(&self) -> RngState {
        let _ = &self.inner;
        unimplemented!()
    }

    /// A generator that continues exactly where the one that produced `state` was (D-3).
    pub fn restore(state: &RngState) -> Rng {
        let _ = state;
        unimplemented!()
    }
}

impl RngState {
    /// Fixed-length little-endian encoding of the state.
    pub fn to_bytes(&self) -> Vec<u8> {
        let _ = self;
        unimplemented!()
    }

    /// Inverse of `to_bytes`. Errors (a `CoreError`) on bytes that are not a valid encoding.
    pub fn from_bytes(bytes: &[u8]) -> Result<RngState, CoreError> {
        let _ = bytes;
        unimplemented!()
    }
}
