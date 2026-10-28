//! Deterministic random numbers (SPEC D-2).

use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

/// The single source of randomness (D-2): ChaCha8 seeded from a `u64`.
///
/// Draw order is part of each caller's contract (D-4).
#[derive(Debug, Clone)]
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
