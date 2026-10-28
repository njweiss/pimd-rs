//! Deterministic random numbers (SPEC D-2).

use rand_chacha::ChaCha8Rng;

/// The single source of randomness (D-2): ChaCha8 seeded from a `u64`.
///
/// Draw order is part of each caller's contract (D-4).
#[derive(Debug, Clone)]
pub struct Rng {
    #[allow(dead_code)] // Phase A stub: used once implemented.
    inner: ChaCha8Rng,
}

impl Rng {
    /// Generator seeded with `ChaCha8Rng::seed_from_u64(seed)` (D-2).
    pub fn from_seed(seed: u64) -> Rng {
        let _ = seed;
        unimplemented!()
    }

    /// One standard normal draw, N(0, 1), from `rand_distr::StandardNormal` (D-2).
    pub fn normal(&mut self) -> f64 {
        unimplemented!()
    }

    /// One uniform draw in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        unimplemented!()
    }
}
