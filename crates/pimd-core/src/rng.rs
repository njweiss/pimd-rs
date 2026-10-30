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
    /// ChaCha8 key (the seed expanded by `seed_from_u64`).
    seed: [u8; 32],
    /// ChaCha stream number.
    stream: u64,
    /// Position in the stream, in 32-bit words (< 2^68).
    word_pos: u128,
}

/// Encoded length: 32-byte key, 8-byte stream, 16-byte word position, little-endian.
const STATE_BYTES: usize = 32 + 8 + 16;

impl Rng {
    /// The current generator position. Does not advance the generator.
    pub fn snapshot(&self) -> RngState {
        RngState {
            seed: self.inner.get_seed(),
            stream: self.inner.get_stream(),
            word_pos: self.inner.get_word_pos(),
        }
    }

    /// A generator that continues exactly where the one that produced `state` was (D-3).
    pub fn restore(state: &RngState) -> Rng {
        let mut inner = ChaCha8Rng::from_seed(state.seed);
        inner.set_stream(state.stream);
        inner.set_word_pos(state.word_pos);
        Rng { inner }
    }
}

impl RngState {
    /// Fixed-length little-endian encoding of the state.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(STATE_BYTES);
        out.extend_from_slice(&self.seed);
        out.extend_from_slice(&self.stream.to_le_bytes());
        out.extend_from_slice(&self.word_pos.to_le_bytes());
        out
    }

    /// Inverse of `to_bytes`.
    ///
    /// Errors: `CoreError::ShapeMismatch` unless `bytes` has the fixed encoded length;
    /// `CoreError::InvalidValue` if the word position is outside ChaCha's 68-bit range.
    pub fn from_bytes(bytes: &[u8]) -> Result<RngState, CoreError> {
        if bytes.len() != STATE_BYTES {
            return Err(CoreError::ShapeMismatch {
                expected: STATE_BYTES,
                actual: bytes.len(),
            });
        }
        let mut seed = [0u8; 32];
        let mut stream = [0u8; 8];
        let mut word_pos = [0u8; 16];
        seed.copy_from_slice(&bytes[..32]);
        stream.copy_from_slice(&bytes[32..40]);
        word_pos.copy_from_slice(&bytes[40..]);
        let state = RngState {
            seed,
            stream: u64::from_le_bytes(stream),
            word_pos: u128::from_le_bytes(word_pos),
        };
        if state.word_pos >> 68 != 0 {
            return Err(CoreError::InvalidValue {
                what: "RNG word position",
                value: state.word_pos as f64,
            });
        }
        Ok(state)
    }
}
