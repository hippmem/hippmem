//! Injectable random source (Rng trait), satisfying the determinism requirement of constitution §4.3.
//!
//! All randomness is obtained through this trait; library logic **MUST NOT** directly use uncontrollable global RNGs.

use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Injectable random source: all randomness inside the library MUST be obtained through this trait.
///
/// See ADR-006 (ULID random part), constitution §4.3 (test reproducibility).
pub trait Rng {
    /// Generates a random u64.
    fn gen_u64(&mut self) -> u64;

    /// Generates a random u128 (for ULID etc.).
    fn gen_u128(&mut self) -> u128;
}

/// Deterministic random source: same seed = same sequence, used for test reproducibility.
///
/// Backed by Xoshiro256++ (SmallRng); seed is u64.
pub struct SeededRng {
    inner: SmallRng,
    seed: u64,
}

impl SeededRng {
    /// Creates a deterministic Rng from a u64 seed. The same seed produces the same sequence.
    pub fn new(seed: u64) -> Self {
        Self {
            inner: SmallRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Returns the seed used at construction.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Rng for SeededRng {
    fn gen_u64(&mut self) -> u64 {
        rand::RngCore::next_u64(&mut self.inner)
    }

    fn gen_u128(&mut self) -> u128 {
        let hi = self.gen_u64() as u128;
        let lo = self.gen_u64() as u128;
        (hi << 64) | lo
    }
}
