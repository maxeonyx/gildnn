use rand::{rngs::StdRng, SeedableRng};

/// Construct a deterministic RNG from a fixed seed.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
