//! acceptance test: Clock and Rng injectable abstractions

use hippmem_core::rng::Rng;
use hippmem_core::time::Clock;
use std::time::Duration;

// ── Clock tests ──

#[test]
fn system_clock_returns_reasonable_timestamp() {
    let clock = hippmem_core::time::SystemClock;
    let now = clock.now();
    // 2020-01-01 00:00:00 UTC = 1577836800000 ms
    // 2030-01-01 00:00:00 UTC = 1893456000000 ms
    assert!(
        now.as_i64() > 1_577_836_800_000,
        "time should be later than 2020, got {}",
        now.as_i64()
    );
    assert!(
        now.as_i64() < 1_893_456_000_000,
        "time should be earlier than 2030, got {}",
        now.as_i64()
    );
}

#[test]
fn system_clock_is_monotonic() {
    let clock = hippmem_core::time::SystemClock;
    let t1 = clock.now();
    // Brief wait
    std::thread::sleep(Duration::from_millis(5));
    let t2 = clock.now();
    assert!(
        t2.as_i64() >= t1.as_i64(),
        "Clock should be monotonically non-decreasing: t1={}, t2={}",
        t1.as_i64(),
        t2.as_i64()
    );
}

#[test]
fn fixed_clock_returns_fixed_time() {
    let ts = hippmem_core::time::Timestamp::from_millis(1_700_000_000_000);
    let clock = hippmem_core::time::FixedClock::new(ts);
    assert_eq!(clock.now(), ts, "FixedClock should return the set value");
    // Multiple calls stay consistent
    assert_eq!(
        clock.now(),
        ts,
        "FixedClock should be consistent across calls"
    );
}

// ── Rng tests ──

#[test]
fn seeded_rng_same_seed_same_sequence() {
    let seed: u64 = 42;
    let mut rng1 = hippmem_core::rng::SeededRng::new(seed);
    let mut rng2 = hippmem_core::rng::SeededRng::new(seed);

    // Same seed should produce the same sequence
    for _ in 0..100 {
        assert_eq!(
            rng1.gen_u64(),
            rng2.gen_u64(),
            "same-seed sequences should be identical"
        );
    }
}

#[test]
fn seeded_rng_different_seed_different_sequence() {
    let mut rng1 = hippmem_core::rng::SeededRng::new(42);
    let mut rng2 = hippmem_core::rng::SeededRng::new(99);

    let first1 = rng1.gen_u64();
    let first2 = rng2.gen_u64();
    assert_ne!(
        first1, first2,
        "different seeds should produce different sequences"
    );
}

#[test]
fn seeded_rng_gen_u128_produces_values() {
    let mut rng = hippmem_core::rng::SeededRng::new(12345);
    let v1 = rng.gen_u128();
    let v2 = rng.gen_u128();
    assert_ne!(v1, v2, "consecutive u128 values should differ");
}

#[test]
fn seeded_rng_reproducible() {
    let mut rng = hippmem_core::rng::SeededRng::new(0xDEAD_BEEF);
    let sequence: Vec<u64> = (0..10).map(|_| rng.gen_u64()).collect();

    // Verify reproducibility: re-creating an Rng with the same seed yields the same sequence
    let mut rng2 = hippmem_core::rng::SeededRng::new(0xDEAD_BEEF);
    let sequence2: Vec<u64> = (0..10).map(|_| rng2.gen_u64()).collect();

    assert_eq!(
        sequence, sequence2,
        "same-seed sequences must be exactly identical"
    );
}
