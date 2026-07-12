//! acceptance test: core newtypes and UnitScore

use hippmem_core::ids::MemoryId;
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use proptest::prelude::*;

// ── UnitScore property tests ──

proptest! {
    /// UnitScore::new(x) for any f32 lands in [0, 1]
    #[test]
    fn unit_score_bounded(x in any::<f32>()) {
        let score = UnitScore::new(x);
        let v = score.value();
        // NaN input should be clamped to 0.0
        if x.is_nan() {
            assert!((0.0..=1.0).contains(&v), "NaN input should clamp to [0,1], got {}", v);
        } else {
            assert!((0.0..=1.0).contains(&v), "{} exceeds [0,1], got {}", x, v);
        }
    }

    /// UnitScore::new(x) preserves values already in [0,1] (no extra clamping)
    #[test]
    fn unit_score_preserves_valid(x in 0.0f32..=1.0f32) {
        let score = UnitScore::new(x);
        let v = score.value();
        assert!((v - x).abs() < f32::EPSILON, "valid value {} was changed to {}", x, v);
    }
}

// ── Serialization round-trip tests ──

#[test]
fn memory_id_roundtrip_bincode() {
    let id = MemoryId::generate();
    let encoded = bincode::serde::encode_to_vec(id, bincode::config::standard())
        .expect("MemoryId encoding failed");
    let decoded: MemoryId =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
            .expect("MemoryId decoding failed")
            .0;
    assert_eq!(id, decoded, "MemoryId bincode round-trip not equal");
}

#[test]
fn memory_id_roundtrip_json() {
    let id = MemoryId::generate();
    let json = serde_json::to_string(&id).expect("MemoryId JSON serialization failed");
    let decoded: MemoryId =
        serde_json::from_str(&json).expect("MemoryId JSON deserialization failed");
    assert_eq!(id, decoded, "MemoryId JSON round-trip not equal");
}

#[test]
fn timestamp_roundtrip_bincode() {
    let ts = Timestamp(1_700_000_000_000); // fixed value for determinism
    let encoded = bincode::serde::encode_to_vec(ts, bincode::config::standard())
        .expect("Timestamp encoding failed");
    let decoded: Timestamp =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
            .expect("Timestamp decoding failed")
            .0;
    assert_eq!(ts, decoded, "Timestamp bincode round-trip not equal");
}

#[test]
fn timestamp_roundtrip_json() {
    let ts = Timestamp(1_700_000_000_000);
    let json = serde_json::to_string(&ts).expect("Timestamp JSON serialization failed");
    let decoded: Timestamp =
        serde_json::from_str(&json).expect("Timestamp JSON deserialization failed");
    assert_eq!(ts, decoded, "Timestamp JSON round-trip not equal");
}

#[test]
fn unit_score_roundtrip_bincode() {
    let score = UnitScore::new(0.75);
    let encoded = bincode::serde::encode_to_vec(score, bincode::config::standard())
        .expect("UnitScore encoding failed");
    let decoded: UnitScore =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
            .expect("UnitScore decoding failed")
            .0;
    // f32 comparison uses epsilon
    assert!(
        (decoded.value() - score.value()).abs() < f32::EPSILON,
        "UnitScore bincode round-trip not equal: {} vs {}",
        score.value(),
        decoded.value()
    );
}

// ── Basic construction and access ──

#[test]
fn memory_id_construct_and_access() {
    let id = MemoryId(42);
    assert_eq!(id.0, 42);
    assert_eq!(id.as_u128(), 42);
}

#[test]
fn timestamp_construct_and_access() {
    let ts = Timestamp(1_700_000_000_000);
    assert_eq!(ts.0, 1_700_000_000_000);
    assert_eq!(ts.as_i64(), 1_700_000_000_000);
}

#[test]
fn unit_score_new_clamps_out_of_range() {
    assert!((UnitScore::new(-0.5).value() - 0.0).abs() < f32::EPSILON);
    assert!((UnitScore::new(1.5).value() - 1.0).abs() < f32::EPSILON);
    assert!((UnitScore::new(f32::NAN).value() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn unit_score_new_preserves_in_range() {
    assert!((UnitScore::new(0.0).value() - 0.0).abs() < f32::EPSILON);
    assert!((UnitScore::new(0.5).value() - 0.5).abs() < f32::EPSILON);
    assert!((UnitScore::new(1.0).value() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn key_type_aliases_exist() {
    // Verify key type aliases are usable (compile-time verification is enough; also verify constructibility)
    let _entity: hippmem_core::ids::EntityKey = 1u64;
    let _topic: hippmem_core::ids::TopicKey = 2u64;
    let _goal: hippmem_core::ids::GoalKey = 3u64;
    let _event: hippmem_core::ids::EventKey = 4u64;
    let _causal: hippmem_core::ids::CausalKey = 5u64;
    let _emotion: hippmem_core::ids::EmotionKey = 1u8;
    let _temporal: hippmem_core::ids::TemporalKey = 42u32;
}
