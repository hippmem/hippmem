//! acceptance test: multi-channel seed recall

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::RecallChannel;
use hippmem_retrieval::seeds::multi_channel_seeds;

/// Each channel produces seeds and scores them independently.
#[test]
fn multiple_channels_produce_seeds() {
    // V9: entity channel uses the passed-in normalized score directly
    let entity = vec![(MemoryId(1), 0.8f32), (MemoryId(2), 0.6f32)];
    let temporal = vec![(MemoryId(3), true), (MemoryId(4), false)];
    let semantic = vec![(MemoryId(5), 0.9), (MemoryId(6), 0.7)];

    let result = multi_channel_seeds(
        "rust database",
        &entity,
        &temporal,
        &semantic,
        &[], // topic
        &[], // bm25
        &[], // binary
        &[], // goal
        &[], // event
        &[], // causal
        &[], // recent
        20,
    );
    assert!(!result.seeds.is_empty(), "should have seeds");

    // Each hit channel should have channel_scores
    let has_bm25 = result
        .channel_scores
        .iter()
        .any(|(c, _)| *c == RecallChannel::Bm25);
    let has_ent = result
        .channel_scores
        .iter()
        .any(|(c, _)| *c == RecallChannel::EntityInverted);
    let has_tmp = result
        .channel_scores
        .iter()
        .any(|(c, _)| *c == RecallChannel::Temporal);
    let has_sem = result
        .channel_scores
        .iter()
        .any(|(c, _)| *c == RecallChannel::SemanticDense);
    assert!(
        has_bm25 || has_ent || has_tmp || has_sem,
        "at least one channel should contribute"
    );
}

/// V2-020: SemanticBinary channel produces seeds from binary_hits.
#[test]
fn semantic_binary_channel_from_binary_hits() {
    let binary = vec![(MemoryId(10), 0.85), (MemoryId(11), 0.72)];
    let result = multi_channel_seeds(
        "query",
        &[],
        &[],
        &[],
        &[],
        &[],
        &binary,
        &[],
        &[],
        &[],
        &[],
        20,
    );
    assert!(!result.seeds.is_empty(), "binary hits should produce seeds");

    // Verify the seed channel is marked as SemanticBinary
    let has_binary = result
        .seeds
        .iter()
        .any(|s| s.channel == RecallChannel::SemanticBinary);
    assert!(has_binary, "binary_hits maps to the SemanticBinary channel");

    // Verify channel_scores contains SemanticBinary
    let has_binary_cs = result
        .channel_scores
        .iter()
        .any(|(c, _)| *c == RecallChannel::SemanticBinary);
    assert!(
        has_binary_cs,
        "channel_scores should contain SemanticBinary"
    );

    // Scores are in [0,1] range
    for s in &result.seeds {
        assert!(
            (0.0..=1.0).contains(&s.score),
            "score {} should be in [0,1]",
            s.score
        );
    }
}

/// Scores are in [0, 1] range.
#[test]
fn seed_scores_in_range() {
    // V9: entity channel uses the passed-in normalized score directly; take a value in [0,1] to verify the range constraint
    let entity = vec![(MemoryId(1), 0.5f32)];
    let result = multi_channel_seeds(
        "q",
        &entity,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        10,
    );
    for s in &result.seeds {
        assert!(
            (0.0..=1.0).contains(&s.score),
            "score {} should be in [0,1]",
            s.score
        );
    }
}
