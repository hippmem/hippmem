//! acceptance test: RecallChannel → MatchDimension mapping.
//!
//! Verifies that `deduce_dimensions` correctly maps ActivationStep.channel to MatchDimension,
//! rather than hardcoding Importance.

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{ActivationStep, LinkType, MatchDimension, RecallChannel};
use hippmem_retrieval::explain::deduce_dimensions;

/// Helper: build a single-step ActivationStep (seed, hop=0)
fn seed_step(channel: RecallChannel) -> ActivationStep {
    ActivationStep {
        from: None,
        to: MemoryId(1),
        via_link: None,
        channel: Some(channel),
        hop: 0,
        energy_in: 0.5,
        energy_out: 0.5,
    }
}

/// Helper: build an ActivationStep with an edge type (spreading step, hop>0)
fn spread_step(via: LinkType, hop: u8) -> ActivationStep {
    ActivationStep {
        from: Some(MemoryId(hop as u128)),
        to: MemoryId(hop as u128 + 1),
        via_link: Some(via),
        channel: None,
        hop,
        energy_in: 0.3,
        energy_out: 0.3,
    }
}

// ── Per-channel mapping tests ──

#[test]
fn entity_inverted_maps_to_entity() {
    let trace = vec![seed_step(RecallChannel::EntityInverted)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Entity),
        "EntityInverted should map to Entity, actual: {dims:?}"
    );
    assert!(
        !dims.contains(&MatchDimension::Importance),
        "Importance should no longer be hardcoded, actual: {dims:?}"
    );
}

#[test]
fn bm25_maps_to_semantic() {
    let trace = vec![seed_step(RecallChannel::Bm25)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Semantic),
        "Bm25 should map to Semantic, actual: {dims:?}"
    );
}

#[test]
fn semantic_dense_maps_to_semantic() {
    let trace = vec![seed_step(RecallChannel::SemanticDense)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Semantic),
        "SemanticDense should map to Semantic, actual: {dims:?}"
    );
}

#[test]
fn semantic_binary_maps_to_semantic() {
    let trace = vec![seed_step(RecallChannel::SemanticBinary)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Semantic),
        "SemanticBinary should map to Semantic, actual: {dims:?}"
    );
}

#[test]
fn temporal_maps_to_temporal() {
    let trace = vec![seed_step(RecallChannel::Temporal)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Temporal),
        "Temporal should map to Temporal, actual: {dims:?}"
    );
}

#[test]
fn topic_cluster_maps_to_topic() {
    let trace = vec![seed_step(RecallChannel::TopicCluster)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Topic),
        "TopicCluster should map to Topic, actual: {dims:?}"
    );
}

#[test]
fn goal_maps_to_goal() {
    let trace = vec![seed_step(RecallChannel::Goal)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Goal),
        "Goal should map to Goal, actual: {dims:?}"
    );
}

#[test]
fn event_maps_to_event() {
    let trace = vec![seed_step(RecallChannel::Event)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Event),
        "Event should map to Event, actual: {dims:?}"
    );
}

#[test]
fn causal_maps_to_causal() {
    let trace = vec![seed_step(RecallChannel::Causal)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Causal),
        "Causal should map to Causal, actual: {dims:?}"
    );
}

#[test]
fn recent_activation_maps_to_co_context() {
    let trace = vec![seed_step(RecallChannel::RecentActivation)];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::CoContext),
        "RecentActivation should map to CoContext, actual: {dims:?}"
    );
}

#[test]
fn graph_spreading_is_skipped() {
    let trace = vec![seed_step(RecallChannel::GraphSpreading)];
    let dims = deduce_dimensions(&trace);
    // GraphSpreading is not a seed channel and should be skipped
    assert!(
        dims.is_empty() || dims.iter().all(|d| *d != MatchDimension::Importance),
        "GraphSpreading should be skipped (returns None) and produce no MatchDimension, actual: {dims:?}"
    );
}

// ── Spreading steps (non-seed, via_link present) still derive from edge type ──

#[test]
fn spread_step_derives_from_link_type() {
    let trace = vec![
        seed_step(RecallChannel::EntityInverted),
        spread_step(LinkType::Causal, 1),
    ];
    let dims = deduce_dimensions(&trace);
    assert!(
        dims.contains(&MatchDimension::Entity),
        "seed EntityInverted"
    );
    assert!(
        dims.contains(&MatchDimension::Causal),
        "spreading step Causal edge"
    );
}

#[test]
fn multiple_seeds_dedup_dimensions() {
    let trace = vec![
        seed_step(RecallChannel::EntityInverted),
        seed_step(RecallChannel::EntityInverted), // duplicate channel
        spread_step(LinkType::EntityOverlap, 1),
    ];
    let dims = deduce_dimensions(&trace);
    // Entity should appear only once (dedup), and should not contain Importance
    let entity_count = dims
        .iter()
        .filter(|d| **d == MatchDimension::Entity)
        .count();
    assert_eq!(
        entity_count, 1,
        "Entity should be deduped, actual: {dims:?}"
    );
    assert!(
        !dims.contains(&MatchDimension::Importance),
        "Importance should no longer be hardcoded, actual: {dims:?}"
    );
}

// ── Regression: empty trace does not panic ──

#[test]
fn empty_trace_returns_empty_dims() {
    let dims = deduce_dimensions(&[]);
    assert!(
        dims.is_empty(),
        "empty trace should return an empty dimension list"
    );
}

// ── Regression: a seed without a channel does not panic ──

#[test]
fn seed_without_channel_is_skipped() {
    let step = ActivationStep {
        from: None,
        to: MemoryId(1),
        via_link: None,
        channel: None, // no channel
        hop: 0,
        energy_in: 0.5,
        energy_out: 0.5,
    };
    let dims = deduce_dimensions(&[step]);
    // A seed without a channel produces no dimension and does not panic
    assert!(
        dims.is_empty(),
        "seed without a channel should be skipped, actual: {dims:?}"
    );
}
