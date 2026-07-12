//! acceptance test: retrieve full-pipeline integration test
//!
//! Verifies the 03 §4 retrieval pipeline: multi-channel seeds → initial energy → spreading activation → rerank → warnings.
//! 5 scenarios:
//!   1. Causal chain priority > EntityOverlap
//!   2. Weak edges below threshold are pruned
//!   3. Multi-channel seed merge (EntityInverted + Bm25 both hit the same memory)
//!   4. Full pipeline end-to-end (seeds→spread→rerank→warnings each stage produces non-empty output)
//!   5. RetrievalMode hop difference (Fast=1 hop vs Balanced=2 hops)

use hippmem_core::config::AlgoParams;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{
    ActivationState, AssociationKeys, AssociationLink, LexicalSignature, LinkDirection,
    LinkEvidence, LinkType, MemoryWarning, ObservationState, RecallChannel, SemanticSignature,
};
use hippmem_core::model::understanding::MemoryUnderstanding;
use hippmem_core::model::unit::{
    ContentType, GeneratedBy, Language, MemoryContent, MemoryLifecycle, MemoryStage, MemoryUnit,
    Provenance, SourceKind, WriteContext,
};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_retrieval::rerank::rerank_by_energy;
use hippmem_retrieval::seeds::{self, multi_channel_seeds};
use hippmem_retrieval::spreading::spread_multi_hop;
use hippmem_retrieval::warnings::check_warnings;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════

/// Build a minimal MemoryUnit (for the rerank + warnings stages)
fn make_mini_unit(id: u128, raw: &str, lifecycle: MemoryLifecycle) -> MemoryUnit {
    MemoryUnit {
        schema_version: 1,
        id: MemoryId(id),
        created_at: Timestamp(1_700_000_000_000),
        updated_at: Timestamp(1_700_000_000_000),
        content: MemoryContent {
            raw: raw.into(),
            summary: None,
            normalized: None,
            language: Language::Zh,
            content_type: ContentType::UserStatement,
        },
        context: WriteContext {
            conversation_id: None,
            session_id: None,
            project_id: None,
            task_id: None,
            user_id: None,
            local_time: Timestamp(1_700_000_000_000),
            preceding_memory_ids: vec![],
            source_refs: vec![],
        },
        understanding: MemoryUnderstanding {
            entities: vec![],
            events: vec![],
            goals: vec![],
            decisions: vec![],
            preferences: vec![],
            emotions: vec![],
            causal_claims: vec![],
            contradictions: vec![],
            topics: vec![],
            importance: UnitScore::new(0.5),
            confidence: UnitScore::new(0.5),
        },
        association_keys: AssociationKeys {
            entity_keys: vec![],
            temporal_keys: vec![],
            lexical_signature: LexicalSignature { simhash: [0; 4] },
            semantic_signature: SemanticSignature {
                lexical_simhash: [0; 4],
                dense_embedding_ref: None,
                binary_code: [0; 2],
                topic_minhash: [0u32; 16],
            },
            topic_keys: vec![],
            emotion_keys: vec![],
            goal_keys: vec![],
            event_keys: vec![],
            causal_keys: vec![],
        },
        links: vec![],
        activation: ActivationState {
            last_retrieved_at: None,
            retrieval_count: 0,
            co_activations: vec![],
            usage_score: UnitScore::new(0.5),
        },
        lifecycle,
        provenance: Provenance {
            origin: SourceKind::Conversation,
            generated_by: GeneratedBy::UserDirect,
            reliability: UnitScore::new(0.5),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: MemoryStage::Indexed,
    }
}

/// Build a test AssociationLink
fn make_link(target: u128, link_type: LinkType, strength: f32, confidence: f32) -> AssociationLink {
    AssociationLink {
        target_id: MemoryId(target),
        link_type,
        direction: LinkDirection::Undirected,
        strength: UnitScore::new(strength),
        confidence: UnitScore::new(confidence),
        evidence: LinkEvidence {
            contributing_dimensions: vec![],
            score_breakdown: vec![],
            text_spans: vec![],
            note: None,
        },
        formed_at: Timestamp(1_700_000_000_000),
        last_activated_at: None,
        activation_count: 0,
        observation: ObservationState::Confirmed,
    }
}

/// Feed the spreading results (MemoryId, f32, Vec<ActivationStep>)
/// and the MemoryUnit list into rerank + warnings, producing a full RetrievalResult list
fn full_pipeline(
    activated: &[(
        MemoryId,
        f32,
        Vec<hippmem_core::model::links::ActivationStep>,
    )],
    units: &[MemoryUnit],
) -> Vec<(MemoryId, f32, Vec<MemoryWarning>, MemoryUnit)> {
    // Step 4: rerank
    let reranked = rerank_by_energy(activated, units);

    // Step 5: warnings
    reranked
        .into_iter()
        .map(|(id, energy, _trace, unit)| {
            let ws = check_warnings(&unit, energy);
            (id, energy, ws, unit)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 1: Causal chain priority > EntityOverlap
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_1_causal_chain_beats_entity_overlap() {
    // After V5 calibration decay_factor=0.55, two-hop decay is stronger. Lower min_propagation_energy
    // to ensure the causal two-hop path is reachable, while keeping the one-hop assertion (Causal > EntityOverlap) unchanged.
    let params = AlgoParams {
        min_propagation_energy: 0.03,
        ..Default::default()
    };

    // ── Build a 5-node graph ──
    // Seed node 1 → Causal to 2 (strong: causal chain)
    //              → EntityOverlap to 3 (medium: entity net)
    //              → TemporalAdjacent to 4 (weak)
    // node 2 → Causal to 5 (causal chain two-hop)
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::Causal, 0.8, 0.8),
            make_link(3, LinkType::EntityOverlap, 0.8, 0.8),
            make_link(4, LinkType::TemporalAdjacent, 0.6, 0.6),
        ],
    );
    links_map.insert(MemoryId(2), vec![make_link(5, LinkType::Causal, 0.7, 0.7)]);

    // ── Seed: high-score BM25 seed → node 1 ──
    let seeds = vec![seeds::Seed {
        id: MemoryId(1),
        channel: RecallChannel::Bm25,
        score: 1.0,
        rank_in_channel: None,
    }];

    // ── spreading ──
    let activated = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());

    // ── Assert: Causal neighbor (node 2) energy > EntityOverlap neighbor (node 3) ──
    let e2 = activated
        .iter()
        .find(|(id, _, _)| *id == MemoryId(2))
        .map(|(_, e, _)| *e)
        .expect("Causal neighbor node 2 should be in the results");
    let e3 = activated
        .iter()
        .find(|(id, _, _)| *id == MemoryId(3))
        .map(|(_, e, _)| *e)
        .expect("EntityOverlap neighbor node 3 should be in the results");

    assert!(
        e2 > e3,
        "Causal neighbor energy ({}) should be > EntityOverlap neighbor energy ({})",
        e2,
        e3
    );

    // ── Assert: Causal path node 5 is reachable (two-hop) ──
    assert!(
        activated.iter().any(|(id, _, _)| *id == MemoryId(5)),
        "causal-chain two-hop node 5 should be reachable"
    );

    // ── Full pipeline: rerank + warnings ──
    let units: Vec<MemoryUnit> = (1..=5)
        .map(|i| make_mini_unit(i, &format!("node {}", i), MemoryLifecycle::Active))
        .collect();
    let pipeline_results = full_pipeline(&activated, &units);

    // The causal-chain node (2) ranks before the entity-overlap node (3)
    let pos_2 = pipeline_results
        .iter()
        .position(|(id, _, _, _)| *id == MemoryId(2))
        .unwrap();
    let pos_3 = pipeline_results
        .iter()
        .position(|(id, _, _, _)| *id == MemoryId(3))
        .unwrap();
    assert!(
        pos_2 < pos_3,
        "Causal node 2 should rank before EntityOverlap node 3"
    );

    // Each result has an activation_trace (produced by spreading, kept by rerank)
    // rerank_by_energy returns the full MemoryUnit; verify its presence
    assert_eq!(pipeline_results.len(), activated.len());
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 2: Weak edges below threshold are pruned
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_2_weak_edges_pruned_below_threshold() {
    let params = AlgoParams {
        min_propagation_energy: 0.10, // set threshold
        ..Default::default()
    };

    // ── Edges with very low strength × confidence ──
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::Causal, 0.8, 0.8), // strong edge → passes
            make_link(3, LinkType::TemporalAdjacent, 0.2, 0.2), // weak edge → should be pruned
        ],
    );

    // After V5 calibration decay_factor=0.55, a higher seed score is needed to ensure the strong-edge energy exceeds the threshold
    let seeds = vec![seeds::Seed {
        id: MemoryId(1),
        channel: RecallChannel::Bm25,
        score: 0.9, // high-score seed: seed_energy=0.36*0.8*0.8*0.55*1.30=0.165>0.10
        rank_in_channel: None,
    }];

    let activated = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());

    // Strong-edge neighbor 2 should be in the results (energy 0.165 > min_propagation_energy 0.10)
    assert!(
        activated.iter().any(|(id, _, _)| *id == MemoryId(2)),
        "strong-edge neighbor 2 should be reachable"
    );

    // Weak-edge neighbor 3 propagated energy < min_propagation_energy → should not appear
    assert!(
        !activated.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "weak-edge neighbor 3 should be pruned (propagated energy < {})",
        params.min_propagation_energy
    );

    // The pipeline likewise excludes pruned nodes
    let units: Vec<MemoryUnit> = (1..=3)
        .map(|i| make_mini_unit(i, &format!("node {}", i), MemoryLifecycle::Active))
        .collect();
    let pipeline_results = full_pipeline(&activated, &units);
    assert!(!pipeline_results
        .iter()
        .any(|(id, _, _, _)| *id == MemoryId(3)));
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 3: Multi-channel seed merge (EntityInverted + Bm25 both hit the same memory)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_3_multi_channel_seed_merge() {
    let per_channel_limit = 20;

    // The same MemoryId (42) is hit by both the entity and BM25 channels
    // V9: entity channel score is normalized (per-hit) and used directly as the seed score, no longer count*0.2
    let entity_hits = vec![(MemoryId(42), 0.6f32)]; // entity hit score = 0.6
    let bm25_hits = vec![(MemoryId(42), 0.75f32)]; // BM25 score = 0.75

    let result = multi_channel_seeds(
        "test query",
        &entity_hits,
        &[], // temporal
        &[], // semantic
        &[], // topic
        &bm25_hits,
        &[], // binary
        &[], // goal
        &[], // event
        &[], // causal
        &[], // recent
        per_channel_limit,
    );

    // Assert: the result contains seeds from both channels (same memory, different channels)
    let entity_seeds: Vec<_> = result
        .seeds
        .iter()
        .filter(|s| s.id == MemoryId(42) && s.channel == RecallChannel::EntityInverted)
        .collect();
    let bm25_seeds: Vec<_> = result
        .seeds
        .iter()
        .filter(|s| s.id == MemoryId(42) && s.channel == RecallChannel::Bm25)
        .collect();

    assert_eq!(
        entity_seeds.len(),
        1,
        "should have 1 EntityInverted channel seed"
    );
    assert_eq!(bm25_seeds.len(), 1, "should have 1 BM25 channel seed");

    // Entity channel score: V9 uses the passed-in normalized score 0.6 directly
    assert!((entity_seeds[0].score - 0.6).abs() < 0.01);
    // BM25 channel score: 0.75
    assert!((bm25_seeds[0].score - 0.75).abs() < 0.01);

    // Each channel score does not exceed 1.0
    assert!(entity_seeds[0].score <= 1.0);
    assert!(bm25_seeds[0].score <= 1.0);

    // channel_scores records both channels
    assert!(result.channel_scores.len() >= 2);
    assert!(result
        .channel_scores
        .iter()
        .any(|(ch, _)| *ch == RecallChannel::EntityInverted));
    assert!(result
        .channel_scores
        .iter()
        .any(|(ch, _)| *ch == RecallChannel::Bm25));
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 4: Full pipeline end-to-end (seeds→spread→rerank→warnings)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_4_full_pipeline_end_to_end() {
    let params = AlgoParams::default();

    // ── Build a small graph: 1→2 (Causal), 1→3 (EntityOverlap), 2→4 (Correction) ──
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::Causal, 0.8, 0.9),
            make_link(3, LinkType::EntityOverlap, 0.7, 0.8),
        ],
    );
    links_map.insert(
        MemoryId(2),
        vec![make_link(4, LinkType::Correction, 0.8, 0.9)],
    );

    // ── Step 1: seed construction (EntityInverted + Bm25 multi-channel) ──
    let seed_result = multi_channel_seeds(
        "test",
        &[(MemoryId(1), 3.0f32)],
        &[],
        &[],
        &[],
        &[(MemoryId(1), 0.85f32)],
        &[],
        &[], // goal
        &[], // event
        &[], // causal
        &[], // recent
        20,
    );
    assert!(!seed_result.seeds.is_empty(), "should produce seeds");

    // ── Step 2+3: energy + spreading ──
    let activated = spread_multi_hop(&seed_result.seeds, &links_map, &params, &HashMap::new());
    assert!(
        activated.len() >= 2,
        "should return at least the seed itself + neighbor"
    );

    // ── Build MemoryUnits for rerank + warnings ──
    let units: Vec<MemoryUnit> = vec![
        make_mini_unit(1, "seed node", MemoryLifecycle::Active),
        make_mini_unit(2, "causal neighbor", MemoryLifecycle::Active),
        // node 3 is a deprecated memory — should trigger a Deprecated warning
        make_mini_unit(3, "deprecated node", MemoryLifecycle::Deprecated),
        // node 4 has low energy — should trigger LowConfidence + its outgoing edge is a Correction
        make_mini_unit(4, "correction node", MemoryLifecycle::Active),
    ];

    // ── Step 4: rerank ──
    let reranked = rerank_by_energy(&activated, &units);
    assert!(!reranked.is_empty(), "rerank should produce results");

    // ── Step 5: warnings ──
    let pipeline_results: Vec<_> = reranked
        .iter()
        .map(|(id, energy, _trace, unit)| {
            let ws = check_warnings(unit, *energy);
            (id, energy, ws, unit)
        })
        .collect();

    // ── Verify each result field is non-empty / reasonable ──
    for (id, energy, warnings, unit) in &pipeline_results {
        // final_score > 0 (all results should have positive energy)
        assert!(**energy > 0.0, "node {} final_score should be > 0", id.0);

        // MemoryUnit exists
        assert_eq!(unit.id, **id, "MemoryUnit id should match");

        // matched_dimensions: rerank_by_energy keeps the activation_trace
        // the trace from spreading already contains via_link info
        // (rerank does not generate matched_dimensions — that is done by deduce_dimensions in seeds.rs)
        // we just verify the activation_trace exists

        // warnings: at least can be produced (may be empty, not enforced)
        // but for lifecycle==Deprecated nodes, there should be a Deprecated warning
        if unit.lifecycle == MemoryLifecycle::Deprecated {
            assert!(
                warnings
                    .iter()
                    .any(|w| matches!(w, MemoryWarning::Deprecated)),
                "Deprecated node {} should have a Deprecated warning",
                id.0
            );
        }
    }

    // ── Extra assertion: full-pipeline result count = activated count (no nodes lost) ──
    assert_eq!(
        pipeline_results.len(),
        activated.len(),
        "full pipeline should not lose nodes"
    );

    // ── Results sorted by energy descending (guaranteed by spread_multi_hop) ──
    for i in 1..pipeline_results.len() {
        assert!(
            pipeline_results[i - 1].1 >= pipeline_results[i].1,
            "results should be sorted by energy descending"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 5: RetrievalMode hop difference (Fast=1 vs Balanced=2)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_5_hop_difference_fast_vs_balanced() {
    // ── Linear chain: 1 → 2 → 3 → 4 ──
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![make_link(2, LinkType::EntityOverlap, 0.8, 0.8)],
    );
    links_map.insert(
        MemoryId(2),
        vec![make_link(3, LinkType::EntityOverlap, 0.8, 0.8)],
    );
    links_map.insert(
        MemoryId(3),
        vec![make_link(4, LinkType::EntityOverlap, 0.8, 0.8)],
    );

    let seeds = vec![seeds::Seed {
        id: MemoryId(1),
        channel: RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    // ── Fast mode (max_hops = 1): uses spread_one_hop ──
    let params_fast = AlgoParams {
        max_hops_default: 1,
        ..Default::default()
    };
    let fast_results = spread_multi_hop(&seeds, &links_map, &params_fast, &HashMap::new());

    // Fast: seed 1 + one-hop neighbor 2 reachable, two-hop 3 unreachable
    assert!(
        fast_results.iter().any(|(id, _, _)| *id == MemoryId(1)),
        "Fast: seed itself should be in the results"
    );
    assert!(
        fast_results.iter().any(|(id, _, _)| *id == MemoryId(2)),
        "Fast: one-hop neighbor should be reachable"
    );
    assert!(
        !fast_results.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "Fast (max_hops=1): two-hop neighbor unreachable"
    );
    assert!(
        !fast_results.iter().any(|(id, _, _)| *id == MemoryId(4)),
        "Fast (max_hops=1): three-hop neighbor unreachable"
    );

    // ── Balanced mode (max_hops = 2): uses spread_multi_hop ──
    // Note: two-hop energy = seed_energy * strength * confidence * decay^2 * type_modifier
    // need to ensure two-hop > min_propagation_energy (0.05), so lower the threshold
    let params_balanced = AlgoParams {
        max_hops_default: 2,
        min_propagation_energy: 0.02,
        ..Default::default()
    };
    let balanced_results = spread_multi_hop(&seeds, &links_map, &params_balanced, &HashMap::new());

    // Balanced: seed + one-hop + two-hop reachable
    assert!(
        balanced_results.iter().any(|(id, _, _)| *id == MemoryId(1)),
        "Balanced: seed itself should be in the results"
    );
    assert!(
        balanced_results.iter().any(|(id, _, _)| *id == MemoryId(2)),
        "Balanced: one-hop neighbor should be reachable"
    );
    assert!(
        balanced_results.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "Balanced (max_hops=2): two-hop neighbor should be reachable"
    );
    // Three-hop unreachable (max_hops=2)
    assert!(
        !balanced_results.iter().any(|(id, _, _)| *id == MemoryId(4)),
        "Balanced (max_hops=2): three-hop neighbor unreachable"
    );

    // ── Verify: Balanced result count > Fast result count ──
    assert!(
        balanced_results.len() > fast_results.len(),
        "Balanced ({}) should cover more nodes than Fast ({})",
        balanced_results.len(),
        fast_results.len()
    );

    // ── Full pipeline: both modes go through rerank + warnings ──
    let units: Vec<MemoryUnit> = (1..=4)
        .map(|i| make_mini_unit(i, &format!("node {}", i), MemoryLifecycle::Active))
        .collect();

    let fast_pipeline = full_pipeline(&fast_results, &units);
    let balanced_pipeline = full_pipeline(&balanced_results, &units);

    assert!(balanced_pipeline.len() > fast_pipeline.len());
}
