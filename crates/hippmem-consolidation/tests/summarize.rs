//! acceptance tests (module layer): summary/merge Summarizer integration + covers chain
//!
//! Verifies 03 §8 compaction and merging: summary trigger → Summarizer call → covers chain → confidence gating.
//! 4 scenarios:
//!   1. 15 similar low-importance memories → trigger summary, 1 Reflection summary memory
//!   2. covers chain: summary covers all source memories (preceding_memory_ids + Elaboration edges)
//!   3. Confidence gating: low-confidence summary does not replace the originals
//!   4. 5 dissimilar memories → no summary triggered

use hippmem_consolidation::summarize::{build_summary_unit, should_summarize};
use hippmem_core::ids::MemoryId;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::LinkType;
use hippmem_core::model::unit::{MemoryContent, MemoryUnit, WriteContext};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_model::deterministic::summarize::DeterministicSummarizer;

// ═══════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════

fn make_unit(id: u128, raw: &str, importance: f32) -> MemoryUnit {
    MemoryUnit {
        schema_version: 1,
        id: MemoryId(id),
        created_at: Timestamp(1_700_000_000_000),
        updated_at: Timestamp(1_700_000_000_000),
        content: MemoryContent {
            raw: raw.into(),
            summary: None,
            normalized: None,
            language: hippmem_core::model::unit::Language::Zh,
            content_type: ContentType::ToolResult,
        },
        context: WriteContext {
            conversation_id: Some(1),
            session_id: Some(1),
            project_id: None,
            task_id: None,
            user_id: None,
            local_time: Timestamp(1_700_000_000_000),
            preceding_memory_ids: vec![],
            source_refs: vec![],
        },
        understanding: hippmem_core::model::understanding::MemoryUnderstanding {
            entities: vec![],
            events: vec![],
            goals: vec![],
            decisions: vec![],
            preferences: vec![],
            emotions: vec![],
            causal_claims: vec![],
            contradictions: vec![],
            topics: vec![],
            importance: UnitScore::new(importance),
            confidence: UnitScore::new(0.5),
        },
        association_keys: hippmem_core::model::links::AssociationKeys {
            entity_keys: vec![],
            temporal_keys: vec![],
            lexical_signature: hippmem_core::model::links::LexicalSignature { simhash: [0; 4] },
            semantic_signature: hippmem_core::model::links::SemanticSignature {
                lexical_simhash: [0; 4],
                dense_embedding_ref: None,
                binary_code: [0, 0],
                topic_minhash: [0u32; 16],
            },
            topic_keys: vec![],
            emotion_keys: vec![],
            goal_keys: vec![],
            event_keys: vec![],
            causal_keys: vec![],
        },
        links: vec![],
        activation: hippmem_core::model::links::ActivationState {
            last_retrieved_at: None,
            retrieval_count: 0,
            co_activations: vec![],
            usage_score: UnitScore::new(0.0),
        },
        lifecycle: hippmem_core::model::unit::MemoryLifecycle::Active,
        provenance: hippmem_core::model::unit::Provenance {
            origin: hippmem_core::model::unit::SourceKind::Conversation,
            generated_by: hippmem_core::model::unit::GeneratedBy::Extractor {
                backend: "test".into(),
            },
            reliability: UnitScore::new(0.5),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: hippmem_core::model::unit::MemoryStage::Indexed,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 1: Summary trigger — 15 similar low-importance memories
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_1_summary_triggered_with_15_similar_memories() {
    let sources: Vec<MemoryUnit> = (0..15)
        .map(|i| {
            make_unit(
                i + 1,
                &format!("Build output: compiler error E{:04} resolved", i),
                0.3,
            )
        })
        .collect();

    let ids: Vec<MemoryId> = sources.iter().map(|u| u.id).collect();
    assert!(
        should_summarize(&ids, 12),
        "15 entries should trigger a summary (threshold=12)"
    );

    // Use DeterministicSummarizer to generate the summary
    let summarizer = DeterministicSummarizer;
    let summary_unit = build_summary_unit(&sources, &summarizer);

    // Verify the summary type
    assert_eq!(
        summary_unit.content.content_type,
        ContentType::Reflection,
        "summary should be of Reflection type"
    );
    assert_eq!(
        summary_unit.stage,
        hippmem_core::model::unit::MemoryStage::Consolidated,
        "summary stage should be Consolidated"
    );
    assert!(
        summary_unit.understanding.importance.value() >= 0.4,
        "summary importance should be >= 0.4, got {}",
        summary_unit.understanding.importance.value()
    );

    // Verify provenance marker
    assert!(matches!(
        summary_unit.provenance.generated_by,
        hippmem_core::model::unit::GeneratedBy::Consolidation
    ));

    // Verify the covers chain: preceding_memory_ids contains all source memories
    assert_eq!(
        summary_unit.context.preceding_memory_ids.len(),
        15,
        "covers chain should cover all 15 source memories"
    );
    for sid in &ids {
        assert!(
            summary_unit.context.preceding_memory_ids.contains(sid),
            "covers chain should contain source memory {:?}",
            sid
        );
    }

    // Verify Elaboration out-edges (summary → each source)
    for link in &summary_unit.links {
        assert_eq!(link.link_type, LinkType::Elaboration);
        assert!(
            ids.contains(&link.target_id),
            "Elaboration edge should point to a source memory"
        );
    }
    assert_eq!(
        summary_unit.links.len(),
        15,
        "should generate 15 Elaboration out-edges"
    );

    // Summary text should be non-empty
    assert!(
        !summary_unit.content.raw.is_empty(),
        "summary text should not be empty"
    );
    assert!(
        summary_unit.content.summary.is_some(),
        "content.summary field should have a value"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 2: covers chain — dual recording (preceding_memory_ids + Elaboration edges)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_2_covers_chain_both_paths() {
    let sources: Vec<MemoryUnit> = (0..3)
        .map(|i| make_unit(i + 100, &format!("Duplicate log message #{}", i), 0.2))
        .collect();

    let summarizer = DeterministicSummarizer;
    let summary_unit = build_summary_unit(&sources, &summarizer);

    // Path 1: context.preceding_memory_ids
    assert_eq!(summary_unit.context.preceding_memory_ids.len(), 3);

    // Path 2: Elaboration out-edges
    let elaboration_links: Vec<_> = summary_unit
        .links
        .iter()
        .filter(|l| l.link_type == LinkType::Elaboration)
        .collect();
    assert_eq!(elaboration_links.len(), 3);
    for source in &sources {
        assert!(
            elaboration_links.iter().any(|l| l.target_id == source.id),
            "should have an Elaboration edge pointing to source {}",
            source.id.0
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 3: Confidence gating — Summarizer does not replace on low confidence
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_3_confidence_gating_low_confidence_skip() {
    // Only 2 entries → DeterministicSummarizer confidence is low (<0.35)
    let sources: Vec<MemoryUnit> = (0..2)
        .map(|i| make_unit(i + 200, &format!("Simple log entry #{}", i), 0.1))
        .collect();

    let summarizer = DeterministicSummarizer;
    let summary_unit = build_summary_unit(&sources, &summarizer);

    // On low confidence, summary_unit.confidence should reflect the low confidence
    // DeterministicSummarizer confidence formula: min(0.6, len * 0.05)
    // 2 sources → 2 * 0.05 = 0.10 < 0.35
    let conf = summary_unit.understanding.confidence.value();
    // Confidence gating: on low confidence (<0.35), the caller should skip summary creation
    // Here we only verify that build_summary_unit returns the correct confidence value
    assert!(
        conf < 0.35,
        "low-confidence scenario should return confidence < 0.35, got {}",
        conf
    );
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 4: No trigger — 5 dissimilar memories
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_4_no_trigger_with_5_dissimilar() {
    let ids: Vec<MemoryId> = (0..5).map(|i| MemoryId(i + 500)).collect();
    assert!(
        !should_summarize(&ids, 12),
        "5 dissimilar memories should not trigger a summary (threshold=12)"
    );
}
