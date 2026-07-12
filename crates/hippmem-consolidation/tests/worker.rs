//! acceptance tests: compaction/merge + background worker cycle.

use hippmem_consolidation::summarize::{build_summary_unit, should_summarize};
use hippmem_consolidation::worker::ConsolidationWorker;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::unit::MemoryUnit;
use hippmem_model::deterministic::summarize::DeterministicSummarizer;

fn make_mini_unit(id: u128, text: &str) -> MemoryUnit {
    MemoryUnit {
        schema_version: 1,
        id: MemoryId(id),
        created_at: hippmem_core::time::Timestamp(0),
        updated_at: hippmem_core::time::Timestamp(0),
        content: hippmem_core::model::unit::MemoryContent {
            raw: text.into(),
            summary: None,
            normalized: None,
            language: hippmem_core::model::unit::Language::Zh,
            content_type: hippmem_core::model::unit::ContentType::UserStatement,
        },
        context: hippmem_core::model::unit::WriteContext {
            conversation_id: None,
            session_id: None,
            project_id: None,
            task_id: None,
            user_id: None,
            local_time: hippmem_core::time::Timestamp(0),
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
            importance: hippmem_core::score::UnitScore::new(0.5),
            confidence: hippmem_core::score::UnitScore::new(0.5),
        },
        association_keys: hippmem_core::model::links::AssociationKeys {
            entity_keys: vec![],
            temporal_keys: vec![],
            lexical_signature: hippmem_core::model::links::LexicalSignature { simhash: [0; 4] },
            semantic_signature: hippmem_core::model::links::SemanticSignature {
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
        activation: hippmem_core::model::links::ActivationState {
            last_retrieved_at: None,
            retrieval_count: 0,
            co_activations: vec![],
            usage_score: hippmem_core::score::UnitScore::new(0.5),
        },
        lifecycle: hippmem_core::model::unit::MemoryLifecycle::Active,
        provenance: hippmem_core::model::unit::Provenance {
            origin: hippmem_core::model::unit::SourceKind::Conversation,
            generated_by: hippmem_core::model::unit::GeneratedBy::UserDirect,
            reliability: hippmem_core::score::UnitScore::new(0.5),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: hippmem_core::model::unit::MemoryStage::Indexed,
    }
}

#[test]
fn should_summarize_threshold() {
    let ids: Vec<MemoryId> = (0..15).map(MemoryId).collect();
    assert!(should_summarize(&ids, 12));
    assert!(!should_summarize(&ids[..5], 12));
}

#[test]
fn build_summary_creates_covers_chain() {
    let sources: Vec<MemoryUnit> = vec![
        make_mini_unit(1, "Fixed a performance regression in the query pipeline"),
        make_mini_unit(
            2,
            "Root cause: Tantivy index was not reusing segment readers across queries",
        ),
        make_mini_unit(
            3,
            "After optimization: query latency reduced by 50%, p99 from 200ms to 95ms",
        ),
    ];
    let summary = build_summary_unit(&sources, &DeterministicSummarizer);
    assert!(
        summary.content.summary.is_some(),
        "should produce summary text"
    );
    assert_eq!(
        summary.links.len(),
        sources.len(),
        "should have a covers chain"
    );
    for link in &summary.links {
        assert_eq!(
            link.link_type,
            hippmem_core::model::links::LinkType::Elaboration
        );
    }
}

#[test]
fn worker_has_initial_state() {
    let worker = ConsolidationWorker::default();
    assert_eq!(worker.cycle_count(), 0);
}

#[test]
fn worker_runs_consolidation_cycle() {
    let mut worker = ConsolidationWorker::default();
    let mut units = vec![make_mini_unit(1, "test memory for consolidation batch")];
    let now = hippmem_core::time::Timestamp::from_millis(1_700_000_000_000);

    // Run one cycle (empty co-activation, normal timestamp)
    let stats = worker.run_cycle(&mut units, &[], now, None);
    assert_eq!(worker.cycle_count(), 1);
    assert!(
        stats.edges_decayed < 100,
        "empty edges should not decay heavily"
    );

    // Empty data should not panic
    let stats = worker.run_cycle(&mut Vec::new(), &[], now, None);
    assert_eq!(worker.cycle_count(), 2);
    assert_eq!(stats.edges_archived, 0);
}
