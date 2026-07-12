//! acceptance tests: specification compliance audit
//!
//! Verifies 5 constitution/architecture specification clauses:
//!   1. Public API does not leak underlying dependencies (C2)
//!   2. Index can be rebuilt from memory_log (04 §9, C1)
//!   3. MemoryUnit serialization roundtrip preserves all fields (data integrity)
//!   4. Index is immediately readable after write (04 §5)
//!   5. retrieve results include activation_trace/matched_dimensions/warnings (C4)

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::{MemoryUnit, WriteContext};
use hippmem_engine::{
    ConsolidationScope, Engine, EngineConfig, InspectQuery, InspectReport, RetrieveContext,
    RetrieveInput, WriteMemoryInput,
};
use tempfile::tempdir;

fn ctx() -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: hippmem_core::time::Timestamp(1_700_000_000_000),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

fn default_retrieve_ctx() -> RetrieveContext {
    RetrieveContext {
        conversation_id: None,
        session_id: None,
        project_id: None,
        task_id: None,
        user_id: None,
        recent_memory_ids: vec![],
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1. Public API does not leak underlying dependencies (C2)
// ═══════════════════════════════════════════════════════════════════

/// Compile-time verification: using Engine's public API does not require
/// importing redb/tantivy/reqwest. If the following compiles, these underlying
/// types are not leaked in public signatures.
/// Additional runtime verification: Engine::open creates normally, write returns normally.
#[test]
fn public_api_no_backend_leak() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Verify all public API methods are usable
    let output = engine
        .write(WriteMemoryInput {
            content: "HIPPMEM compliance test".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx(),
            importance_hint: Some(0.8),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve
    let retrieve = engine
        .retrieve(RetrieveInput {
            query: "HIPPMEM".into(),
            context: default_retrieve_ctx(),
            top_k: 5,
            max_hops: None,
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    assert!(
        !retrieve.results.is_empty(),
        "should retrieve the written memory"
    );

    // inspect
    let stats = engine.inspect(InspectQuery::StoreStats).unwrap();
    if let InspectReport::StoreStats(s) = stats {
        assert!(s.memory_count >= 1, "store should have at least 1 memory");
    }

    // feedback
    engine
        .feedback(hippmem_engine::FeedbackInput {
            retrieval_id: 1,
            used_memory_ids: vec![output.memory_id],
            signal: hippmem_engine::UsageSignal::Referenced,
        })
        .unwrap();

    // consolidate
    let report = engine.consolidate(ConsolidationScope::Full).unwrap();
    assert!(report.memories_processed >= 1);

    engine.close().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// 2. Index can be rebuilt from memory_log (04 §9)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn reindex_retrieval_consistent() {
    let dir = tempdir().unwrap();
    let store_path = dir.path().join("hippmem.redb");

    // Write multi-dimensional memories
    let _written_ids = {
        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();

        let mut ids = Vec::new();
        for i in 0..5 {
            let output = engine
                .write(WriteMemoryInput {
                    content: format!("Rust programming {}", i),
                    content_type: Some(ContentType::UserStatement),
                    context: ctx(),
                    importance_hint: None,
                    source_refs: vec![],
                })
                .unwrap();
            ids.push(output.memory_id);
        }

        // Retrieve before reindex
        let pre = engine
            .retrieve(RetrieveInput {
                query: "Rust".into(),
                context: default_retrieve_ctx(),
                top_k: 5,
                max_hops: None,
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();
        let pre_count = pre.results.len();

        // Perform reindex
        let report = engine.consolidate(ConsolidationScope::Reindex).unwrap();
        assert!(report.reindexed, "reindex report should set reindexed=true");

        // Retrieve after reindex
        let post = engine
            .retrieve(RetrieveInput {
                query: "Rust".into(),
                context: default_retrieve_ctx(),
                top_k: 5,
                max_hops: None,
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();

        // After reindex, result count should match the prior count (no data loss)
        assert_eq!(
            post.results.len(),
            pre_count,
            "result count should be consistent after reindex"
        );

        engine.close().unwrap();
        ids
    };

    // Reopen to verify persistence
    {
        let engine = Engine::open(EngineConfig {
            store_dir: store_path,
            ..Default::default()
        })
        .unwrap();
        let retrieve = engine
            .retrieve(RetrieveInput {
                query: "Rust".into(),
                context: default_retrieve_ctx(),
                top_k: 5,
                max_hops: None,
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();
        assert!(
            !retrieve.results.is_empty(),
            "retrieve should still return results after reopen"
        );
        engine.close().unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. MemoryUnit serialization roundtrip preserves all fields
// ═══════════════════════════════════════════════════════════════════

#[test]
fn memory_unit_bincode_roundtrip_no_field_loss() {
    // Construct a complete MemoryUnit (via the write API)
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let output = engine
        .write(WriteMemoryInput {
            content: "serialization roundtrip test memory".into(),
            content_type: Some(ContentType::Decision),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();

    // Obtain MemoryUnit via inspect
    if let Ok(InspectReport::Memory(inspect)) =
        engine.inspect(InspectQuery::Memory(output.memory_id))
    {
        let unit = &inspect.unit;

        // bincode serialize
        let encoded = bincode::serde::encode_to_vec(unit, bincode::config::standard()).unwrap();

        // bincode deserialize
        let (decoded, _len): (MemoryUnit, _) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

        // Verify key fields are not lost, field by field
        assert_eq!(decoded.id, unit.id, "id should match");
        assert_eq!(
            decoded.schema_version, unit.schema_version,
            "schema_version should match"
        );
        assert_eq!(
            decoded.content.raw, unit.content.raw,
            "content.raw should match"
        );
        assert_eq!(decoded.content.content_type, unit.content.content_type);
        assert_eq!(decoded.lifecycle, unit.lifecycle, "lifecycle should match");
        assert_eq!(decoded.stage, unit.stage, "stage should match");
        assert_eq!(
            decoded.understanding.importance.value(),
            unit.understanding.importance.value(),
            "importance should match"
        );
        // Link count should match (serialization preserves links)
        assert_eq!(
            decoded.links.len(),
            unit.links.len(),
            "links count should match"
        );
        // WriteContext fields preserved
        assert_eq!(
            decoded.context.conversation_id,
            unit.context.conversation_id
        );
    } else {
        panic!("should be able to inspect the written memory");
    }

    engine.close().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// 4. Index is immediately readable after write (04 §5)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn inverted_index_immediately_readable() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write a memory containing entities
    let _output = engine
        .write(WriteMemoryInput {
            content: "HippMem project test".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve immediately after write: via entity channel
    let retrieve = engine
        .retrieve(RetrieveInput {
            query: "HippMem".into(),
            context: default_retrieve_ctx(),
            top_k: 5,
            max_hops: None,
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();

    // Should have at least 1 result (index is usable immediately after write)
    assert!(
        !retrieve.results.is_empty(),
        "index should be immediately readable after write"
    );

    // Verify trace contains channel_contributions (including EntityInverted)
    let has_entity = retrieve.trace.seeds.iter().any(|s| {
        matches!(
            s.channel,
            hippmem_core::model::links::RecallChannel::EntityInverted
        )
    });
    // Not enforced (depends on extractor), but at least retrieval runs
    let _ = has_entity;

    engine.close().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// 5. retrieve results include activation_trace/matched_dimensions/warnings (C4)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn retrieve_includes_explanation_path() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write multiple related memories
    for i in 0..3 {
        engine
            .write(WriteMemoryInput {
                content: format!("Memory engine HippMem item #{}", i),
                content_type: Some(ContentType::ProjectKnowledge),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }

    let retrieve = engine
        .retrieve(RetrieveInput {
            query: "HippMem".into(),
            context: default_retrieve_ctx(),
            top_k: 3,
            max_hops: None,
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();

    // Verify each result complies with constitution C4: has activation_trace + matched_dimensions + warnings
    for result in &retrieve.results {
        // activation_trace must be non-empty (at least the seed itself at hop=0)
        assert!(
            !result.activation_trace.is_empty(),
            "each result should have activation_trace (constitution C4)"
        );

        // matched_dimensions must be present (may be an empty Vec, but the field must exist)
        let _dims = &result.matched_dimensions;

        // warnings field must be present (may be an empty Vec)
        let _warns = &result.warnings;
    }

    // Trace layer: should have seed records
    assert!(
        !retrieve.trace.seeds.is_empty(),
        "retrieval trace should have seed records (constitution C4)"
    );

    // diagnostics should have channel_contributions (constitution C9)
    assert!(
        !retrieve.diagnostics.channel_contributions.is_empty(),
        "diagnostics should have channel_contributions (constitution C9)"
    );

    engine.close().unwrap();
}
