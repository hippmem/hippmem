//! F3 regression tests (0.3.1): compressed summary sources must be excluded at
//! the SEED stage of retrieval, not only from the final results.
//!
//! Background (2026-08-11 test report P1-2): sources were only filtered after
//! reranking (retrieve_api 7c) and in the fallback seed path. In the normal
//! path they still acted as seeds and injected energy into the summary through
//! their Elaboration edges, amplifying it above concrete memories.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::{GeneratedBy, MemoryLifecycle, MemoryUnit, WriteContext};
use hippmem_engine::{
    ConsolidationScope, Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput,
};
use redb::ReadableDatabase;
use redb::ReadableTable;
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

fn retrieve_ctx() -> RetrieveContext {
    RetrieveContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        recent_memory_ids: vec![],
    }
}

/// Ids of all units whose lifecycle is Compressed (summary sources).
fn compressed_ids(db_path: &std::path::Path) -> Vec<u128> {
    let db = redb::Database::create(db_path).unwrap();
    let read_txn = db.begin_read().unwrap();
    let table = read_txn
        .open_table(hippmem_store::store::MEMORY_KV)
        .unwrap();
    let mut out = Vec::new();
    for entry in table.iter().unwrap().flatten() {
        if let Ok((unit, _)) = bincode::serde::decode_from_slice::<MemoryUnit, _>(
            entry.1.value(),
            bincode::config::standard(),
        ) {
            if matches!(unit.lifecycle, MemoryLifecycle::Compressed { .. }) {
                out.push(unit.id.0);
            }
        }
    }
    out
}

/// Builds a store whose 13 near-duplicate memories trigger a summary cluster
/// (Jaccard ≥ threshold, count ≥ trigger, mostly low importance).
/// Chinese near-duplicates with default importance: the same shape as the
/// existing consolidate_summary_indexed test, which is known to pass the
/// summarizer's confidence gate.
fn build_cluster_store() -> (tempfile::TempDir, Engine) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("hippmem.redb");
    let engine = Engine::open(EngineConfig {
        store_dir: db_path.clone(),
        ..Default::default()
    })
    .unwrap();
    for i in 0..13u32 {
        engine
            .write(WriteMemoryInput {
                content: format!("项目{i}采用了 Rust 编写核心引擎，重点优化内存检索性能。"),
                content_type: Some(ContentType::ProjectKnowledge),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }
    (dir, engine)
}

#[test]
fn compressed_sources_are_not_seeds_after_consolidation() {
    let (dir, engine) = build_cluster_store();
    let db_path = dir.path().join("hippmem.redb");

    // Consolidate → a summary is created and its sources become Compressed.
    let report = engine.consolidate(ConsolidationScope::Incremental).unwrap();
    assert!(
        report.summaries_created >= 1,
        "cluster of 13 similar low-importance memories must trigger a summary"
    );
    // redb is single-writer: close before reading the store directly.
    engine.close().unwrap();
    let sources = compressed_ids(&db_path);
    assert!(
        !sources.is_empty(),
        "summary sources must be marked Compressed"
    );

    // Reopen and retrieve with a query that matches the cluster (and the summary text).
    let engine = Engine::open(EngineConfig {
        store_dir: db_path.clone(),
        ..Default::default()
    })
    .unwrap();
    let out = engine
        .retrieve(RetrieveInput {
            query: "项目采用 Rust 编写核心引擎，重点优化内存检索性能".to_string(),
            context: retrieve_ctx(),
            top_k: 10,
            max_hops: Some(2),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();

    // Contract 1: no compressed source appears as a retrieval seed (F3 fix).
    let seed_ids: Vec<u128> = out.trace.seeds.iter().map(|s| s.id.0).collect();
    for sid in &sources {
        assert!(
            !seed_ids.contains(sid),
            "compressed source {sid} must not act as a retrieval seed"
        );
    }

    // Contract 2: no compressed source appears in results (7c, pre-existing).
    let result_ids: Vec<u128> = out.results.iter().map(|r| r.memory.id.0).collect();
    for sid in &sources {
        assert!(
            !result_ids.contains(sid),
            "compressed source {sid} must not appear in retrieval results"
        );
    }

    // Contract 3 (B1, 0.4.0): the summary must NOT hit the direct channels.
    // A summary text is a concatenation of its sources, so it would match
    // every query about them and crowd out the concrete memories — it is
    // excluded from seeds. (Reachability via graph edges is a separate design
    // item tracked with B5; until then summaries stay out of top-k results.)
    // redb is single-writer: close before reading the store directly.
    engine.close().unwrap();
    let summary_ids = summary_ids(&db_path);
    assert!(!summary_ids.is_empty(), "summary units must exist");
    for sid in &summary_ids {
        assert!(
            !seed_ids.contains(sid),
            "summary {sid} must not act as a retrieval seed (B1)"
        );
    }
}

/// Ids of all consolidation-generated units (summaries).
fn summary_ids(db_path: &std::path::Path) -> Vec<u128> {
    let db = redb::Database::create(db_path).unwrap();
    let read_txn = db.begin_read().unwrap();
    let table = read_txn
        .open_table(hippmem_store::store::MEMORY_KV)
        .unwrap();
    let mut out = Vec::new();
    for entry in table.iter().unwrap().flatten() {
        if let Ok((unit, _)) = bincode::serde::decode_from_slice::<MemoryUnit, _>(
            entry.1.value(),
            bincode::config::standard(),
        ) {
            if unit.provenance.generated_by == GeneratedBy::Consolidation {
                out.push(unit.id.0);
            }
        }
    }
    out
}
