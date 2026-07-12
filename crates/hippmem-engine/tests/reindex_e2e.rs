//! V6 Reindex end-to-end tests.
//!
//! Verifies behavioral correctness of (consolidate(Reindex) rebuilds all secondary
//! indexes from memory_log). Uses deterministic backends only; no network dependency.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{
    ConsolidationScope, Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput,
};
use tempfile::tempdir;

fn make_ctx() -> WriteContext {
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

fn open_engine(dir: &tempfile::TempDir) -> Engine {
    let store_path = dir.path().join("hippmem_data");
    let config = EngineConfig {
        store_dir: store_path,
        ..Default::default()
    };
    Engine::open(config).expect("open engine failed")
}

/// Test 1: write 3 memories -> Reindex -> retrieval non-empty and MemoryId unchanged.
#[test]
fn reindex_retrieval_nonempty_and_ids_unchanged() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    // Write 3 memories
    let contents = [
        "Because RocksDB compilation was too slow, decided to use redb as embedded storage",
        "Love Rust's trait system and pattern matching",
        "Cache solution uses Redis Write-Through strategy",
    ];
    let mut original_ids = Vec::new();
    for content in &contents {
        let out = engine
            .write(WriteMemoryInput {
                content: content.to_string(),
                content_type: Some(ContentType::Decision),
                context: make_ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        original_ids.push(out.memory_id);
    }

    // Capture original retrieval result
    let before = engine
        .retrieve(RetrieveInput {
            query: "why choose redb".into(),
            context: RetrieveContext::default(),
            top_k: 5,
            max_hops: Some(2),
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap();
    assert!(
        !before.results.is_empty(),
        "retrieval should be non-empty after write"
    );

    // Run Reindex
    let report = engine
        .consolidate(ConsolidationScope::Reindex)
        .expect("reindex should succeed");
    assert!(report.reindexed, "report.reindexed should be true");
    assert_eq!(report.memories_processed, 3, "should process 3 memories");

    // Retrieval still non-empty after Reindex
    let after = engine
        .retrieve(RetrieveInput {
            query: "why choose redb".into(),
            context: RetrieveContext::default(),
            top_k: 5,
            max_hops: Some(2),
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap();
    assert!(
        !after.results.is_empty(),
        "retrieval should be non-empty after Reindex"
    );

    // Verify the first original memory (redb) is present in results
    let after_ids: std::collections::HashSet<_> =
        after.results.iter().map(|r| r.memory.id).collect();
    assert!(
        after_ids.contains(&original_ids[0]),
        "redb memory should still be retrievable after Reindex"
    );

    engine.close().expect("close failed");
}

/// Test 2: MemoryId set unchanged after Reindex (no data loss).
#[test]
fn reindex_preserves_all_memory_ids() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    let contents = [
        "Memory A: Project uses Rust for development",
        "Memory B: Database choice is Redb",
        "Memory C: Deployed on AWS",
        "Memory D: Uses GitHub Actions CI",
        "Memory E: Monitoring uses Prometheus",
    ];
    let mut original_ids = Vec::new();
    for content in &contents {
        let out = engine
            .write(WriteMemoryInput {
                content: content.to_string(),
                content_type: Some(ContentType::ProjectKnowledge),
                context: make_ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        original_ids.push(out.memory_id);
    }

    // Reindex
    engine
        .consolidate(ConsolidationScope::Reindex)
        .expect("reindex should succeed");

    // Verify all MemoryIds are still retrievable
    let all = engine
        .retrieve(RetrieveInput {
            query: "memory".into(),
            context: RetrieveContext::default(),
            top_k: 10,
            max_hops: Some(2),
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap();

    let all_ids: std::collections::HashSet<_> = all.results.iter().map(|r| r.memory.id).collect();
    assert_eq!(
        all_ids.len(),
        5,
        "should retrieve all 5 memories, got: {}",
        all_ids.len()
    );
    for id in &original_ids {
        assert!(all_ids.contains(id), "MemoryId {:?} missing", id);
    }

    engine.close().expect("close failed");
}

/// Test 3: Reindex on empty database does not error (idempotent).
#[test]
fn reindex_empty_database_is_idempotent() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    // Running Reindex on an empty database should not error
    let report = engine
        .consolidate(ConsolidationScope::Reindex)
        .expect("Reindex on empty database should succeed");
    assert!(report.reindexed);
    assert_eq!(report.memories_processed, 0);

    // Running Reindex again also does not error
    let report2 = engine
        .consolidate(ConsolidationScope::Reindex)
        .expect("second Reindex should succeed");
    assert!(report2.reindexed);
    assert_eq!(report2.memories_processed, 0);

    engine.close().expect("close failed");
}

/// Test 4: write and retrieve work normally after Reindex.
#[test]
fn reindex_then_write_and_retrieve_works() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    // Write one memory first
    engine
        .write(WriteMemoryInput {
            content: "Original memory: User prefers Rust".into(),
            content_type: Some(ContentType::Preference),
            context: make_ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // Reindex
    engine
        .consolidate(ConsolidationScope::Reindex)
        .expect("reindex should succeed");

    // Write still works after Reindex
    let new_out = engine
        .write(WriteMemoryInput {
            content: "New memory: User also started learning Go".into(),
            content_type: Some(ContentType::UserStatement),
            context: make_ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // Newly written memory is retrievable
    let results = engine
        .retrieve(RetrieveInput {
            query: "Go".into(),
            context: RetrieveContext::default(),
            top_k: 5,
            max_hops: Some(2),
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap();

    assert!(
        !results.results.is_empty(),
        "new write should be retrievable after Reindex"
    );
    assert!(
        results
            .results
            .iter()
            .any(|r| r.memory.id == new_out.memory_id),
        "newly written memory should be in retrieval results"
    );

    engine.close().expect("close failed");
}
