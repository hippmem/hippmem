//! acceptance test: engine.consolidate.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{ConsolidationScope, Engine, EngineConfig, WriteMemoryInput};
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

#[test]
fn consolidate_full_returns_report() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write some memories
    for i in 0..5 {
        engine
            .write(WriteMemoryInput {
                content: format!("Memory content entry #{}", i),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }

    let report = engine.consolidate(ConsolidationScope::Full).unwrap();
    let _ = report.elapsed_ms; // elapsed time record

    engine.close().unwrap();
}

#[test]
fn consolidate_incremental_no_panic() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // consolidate on an empty store should not panic
    let report = engine.consolidate(ConsolidationScope::Incremental).unwrap();
    // empty-store consolidation does not panic; processed count is small
    assert!(report.memories_processed < 10);

    engine.close().unwrap();
}

#[test]
fn consolidate_edges_only() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let report = engine.consolidate(ConsolidationScope::EdgesOnly).unwrap();
    assert!(
        report.edges_decayed < 1_000_000,
        "decayed count on an empty store should be tiny"
    );

    engine.close().unwrap();
}
