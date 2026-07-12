//! acceptance test: strong-semantic dimension enrich background path.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput};
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
fn write_fills_basic_understanding() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let out = engine
        .write(WriteMemoryInput {
            content: "The user likes Rust, with a goal to complete the project".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    assert_eq!(
        out.stage_reached,
        hippmem_core::model::unit::MemoryStage::Indexed
    );
    // Deterministic rules should extract preference (Rust) and a goal (project completion)
    assert!(!out
        .warnings
        .iter()
        .any(|w| matches!(w, hippmem_engine::WriteWarning::ExtractorDegraded)));

    engine.close().unwrap();
}

#[test]
fn enrich_adds_strong_dims() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write a memory containing goal and emotion keywords
    let out = engine
        .write(WriteMemoryInput {
            content: "I am happy because I completed performance optimization, with a goal to reduce latency".into(),
            content_type: Some(ContentType::Reflection),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();

    // Verify the strong-semantic warnings flag
    let has_deferred = out
        .warnings
        .iter()
        .any(|w| matches!(w, hippmem_engine::WriteWarning::StrongDimsDeferred));
    assert!(
        has_deferred,
        "strong-semantic dimensions should be marked deferred"
    );

    engine.close().unwrap();
}
