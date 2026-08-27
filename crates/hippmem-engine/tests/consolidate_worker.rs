//! acceptance test: background consolidation worker (方案 B,
//! consolidate-worker-design — engine-side periodic consolidate).
//!
//! The worker holds only a Weak<Engine>, runs `consolidate` every
//! `consolidate_interval_ms`, and is stopped+joined by `close`.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{BackgroundConfig, Engine, EngineConfig, WriteMemoryInput};
use std::time::Duration;
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

fn write_one(engine: &Engine, content: &str) {
    engine
        .write(WriteMemoryInput {
            content: content.into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
}

/// With a short interval the worker runs several consolidations during the
/// engine's life; close() must stop and join it without hanging or
/// panicking, and the store must still be usable.
#[test]
fn background_worker_runs_and_close_is_clean() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        background: BackgroundConfig {
            consolidate_interval_ms: 100,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();

    write_one(&engine, "小明住在北京海淀区。");
    write_one(&engine, "小明和李华是高中同学。");

    // Let the worker tick a few times while the engine is live.
    std::thread::sleep(Duration::from_millis(400));

    // Still functional after worker ticks.
    let out = engine
        .list(hippmem_engine::ListInput {
            limit: 10,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(out.total, 2, "memories survive background consolidations");

    // close() joins the worker — must return promptly (no hang) and cleanly.
    engine.close().unwrap();
}

/// interval = 0 disables the worker entirely.
#[test]
fn background_disabled_with_zero_interval() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        background: BackgroundConfig {
            consolidate_interval_ms: 0,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    write_one(&engine, "王芳毕业于北京大学。");
    engine.close().unwrap();
}
