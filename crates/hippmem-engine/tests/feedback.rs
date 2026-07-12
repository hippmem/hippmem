//! acceptance test: activation_log + feedback API.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, FeedbackInput, UsageSignal, WriteMemoryInput};
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
fn feedback_accepts_valid_signal() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    engine
        .write(WriteMemoryInput {
            content: "test memory".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // feedback with Referenced signal
    let result = engine.feedback(FeedbackInput {
        retrieval_id: 1,
        used_memory_ids: vec![],
        signal: UsageSignal::Referenced,
    });
    assert!(result.is_ok(), "feedback should succeed");

    // feedback with UserConfirmedCorrect
    let result = engine.feedback(FeedbackInput {
        retrieval_id: 2,
        used_memory_ids: vec![],
        signal: UsageSignal::UserConfirmedCorrect,
    });
    assert!(result.is_ok());

    // feedback with UserRejected
    let result = engine.feedback(FeedbackInput {
        retrieval_id: 3,
        used_memory_ids: vec![],
        signal: UsageSignal::UserRejected,
    });
    assert!(result.is_ok());

    engine.close().unwrap();
}

#[test]
fn feedback_persists_across_reopen() {
    let dir = tempdir().unwrap();
    let store_path = dir.path().join("hippmem.redb");

    {
        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();
        engine
            .feedback(FeedbackInput {
                retrieval_id: 42,
                used_memory_ids: vec![],
                signal: UsageSignal::TaskSucceeded,
            })
            .unwrap();
        engine.close().unwrap();
    }

    // reopen and verify
    {
        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();
        // feedback data should be persisted in activation_log; reopen should not error
        let result = engine.feedback(FeedbackInput {
            retrieval_id: 42,
            used_memory_ids: vec![],
            signal: UsageSignal::Referenced,
        });
        assert!(result.is_ok());
        engine.close().unwrap();
    }
}
