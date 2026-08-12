//! Result-set reject contract tests (0.4.1).
//!
//! An empty used_memory_ids + UserRejected is a *retrieval-quality* signal:
//! "this retrieval returned no correct answer" (trap questions, noisy stores).
//! Under the 0.4.1 semantics it is a no-op on memories:
//! - usage_score is NOT lowered (the 0.4.0 D-B -0.05 was removed);
//! - the recent channel is NOT suppressed (the 0.4.0 B4 §4.2 retain was removed).
//!
//! Why removed: trap questions trigger a result-set reject by construction
//! (the store has no answer, retrieval must still return a list), so the
//! 0.4.0 behavior permanently suppressed innocent memories that merely
//! appeared in the rejected result set — even after explicit confirmation
//! (2026-08-12 test report O1). Targeted rejects (non-empty used_memory_ids)
//! are unaffected: they still weaken the named memories (see
//! feedback_reject_weaken.rs).

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Timestamp;
use hippmem_engine::{
    Engine, EngineConfig, FeedbackInput, InspectQuery, InspectReport, RetrieveContext,
    RetrieveInput, UsageSignal, WriteMemoryInput,
};
use tempfile::tempdir;

fn ctx() -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: Timestamp(1_700_000_000_000),
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

fn usage_of(engine: &Engine, id: hippmem_core::ids::MemoryId) -> f32 {
    match engine.inspect(InspectQuery::Memory(id)) {
        Ok(InspectReport::Memory(m)) => m.unit.activation.usage_score.value(),
        _ => panic!("inspect failed for memory {id:?}"),
    }
}

fn run_retrieval(engine: &Engine, query: &str) -> Vec<(hippmem_core::ids::MemoryId, f32)> {
    engine
        .retrieve(RetrieveInput {
            query: query.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap()
        .results
        .iter()
        .map(|r| (r.memory.id, r.final_score))
        .collect()
}

fn build_store() -> (tempfile::TempDir, Engine) {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();
    let contents = [
        "Zhaoqiang handles deployment, Zhouyu handles monitoring and alerts",
        "Zhouyu develops large-model applications at Alibaba Cloud",
        "Wangfang evaluates model quality, Xiaoming handles system rollout",
    ];
    for c in contents {
        engine
            .write(WriteMemoryInput {
                content: c.to_string(),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: Some(0.5),
                source_refs: vec![],
            })
            .unwrap();
    }
    (dir, engine)
}

#[test]
fn empty_reject_has_no_usage_score_side_effect() {
    let (_dir, engine) = build_store();

    let out = engine
        .retrieve(RetrieveInput {
            query: "What work does Zhouyu do?".to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    let result_ids: Vec<_> = out.results.iter().map(|r| r.memory.id).collect();
    assert!(!result_ids.is_empty(), "query must return results");
    for id in &result_ids {
        assert_eq!(usage_of(&engine, *id), 0.5, "fresh memory starts neutral");
    }

    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![],
            signal: UsageSignal::UserRejected,
        })
        .unwrap();

    // 0.4.1: result-set reject is a retrieval-quality signal with no memory
    // side effects — usage_score stays neutral (0.4.0 lowered it by 0.05).
    for id in &result_ids {
        assert_eq!(
            usage_of(&engine, *id),
            0.5,
            "result-set reject must not lower usage_score (0.4.1 contract)"
        );
    }
}

/// A result-set reject must not undo the recency boost of a confirmed memory.
/// Confirm a memory (it gets boosted via the recent channel), then result-set
/// reject a later retrieval of the same query — the boost must survive.
/// (0.4.0 removed the boost via `recent_map.retain`; that was the O1 bug:
/// a trap question permanently suppressed a memory the user then confirmed.)
#[test]
fn result_set_reject_keeps_confirmation_boost() {
    let (_dir, engine) = build_store();
    let q = "What work does Zhouyu do?";

    // Baseline.
    let before = run_retrieval(&engine, q);
    let id = before[2].0;
    let before_score = before[2].1;
    assert!(before_score > 0.0, "third result must have a real score");

    // Confirm the third result → it is boosted by the recent channel.
    let out = engine
        .retrieve(RetrieveInput {
            query: q.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![id],
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();
    let confirmed = run_retrieval(&engine, q);
    let confirmed_score = confirmed
        .iter()
        .find(|(mid, _)| *mid == id)
        .map(|(_, s)| *s)
        .unwrap();
    assert!(
        confirmed_score > before_score,
        "confirmation must boost the memory (recency channel), \
         before={before_score}, after={confirmed_score}"
    );

    // Result-set reject of the same query → the boost must survive.
    let out = engine
        .retrieve(RetrieveInput {
            query: q.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![],
            signal: UsageSignal::UserRejected,
        })
        .unwrap();
    let after_reject = run_retrieval(&engine, q);
    let after_score = after_reject
        .iter()
        .find(|(mid, _)| *mid == id)
        .map(|(_, s)| *s)
        .unwrap();
    assert_eq!(
        after_score, confirmed_score,
        "result-set reject must not remove the confirmation boost, \
         confirmed={confirmed_score}, after-reject={after_score}"
    );
}
