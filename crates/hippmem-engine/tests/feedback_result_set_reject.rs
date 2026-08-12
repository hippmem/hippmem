//! Result-set reject tests (0.4.0, D-B / B4 §4.2).
//!
//! An empty used_memory_ids + UserRejected means "the whole result set was
//! wrong" (trap questions, noisy stores). Under the 0.4.0 usage semantics:
//! - usage_score is still lowered by 0.05 per result-set memory (record field);
//! - the observable retrieval effect is RecentActivation suppression: result-set
//!   memories are excluded from recency boosts, so their prior confirmations
//!   no longer lift them via the recent channel.

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
fn empty_reject_lowers_usage_score_as_a_record() {
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

    // Record semantics (B4): usage_score still drops by 0.05 per memory.
    for id in &result_ids {
        assert_eq!(
            usage_of(&engine, *id),
            0.45,
            "result-set reject must lower each returned memory's usage score by 0.05 (record)"
        );
    }
}

/// The observable retrieval effect of a result-set reject: it cancels the
/// recency boost of those memories. Confirm first (they get boosted via the
/// recent channel), then result-set-reject, then the boost must be gone.
///
/// Confirms the third-ranked memory, not the top one: the top memory is
/// already the RRF maximum, so its normalized seed energy (and thus its
/// score) does not move when recency adds to it.
#[test]
fn result_set_reject_removes_recency_boost() {
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

    // Result-set reject of the same query → the boost must disappear.
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
    assert!(
        after_score < confirmed_score,
        "result-set reject must remove the recency boost, \
         confirmed={confirmed_score}, after-reject={after_score}"
    );
}
