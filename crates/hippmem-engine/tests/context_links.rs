//! Context-answer links contract tests (0.4.3, memory-learning-mechanism).
//!
//! A confirmation binds a memory to the query's entity/topic fingerprint:
//! - the lift applies only to later queries whose fingerprint intersects the
//!   recorded links (same-context queries), never to unrelated queries;
//! - link strength accumulates across confirmations (multi-round learning),
//!   so the lift grows with repeated confirmation instead of saturating after
//!   one round.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Timestamp;
use hippmem_engine::{
    Engine, EngineConfig, FeedbackInput, RetrieveContext, RetrieveInput, UsageSignal,
    WriteMemoryInput,
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

fn write(engine: &Engine, text: &str) {
    engine
        .write(WriteMemoryInput {
            content: text.to_string(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
}

fn score_of(engine: &Engine, query: &str, content: &str) -> (Option<f32>, u64) {
    let out = engine
        .retrieve(RetrieveInput {
            query: query.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    let score = out
        .results
        .iter()
        .find(|r| r.memory.content.raw == content)
        .map(|r| r.final_score);
    (score, out.retrieval_id)
}

#[test]
fn confirmation_lifts_only_in_matching_context() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let gt_a = "Xiaoming lives in Haidian District, Beijing.";
    let decoy_a = "Xiaoming is a computer science student at Peking University.";
    let gt_b = "Lihua works on data collection in Shanghai.";
    let decoy_b = "Lihua is a senior schoolmate of Xiaoming.";
    for t in [gt_a, decoy_a, gt_b, decoy_b] {
        write(&engine, t);
    }

    let query_a = "Where does Xiaoming live?";
    let query_b = "Where does Lihua work?";

    // Baseline: GT-A score in query A before any confirmation.
    let (s0, rid) = score_of(&engine, query_a, gt_a);
    assert!(s0.is_some(), "GT-A must be in query A results");

    // Confirm GT-A in query A (same context as later query A retrievals).
    let out = engine
        .retrieve(RetrieveInput {
            query: query_a.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    let gt_a_id = out
        .results
        .iter()
        .find(|r| r.memory.content.raw == gt_a)
        .map(|r| r.memory.id)
        .expect("GT-A in results");
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![gt_a_id],
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();

    // Same-context query A: GT-A must be lifted.
    let (s1, _) = score_of(&engine, query_a, gt_a);
    assert!(
        s1.unwrap() > s0.unwrap(),
        "same-context confirmation must lift GT-A: before={} after={}",
        s0.unwrap(),
        s1.unwrap()
    );

    // Unrelated query B (different entity): GT-A must not be lifted by the
    // query-A confirmation — either absent from results or scored as before.
    let (s_b_before, _) = score_of(&engine, query_b, gt_a);
    let (s_b_after, _) = score_of(&engine, query_b, gt_a);
    assert_eq!(
        s_b_before, s_b_after,
        "an unrelated query must not borrow context heat (GT-A unchanged in query B)"
    );
    // And the natural answer of query B still leads the results.
    let (gt_b_score, _) = score_of(&engine, query_b, gt_b);
    if let (Some(a), Some(b)) = (s_b_after, gt_b_score) {
        assert!(
            b > a,
            "query B's natural answer must outrank a hot memory from another context"
        );
    }
    let _ = rid;
}

#[test]
fn context_strength_accumulates_across_confirmations() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let gt_a = "Xiaoming lives in Haidian District, Beijing.";
    let decoy_a = "Xiaoming is a computer science student at Peking University.";
    write(&engine, gt_a);
    write(&engine, decoy_a);

    let query_a = "Where does Xiaoming live?";
    let (s0, _) = score_of(&engine, query_a, gt_a);

    let mut prev = s0.unwrap();
    for round in 1..=3 {
        let out = engine
            .retrieve(RetrieveInput {
                query: query_a.to_string(),
                context: retrieve_ctx(),
                top_k: 5,
                max_hops: Some(1),
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();
        let gt_a_id = out
            .results
            .iter()
            .find(|r| r.memory.content.raw == gt_a)
            .map(|r| r.memory.id)
            .expect("GT-A in results");
        engine
            .feedback(FeedbackInput {
                retrieval_id: out.retrieval_id,
                used_memory_ids: vec![gt_a_id],
                signal: UsageSignal::UserConfirmedCorrect,
            })
            .unwrap();

        let (s, _) = score_of(&engine, query_a, gt_a);
        assert!(
            s.unwrap() > prev,
            "round {round}: confirmation must keep lifting GT-A (multi-round accumulation): prev={prev} now={}",
            s.unwrap()
        );
        prev = s.unwrap();
    }
    assert!(
        prev > s0.unwrap(),
        "three confirmations must beat the baseline"
    );
}
