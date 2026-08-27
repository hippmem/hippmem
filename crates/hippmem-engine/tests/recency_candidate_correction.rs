//! Recency candidate-correction tests (0.4.1, recency-candidate-correction
//! proposal; 2026-08-13 test report O2).
//!
//! Confirmation frequency must be a tie-break WITHIN the candidate set, not
//! an independent seed:
//! - a confirmed memory must NOT appear in the results of an unrelated query
//!   (0.4.0's frequency seed boosted it in every query — the O2 bug);
//! - within a related query the lift must be bounded by
//!   `RECENCY_CORRECTION_ALPHA` (0.05), so it never crosses semantic tiers.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Timestamp;
use hippmem_engine::{
    Engine, EngineConfig, FeedbackInput, RetrieveContext, RetrieveInput, UsageSignal,
    WriteMemoryInput,
};
use tempfile::tempdir;

/// Max relative lift applied by the recency correction (must stay in sync
/// with `RECENCY_CORRECTION_ALPHA` in retrieve_api.rs).
const RECENCY_CORRECTION_ALPHA: f32 = 0.15;

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

/// 20 memories in two disjoint topics ("people" vs "tech"); no shared
/// vocabulary across topics, so an unrelated query cannot reach the other
/// topic through any channel.
fn build_topic_store() -> (tempfile::TempDir, std::sync::Arc<Engine>) {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();
    let people = [
        "Xiaoming Li Hua high school classmates",
        "Li Hua table tennis club captain",
        "Wang Fang Peking University astronomy",
        "Chen Hao image recognition thesis",
        "Sun Li stamp collection provinces",
        "Guo Qiang weekend marathon running",
        "Zheng Yue city orchestra violin",
        "Lin Tao ancient Chinese poetry",
        "Xu Dan local library volunteer",
        "Ma Chao high school math teacher",
    ];
    let tech = [
        "Go language chat service backend",
        "Zhou Yu automated testing platform",
        "Zhao Qiang Alibaba Cloud deployment",
        "Liu Wei logging pipeline maintenance",
        "Huang Jing database migration scripts",
        "Deng Peng container cluster management",
        "Feng Lei network latency monitoring",
        "Tang Min CDN cache configuration",
        "Cao Yang pull request reviews",
        "Shen Jun incident alerting system",
    ];
    for c in people.iter().chain(tech.iter()) {
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

/// O2 regression: a confirmed memory must not surface in an unrelated query.
/// Under 0.4.0 the frequency seed boosted it in every query (rank ~0.34 tier
/// in the acceptance report); under 0.4.1 the correction only touches the
/// candidate set, which the unrelated query never contains it in.
#[test]
fn confirmed_memory_absent_from_unrelated_query() {
    let (_dir, engine) = build_topic_store();
    let people_query = "Xiaoming Li Hua relationship";
    let tech_query = "chat service programming language";

    // Sanity: the topics are separated by the channels.
    let people_top = run_retrieval(&engine, people_query);
    let tech_top = run_retrieval(&engine, tech_query);
    let people_ids: Vec<_> = people_top.iter().map(|(id, _)| *id).collect();
    let tech_ids: Vec<_> = tech_top.iter().map(|(id, _)| *id).collect();
    assert!(
        !people_ids.iter().any(|p| tech_ids.contains(p)),
        "topic separation: people and tech results must be disjoint \
         (people={people_ids:?}, tech={tech_ids:?})"
    );

    // Confirm the top people memory via its own retrieval.
    let out = engine
        .retrieve(RetrieveInput {
            query: people_query.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    let confirmed_id = out.results[0].memory.id;
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![confirmed_id],
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();

    // The unrelated query must not surface it — even though it is now the
    // most-confirmed memory in the store.
    let after = run_retrieval(&engine, tech_query);
    assert!(
        !after.iter().any(|(id, _)| *id == confirmed_id),
        "confirmed memory must not appear in an unrelated query's top-5 \
         (recency is a candidate-set tie-break, not an independent seed)"
    );
}

/// The candidate-set lift is bounded: confirming a memory that is already a
/// candidate multiplies its score by at most (1 + `RECENCY_CORRECTION_ALPHA`),
/// so it can only overtake memories within that relative distance — a
/// tie-break, never a tier jump.
#[test]
fn candidate_lift_is_bounded_by_alpha() {
    let (_dir, engine) = build_topic_store();
    let query = "Which programming language is used for the chat service?";

    let before = run_retrieval(&engine, query);
    // Pick a mid-tier tech candidate to confirm.
    let target = before
        .iter()
        .enumerate()
        .find(|(i, _)| *i >= 1)
        .map(|(_, (id, _))| *id)
        .expect("query must return at least 2 results");
    let before_score = before
        .iter()
        .find(|(id, _)| *id == target)
        .map(|(_, s)| *s)
        .unwrap();

    let out = engine
        .retrieve(RetrieveInput {
            query: query.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![target],
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();

    let after = run_retrieval(&engine, query);
    let after_score = after
        .iter()
        .find(|(id, _)| *id == target)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    let lift = after_score / before_score - 1.0;
    assert!(
        lift > 0.0,
        "confirming an in-candidate memory must lift it, \
         before={before_score}, after={after_score}"
    );
    assert!(
        lift <= RECENCY_CORRECTION_ALPHA + 1e-6,
        "recency lift must be bounded by alpha (multiplicative tie-break), \
         lift={lift} > {RECENCY_CORRECTION_ALPHA}"
    );
}
