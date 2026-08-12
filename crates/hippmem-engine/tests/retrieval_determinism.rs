//! Determinism contract tests (0.3.1, F1 from the 2026-08-11 test report).
//!
//! Contract: **the same library state + same query + same parameters must produce
//! bit-identical retrieval scores**. Verified by building two identical stores
//! (same writes, same order) and retrieving once from each — every HashMap in the
//! retrieval path gets a fresh RandomState per call, so any iteration-order
//! dependence (non-associative multi-path energy merges, tied ranks) surfaces as
//! a score difference between the two stores.
//!
//! Since 0.4.0/B2, repeated retrieval in the SAME store is also bit-identical:
//! retrieval still appends to the activation log (audit trail), but "retrieve"
//! is no longer a positive signal, so the RecentActivation channel no longer
//! sees its own history as candidates. The full contract — same state → same
//! output, including repeated calls — is tested below.
//!
//! Boundary: the RecentActivation channel breaks ties by MemoryId. Within one
//! store the ids are fixed, so rankings are reproducible; across two stores
//! with different generated ids, tied entries may rank differently. That is
//! why the cross-store tests compare score vectors only.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Timestamp;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};
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

/// Builds a store whose memories share entities (EntityOverlap edges) so that a
/// query reaches multiple memories through both direct channels and graph paths.
fn build_shared_entity_store() -> (tempfile::TempDir, Engine) {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let contents = [
        "Xiaoming and Lihua are high school classmates in Beijing",
        "Xiaoming is a computer science student at Peking University",
        "Lihua works as a Java engineer at Alibaba Cloud",
        "Wangfang is Xiaoming's senior and studies AI models",
        "Xiaoming's team uses Rust for backend services",
        "Lihua's team deploys monitoring systems with Python",
        "Wangfang evaluates model quality for the AI platform",
        "The team collaborates closely on the shared project",
        "Xiaoming lives in Haidian district of Beijing",
        "Lihua and Wangfang cooperate on data pipelines",
    ];
    for (i, c) in contents.iter().enumerate() {
        engine
            .write(WriteMemoryInput {
                content: c.to_string(),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: Some(0.4 + (i % 3) as f32 * 0.2),
                source_refs: vec![],
            })
            .unwrap();
    }
    (dir, engine)
}

fn run_retrieval(engine: &Engine, query: &str, top_k: usize, max_hops: Option<usize>) -> Vec<f32> {
    engine
        .retrieve(RetrieveInput {
            query: query.to_string(),
            context: retrieve_ctx(),
            top_k,
            max_hops,
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap()
        .results
        .iter()
        .map(|r| r.final_score)
        .collect()
}

/// Contract: two identical stores (same writes, same order) produce identical
/// scores for the same query — single-hop case (the path that showed rank3/rank5
/// score drift in the test report).
///
/// Each query gets a fresh store pair and is the FIRST retrieval on those
/// stores: afterwards the activation log is no longer empty, and the
/// RecentActivation channel breaks ties by MemoryId, which differs across
/// stores. Empty activation log keeps the comparison exact.
#[test]
fn identical_stores_produce_identical_scores() {
    for query in [
        "What is the relationship between Xiaoming and Lihua?",
        "Where does Xiaoming live?",
        "Who evaluates model quality?",
    ] {
        let (_dir_a, engine_a) = build_shared_entity_store();
        let (_dir_b, engine_b) = build_shared_entity_store();
        let a = run_retrieval(&engine_a, query, 5, Some(1));
        let b = run_retrieval(&engine_b, query, 5, Some(1));
        assert_eq!(
            a, b,
            "identical stores must produce bit-identical scores (query: {query})"
        );
    }
}

/// Contract: multi-hop traversal (where non-associative multi-path merges occur)
/// is also reproducible across identical stores.
#[test]
fn identical_stores_multi_hop_produce_identical_scores() {
    let (_dir_a, engine_a) = build_shared_entity_store();
    let (_dir_b, engine_b) = build_shared_entity_store();

    let a = run_retrieval(
        &engine_a,
        "What is the relationship between Xiaoming and Lihua?",
        8,
        Some(2),
    );
    let b = run_retrieval(
        &engine_b,
        "What is the relationship between Xiaoming and Lihua?",
        8,
        Some(2),
    );
    assert_eq!(a, b, "multi-hop retrieval must be bit-identical");
}

/// Contract (0.4.0/B2): repeated retrieval on the SAME store is bit-identical.
/// This is the exact scenario from the 2026-08-11 report (three consecutive
/// calls, no other operations in between). It only holds after "retrieve"
/// stopped being a positive signal — before that, each call seeded the
/// RecentActivation channel with the previous call's results.
#[test]
fn repeated_retrieval_same_store_is_bit_identical() {
    let (_dir, engine) = build_shared_entity_store();

    for query in [
        "What is the relationship between Xiaoming and Lihua?",
        "Where does Xiaoming live?",
    ] {
        let a = run_retrieval(&engine, query, 5, Some(1));
        let b = run_retrieval(&engine, query, 5, Some(1));
        let c = run_retrieval(&engine, query, 5, Some(1));
        assert_eq!(
            a, b,
            "repeated retrieval must be bit-identical (query: {query})"
        );
        assert_eq!(
            b, c,
            "repeated retrieval must be bit-identical across calls (query: {query})"
        );
    }
}
