//! Entity coverage boost contract tests (0.4.2, entity-coverage-query-boost).
//!
//! For multi-entity queries ("what is the relationship between X and Y") the
//! answer must involve both entities, yet the plain per-entity channel scores
//! gave a single-entity memory (word-surface pseudo-relevance: it shares one
//! entity word with the query) the same entity-channel standing as the
//! full-coverage ground truth. 2026-08-12 C-scenario round 5 measured exactly
//! that: the GT "Xiaoming and Lihua are high school classmates" stayed at rank
//! 3 behind "Xiaoming is a computer science student at Peking University"
//! (1/2 coverage) and an entirely unrelated memory (0/2 coverage) — and five
//! rounds of confirmations could not lift it, because feedback is a
//! candidate-set tie-break, not a relevance signal.
//!
//! The fix has two tiers:
//! - seed tier (retrieve_api.rs 2a): a memory covering k query entities gets
//!   a higher entity-channel hit score (0.2 / 0.35 / 0.5 by k);
//! - rerank tier (retrieve_api.rs 7d2): candidates covering k of the query's
//!   N ≥ 2 entities are multiplied by (1 + 0.2·k/N).
//! Both tiers only lift memories that already reached the candidate set —
//! coverage is a query-side relevance signal, never a global gain.

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

fn run_retrieval(engine: &Engine, query: &str) -> Vec<(String, f32)> {
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
        .map(|r| (r.memory.content.raw.clone(), r.final_score))
        .collect()
}

/// C-scenario-equivalent store (English, P7-compliant): the ground truth
/// covers both query entities but has weak surface overlap with the query;
/// the pseudo-relevant decoy covers one entity and carries higher importance
/// (mirrors the 2026-08-12 C5 measurement where importance 0.3 vs ~0 widened
/// the gap to 1.62×).
fn build_store() -> (tempfile::TempDir, Engine) {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let contents: &[(&str, Option<f32>)] = &[
        // GT: full coverage (2/2), weak surface match
        ("Xiaoming and Lihua are high school classmates.", None),
        // Decoy: 1/2 coverage, high importance, strong surface match
        (
            "Xiaoming is a computer science student at Peking University.",
            Some(0.3),
        ),
        // Unrelated: 0/2 coverage
        ("Chenhao trains deep learning models with PyTorch.", None),
        // Related but not the answer: 2/2 coverage
        (
            "In Xiaoming's project group, Lihua handles data collection.",
            None,
        ),
        // Edge: 1/2 coverage
        ("Wangfang is Xiaoming's senior schoolmate.", None),
    ];
    for (c, importance) in contents {
        engine
            .write(WriteMemoryInput {
                content: c.to_string(),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: *importance,
                source_refs: vec![],
            })
            .unwrap();
    }
    (dir, engine)
}

#[test]
fn multi_entity_query_lifts_full_coverage_to_top() {
    let (_dir, engine) = build_store();
    let results = run_retrieval(
        &engine,
        "What is the relationship between Xiaoming and Lihua?",
    );

    let gt_text = "Xiaoming and Lihua are high school classmates.";
    let decoy_text = "Xiaoming is a computer science student at Peking University.";
    let unrelated_text = "Chenhao trains deep learning models with PyTorch.";

    assert!(!results.is_empty(), "retrieval must return results");
    let gt_rank = results
        .iter()
        .position(|(t, _)| t == gt_text)
        .unwrap_or_else(|| panic!("GT must be in the results: {results:?}"));
    let decoy_rank = results
        .iter()
        .position(|(t, _)| t == decoy_text)
        .unwrap_or_else(|| panic!("decoy must be in the results: {results:?}"));
    let unrelated_rank = results
        .iter()
        .position(|(t, _)| t == unrelated_text)
        .expect("unrelated memory must stay in the results");

    // The full-coverage GT must overtake the 1/2-coverage decoy, and the
    // 0/2-coverage memory must stay below it (it gets no lift).
    assert!(
        gt_rank < decoy_rank,
        "full-coverage GT must rank above the 1/2-coverage decoy; GT={gt_rank} decoy={decoy_rank}: {results:?}"
    );
    assert!(
        gt_rank < unrelated_rank,
        "0/2-coverage memory must stay below the GT; GT={gt_rank} unrelated={unrelated_rank}: {results:?}"
    );
}

#[test]
fn single_entity_query_is_untouched() {
    let (_dir, engine) = build_store();
    // N = 1: the coverage correction is disabled by construction
    // (query_entity_count >= 2 gate); this test guards the recall path.
    let results = run_retrieval(&engine, "What does Xiaoming study?");

    let decoy_text = "Xiaoming is a computer science student at Peking University.";
    assert!(
        results.iter().any(|(t, _)| t == decoy_text),
        "single-entity query must still surface the decoy: {results:?}"
    );
}

#[test]
fn multi_entity_query_is_deterministic() {
    let (_dir, engine) = build_store();
    let query = "What is the relationship between Xiaoming and Lihua?";
    let first = run_retrieval(&engine, query);
    let second = run_retrieval(&engine, query);
    assert_eq!(
        first, second,
        "same store state + same query must produce bit-identical results"
    );
}
