//! Retrieval path reinforcement contract tests (0.4.3, memory-learning-mechanism).
//!
//! A seed answer (directly hit, no propagation path recorded) must never
//! touch any edge — the graph stays exactly as it was before the
//! confirmation. A real propagation scenario cannot be constructed in a
//! small test store (the semantic channel's top-k has no similarity floor,
//! so every memory becomes a seed) — real propagation is covered
//! end-to-end by the batch2 scenario in P4.

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

fn write(engine: &Engine, text: &str) -> hippmem_core::ids::MemoryId {
    engine
        .write(WriteMemoryInput {
            content: text.to_string(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap()
        .memory_id
}

fn edge_strength(
    engine: &Engine,
    from: hippmem_core::ids::MemoryId,
    to: hippmem_core::ids::MemoryId,
) -> f32 {
    match engine.inspect(InspectQuery::Memory(from)) {
        Ok(InspectReport::Memory(m)) => m
            .unit
            .links
            .iter()
            .find(|l| l.target_id == to)
            .map(|l| l.strength.value())
            .unwrap_or(0.0),
        _ => panic!("inspect failed"),
    }
}

/// A seed answer (directly hit, no propagation path recorded) must not touch
/// any edge — the graph stays exactly as it was before the confirmation.
#[test]
fn seed_confirmation_touches_no_edge() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let guide_id = write(&engine, "Xiaoming manages the supply chain test at BYD.");
    let answer_id = write(
        &engine,
        "Xiaoming tracks quality metrics in weekly reviews.",
    );

    let out = engine
        .retrieve(RetrieveInput {
            query: "Who manages the BYD test?".to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();

    let before = edge_strength(&engine, guide_id, answer_id);
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![answer_id],
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();
    assert_eq!(
        edge_strength(&engine, guide_id, answer_id),
        before,
        "a seed confirmation must not touch edges"
    );
}
