//! Reverse-Hebbian tests (0.4.0, B4 §4.1).
//!
//! A targeted reject (used_memory_ids = [m], UserRejected) weakens the
//! association edges of m during the next consolidation cycle — the
//! graph-level counterpart of the removed global usage penalty. The memory
//! stays retrievable but is harder to reach via spreading.
//!
//! Contract: after reject + consolidate, every out-edge of the rejected
//! memory has strength lowered by `learning_rate` (0.08), floored at 0.12.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Timestamp;
use hippmem_engine::{
    ConsolidationScope, Engine, EngineConfig, FeedbackInput, InspectQuery, InspectReport,
    RetrieveContext, RetrieveInput, UsageSignal, WriteMemoryInput,
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

fn out_strengths(engine: &Engine, id: hippmem_core::ids::MemoryId) -> Vec<f32> {
    match engine.inspect(InspectQuery::Memory(id)) {
        Ok(InspectReport::Memory(m)) => m.out_edges.iter().map(|e| e.strength).collect(),
        _ => panic!("inspect failed for memory {id:?}"),
    }
}

#[test]
fn targeted_reject_weakens_rejected_memory_edges() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Shared-entity memories: the write path builds EntityOverlap edges.
    let contents = [
        "Xiaoming and Lihua are high school classmates in Beijing",
        "Xiaoming is a computer science student at Peking University",
        "Lihua works as a Java engineer at Alibaba Cloud",
        "Wangfang is Xiaoming's senior and studies AI models",
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

    // Pick the memory to reject: it must appear in the result set.
    let out = engine
        .retrieve(RetrieveInput {
            query: "What is the relationship between Xiaoming and Lihua?".to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    let rejected = out.results[1].memory.id;
    let before = out_strengths(&engine, rejected);
    assert!(
        !before.is_empty(),
        "rejected memory must have out-edges (shared entities)"
    );

    // Targeted reject, then one consolidation cycle applies the weakening.
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![rejected],
            signal: UsageSignal::UserRejected,
        })
        .unwrap();
    engine
        .consolidate(ConsolidationScope::Incremental)
        .expect("consolidate should succeed");

    let after = out_strengths(&engine, rejected);
    assert_eq!(
        before.len(),
        after.len(),
        "reject must weaken edges, not delete them"
    );
    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            a < b,
            "each out-edge of the rejected memory must be weaker after reject+consolidate \
             ({b} → {a})"
        );
        assert!(
            *a >= 0.12 - 1e-6,
            "weakening must floor at min_retained_strength (0.12), got {a}"
        );
    }
}
