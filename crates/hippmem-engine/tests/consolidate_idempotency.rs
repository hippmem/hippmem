//! Idempotency contract tests (0.3.1, F2 from the 2026-08-11 test report).
//!
//! Regression: every consolidation cycle unconditionally rebuilt all
//! co-activation edges (`unit.links.push` without a dedup check), so edge
//! counts accumulated across cycles (2079 edges in one round in the report,
//! 3272 over three). Contract: consolidation is idempotent — a second cycle
//! with the same activation log must not create any new edges.
//!
//! Note: `edges_merged` counts Hebbian reinforcement of EXISTING edges too
//! (strength gains, `activation_count` increments), so it is legitimately
//! non-zero on the second cycle. The edge COUNT is the contract, checked via
//! InspectReport.

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

/// Total outgoing edge count across all memories in the store.
fn total_edge_count(engine: &Engine) -> u64 {
    match engine.inspect(InspectQuery::StoreStats) {
        Ok(InspectReport::StoreStats(stats)) => stats.edge_count,
        _ => panic!("StoreStats inspect should succeed"),
    }
}

#[test]
fn second_consolidate_cycle_creates_no_new_edges() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Shared-entity memories: co-activation edges get built between them.
    let contents = [
        "Xiaoming and Lihua are high school classmates in Beijing",
        "Xiaoming is a computer science student at Peking University",
        "Lihua works as a Java engineer at Alibaba Cloud",
        "Wangfang is Xiaoming's senior and studies AI models",
        "Xiaoming's team uses Rust for backend services",
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

    // One retrieval + confirmation → activation log with positive signals.
    let out = engine
        .retrieve(RetrieveInput {
            query: "What is the relationship between Xiaoming and Lihua?".to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    let used: Vec<_> = out.results.iter().take(2).map(|r| r.memory.id).collect();
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: used,
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();

    // Cycle 1: builds co-activation edges.
    let r1 = engine.consolidate(ConsolidationScope::Incremental).unwrap();
    let edges_after_1 = total_edge_count(&engine);

    // Cycle 2: same activation log, nothing new to add.
    let r2 = engine.consolidate(ConsolidationScope::Incremental).unwrap();
    let edges_after_2 = total_edge_count(&engine);

    assert!(
        edges_after_2 == edges_after_1,
        "consolidation must be idempotent: edges after cycle 2 ({edges_after_2}) must equal \
         after cycle 1 ({edges_after_1}) — duplicate co-activation edges accumulate otherwise"
    );
    assert!(
        r2.edges_merged <= r1.edges_merged,
        "second cycle must not merge more edges than the first \
         (cycle1={}, cycle2={})",
        r1.edges_merged,
        r2.edges_merged
    );
}
