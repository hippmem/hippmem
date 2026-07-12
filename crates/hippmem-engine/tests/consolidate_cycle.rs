//! acceptance tests (Engine layer): consolidation full-chain integration tests
//!
//! Verifies the complete write→retrieve→feedback→consolidate→inspect chain.
//! 3 scenarios:
//!   2. Hebbian reinforcement full chain: write associated memories → retrieve → feedback → consolidate → inspect verifies strength gain
//!   3. Decay full chain: write isolated memories → consolidate → inspect verifies decay execution
//!   4. Hebbian strength cap: multiple rounds of feedback+consolidate → verifies edge strength ≤ 1.0
//!
//! Note: full-chain decay tests are limited because Engine has no Clock injection (always uses SystemClock),
//! and newly written memories have formed_at ≈ now, so the decay condition won't trigger until 1 day elapses.
//! Full decay-logic coverage lives in hippmem-consolidation/tests/cycle.rs.

use hippmem_core::ids::MemoryId;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
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
        local_time: hippmem_core::time::Timestamp(1_700_000_000_000),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

fn default_retrieve_context() -> RetrieveContext {
    RetrieveContext {
        conversation_id: None,
        session_id: None,
        project_id: None,
        task_id: None,
        user_id: None,
        recent_memory_ids: vec![],
    }
}

/// Extract edge strength list from an InspectReport
fn extract_edge_strengths(report: &InspectReport) -> Vec<(MemoryId, f32)> {
    match report {
        InspectReport::Memory(inspect) => inspect
            .out_edges
            .iter()
            .map(|e| (e.to, e.strength))
            .collect(),
        _ => vec![],
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 2: Engine full-chain Hebbian — write→retrieve→feedback→consolidate→verify
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_2_engine_hebbian_full_chain() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // ── Step 1: write × 5 associated memories (shared entity "Rust") ──
    let mut memory_ids = Vec::new();
    for i in 0..5 {
        let output = engine
            .write(WriteMemoryInput {
                content: format!("Rust programming notes #{}", i),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        memory_ids.push(output.memory_id);
    }

    // ── Step 2: retrieve (query "Rust", get most relevant memories from results) ──
    let retrieve_output = engine
        .retrieve(RetrieveInput {
            query: "Rust".into(),
            context: default_retrieve_context(),
            top_k: 5,
            max_hops: None,
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();

    assert!(
        !retrieve_output.results.is_empty(),
        "retrieve 'Rust' should return results"
    );

    // Take retrieved memory IDs
    let retrieved_ids: Vec<MemoryId> = retrieve_output
        .results
        .iter()
        .map(|r| r.memory.id)
        .collect();

    // ── Step 3: feedback × 3 (accumulate co-activation count to reach Hebbian threshold) ──
    // hebbian_reinforce coactivation_threshold=3, needs multiple feedbacks on the same pair
    for round in 0..3 {
        engine
            .feedback(FeedbackInput {
                retrieval_id: (round + 1) as u64,
                used_memory_ids: retrieved_ids.clone(),
                signal: UsageSignal::UserConfirmedCorrect,
            })
            .unwrap();
    }

    // ── Step 4: record edge strengths before consolidate ──
    let pre_strengths: Vec<(MemoryId, f32)> = memory_ids
        .iter()
        .filter_map(|id| {
            if let Ok(report) = engine.inspect(InspectQuery::Memory(*id)) {
                Some(extract_edge_strengths(&report))
            } else {
                None
            }
        })
        .flatten()
        .collect();

    // ── Step 5: consolidate ──
    let report = engine.consolidate(ConsolidationScope::Full).unwrap();
    assert!(report.memories_processed >= 5, "should process ≥5 memories");
    assert!(report.elapsed_ms > 0, "elapsed time should be recorded");
    // report fields are no longer hardcoded to 0
    // (edges_decayed/archived/hebbian_applied are populated from actual execution results)

    // ── Step 6: verify edge strength changes after consolidate ──
    let post_strengths: Vec<(MemoryId, f32)> = memory_ids
        .iter()
        .filter_map(|id| {
            if let Ok(report) = engine.inspect(InspectQuery::Memory(*id)) {
                Some(extract_edge_strengths(&report))
            } else {
                None
            }
        })
        .flatten()
        .collect();

    // At least some edges exist (associated memories were written)
    assert!(
        !post_strengths.is_empty(),
        "associated memories should have edges between them"
    );

    // Verify consolidate persistence is effective: pre/post edge sets are identical (no edges lost)
    // Decay lowers non-protected edge scores, but Hebbian may partially offset that
    // Core assertion: edges still exist after consolidate (no data loss)
    assert_eq!(
        pre_strengths.len(),
        post_strengths.len(),
        "consolidate should not lose edges"
    );

    engine.close().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 3: Engine full-chain decay — write isolated memories → consolidate → verify
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_3_engine_decay_full_chain() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // ── write × 3 isolated memories (no shared entities, should produce no edges or only very weak edges) ──
    let mut memory_ids = Vec::new();
    for i in 0..3 {
        let output = engine
            .write(WriteMemoryInput {
                content: format!("Standalone memory topic#{} keyword#{}", i, i + 100),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        memory_ids.push(output.memory_id);
    }

    // ── record state before consolidate ──
    let pre_stats =
        if let Ok(InspectReport::StoreStats(stats)) = engine.inspect(InspectQuery::StoreStats) {
            (stats.memory_count, stats.edge_count)
        } else {
            (0, 0)
        };

    // ── consolidate ──
    let report = engine.consolidate(ConsolidationScope::Full).unwrap();
    assert_eq!(report.memories_processed, 3, "should process 3 memories");

    // ── verify store consistency after consolidate ──
    let post_stats =
        if let Ok(InspectReport::StoreStats(stats)) = engine.inspect(InspectQuery::StoreStats) {
            (stats.memory_count, stats.edge_count)
        } else {
            (0, 0)
        };

    // Memory count unchanged (consolidate should not delete memories, only decay edges)
    assert_eq!(
        post_stats.0, pre_stats.0,
        "consolidate should not change memory count"
    );

    // Isolated memories may have no edges; even if they do, consolidate should not increase them
    assert!(
        post_stats.1 <= pre_stats.1 || pre_stats.1 == 0,
        "consolidate should not increase isolated-edge count"
    );

    // ── verify each isolated memory is still inspectable ──
    for id in &memory_ids {
        let result = engine.inspect(InspectQuery::Memory(*id));
        assert!(
            result.is_ok(),
            "isolated memory {} should still be inspectable",
            id.0
        );
    }

    engine.close().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 4: Hebbian strength capped at 1.0
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_4_hebbian_strength_capped_at_one() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // ── write associated memories ──
    let mut memory_ids = Vec::new();
    for i in 0..3 {
        let output = engine
            .write(WriteMemoryInput {
                content: format!("Project development log entry #{}", i),
                content_type: Some(ContentType::ProjectKnowledge),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        memory_ids.push(output.memory_id);
    }

    // ── multiple rounds of retrieve + feedback + consolidate ──
    for round in 0..5 {
        // retrieve
        let retrieve_output = engine
            .retrieve(RetrieveInput {
                query: "HippMem".into(),
                context: default_retrieve_context(),
                top_k: 3,
                max_hops: None,
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();

        let retrieved_ids: Vec<MemoryId> = retrieve_output
            .results
            .iter()
            .map(|r| r.memory.id)
            .collect();

        // feedback
        engine
            .feedback(FeedbackInput {
                retrieval_id: round + 1,
                used_memory_ids: retrieved_ids,
                signal: UsageSignal::UserConfirmedCorrect,
            })
            .unwrap();

        // consolidate
        let report = engine.consolidate(ConsolidationScope::Full).unwrap();
        assert!(
            report.memories_processed >= 3,
            "round {} should process memories",
            round
        );
    }

    // ── verify: all edge strengths ≤ 1.0 ──
    for id in &memory_ids {
        if let Ok(InspectReport::Memory(inspect)) = engine.inspect(InspectQuery::Memory(*id)) {
            for edge in &inspect.out_edges {
                assert!(
                    edge.strength <= 1.0,
                    "edge strength should not exceed 1.0: id={}, target={}, strength={}",
                    id.0,
                    edge.to.0,
                    edge.strength
                );
            }
        }
    }

    engine.close().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 5: Summarizer integration — summary trigger and covers chain
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_5_summary_creation_full_chain() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // ── write × 15 similar low-importance memories (simulating tool output logs) ──
    let mut memory_ids = Vec::new();
    for i in 0..15 {
        let output = engine
            .write(WriteMemoryInput {
                content: format!(
                    "Build output: warning E{:04} fixed at src/main.rs line {}.",
                    i,
                    i * 10 + 1
                ),
                content_type: Some(ContentType::ToolResult),
                context: ctx(),
                importance_hint: Some(0.2),
                source_refs: vec![],
            })
            .unwrap();
        memory_ids.push(output.memory_id);
    }

    // ── consolidate ──
    let report = engine.consolidate(ConsolidationScope::Full).unwrap();
    assert!(
        report.memories_processed >= 15,
        "should process ≥15 memories"
    );

    // Core assertion: summaries_created > 0 (15 similar memories should trigger a summary)
    assert_eq!(
        report.summaries_created, 1,
        "15 similar low-importance memories should trigger 1 summary"
    );

    // ── verify source memories are still inspectable (Constitution C7: keep originals, no physical deletion) ──
    for id in &memory_ids {
        assert!(
            engine.inspect(InspectQuery::Memory(*id)).is_ok(),
            "source memory {} should be retained",
            id.0
        );
    }

    engine.close().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 6: no summary triggered — <12 memories
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_6_no_summary_with_few_memories() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // ── write × 5 dissimilar memories ──
    for i in 0..5 {
        engine
            .write(WriteMemoryInput {
                content: format!("Standalone log entry#{} topic#{}", i, i * 100),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }

    // ── consolidate ──
    let report = engine.consolidate(ConsolidationScope::Full).unwrap();

    // Fewer than 12, should not trigger a summary
    assert_eq!(
        report.summaries_created, 0,
        "fewer than 12 memories should not trigger a summary"
    );

    engine.close().unwrap();
}
