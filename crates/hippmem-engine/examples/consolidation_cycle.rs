//! HIPPMEM consolidation cycle example.
//!
//! Demonstrates the full lifecycle of a memory from write to consolidation:
//!   batch write → pre-consolidation check → incremental consolidation → post-consolidation comparison
//!
//! Run: cargo run --example consolidation_cycle

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{ConsolidationScope, Engine, EngineConfig, InspectQuery, WriteMemoryInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("cycle.redb"),
        ..Default::default()
    })?;

    // ── Batch write 20 memories ──
    println!("=== writing 20 memories ===");
    for i in 0..20 {
        let importance = if i < 3 {
            0.9 // High importance: core memory
        } else if i < 10 {
            0.5 // Medium importance
        } else {
            0.1 // Low importance: prone to decay
        };

        let content = if i == 0 {
            "HIPPMEM's core design: memory is not just data — it's data + associations + activation history.".to_string()
        } else if i == 1 {
            "Tech stack: Rust + redb + Tantivy + HNSW — fully embedded, zero external dependencies."
                .to_string()
        } else if i == 2 {
            "Goal: give AI agents long-term memory with a complete offline pipeline.".to_string()
        } else {
            format!("Test memory #{i}: used to verify consolidation decay and compaction behavior.")
        };

        let out = engine.write(WriteMemoryInput {
            content,
            content_type: Some(ContentType::UserStatement),
            context: ctx_default(),
            importance_hint: Some(importance),
            source_refs: vec![],
        })?;

        if i < 3 {
            println!(
                "  #{i} memory_id={} importance={importance} (core memory)",
                out.memory_id.0
            );
        }
    }
    println!("all writes completed\n");

    // ── Pre-consolidation state ──
    let before = engine.inspect(InspectQuery::StoreStats)?;
    if let hippmem_engine::InspectReport::StoreStats(ref s) = before {
        println!("=== before consolidation ===");
        println!("memories: {} edges: {}", s.memory_count, s.edge_count);
    }

    // ── Incremental consolidation ──
    println!("\n=== incremental consolidation ===");
    let report = engine.consolidate(ConsolidationScope::Incremental)?;
    println!("processed: {} memories", report.memories_processed);
    println!(
        "decayed: {} edges (strength × decay_per_cycle)",
        report.edges_decayed
    );
    println!(
        "archived: {} weak edges (below min_retained_strength)",
        report.edges_archived
    );
    println!("merged: {} edges", report.edges_merged);
    println!(
        "observation promoted: {} edges",
        report.observation_promoted
    );
    println!("summaries created: {}", report.summaries_created);
    println!("elapsed: {}ms", report.elapsed_ms);

    // ── Post-consolidation state ──
    let after = engine.inspect(InspectQuery::StoreStats)?;
    if let hippmem_engine::InspectReport::StoreStats(ref s) = after {
        println!("\n=== after consolidation ===");
        println!("memories: {} edges: {}", s.memory_count, s.edge_count);
    }

    // ── Full consolidation ──
    println!("\n=== full consolidation ===");
    let report = engine.consolidate(ConsolidationScope::Full)?;
    println!(
        "processed: {}, elapsed: {}ms",
        report.memories_processed, report.elapsed_ms
    );

    // ── Edges-only consolidation ──
    println!("\n=== edges-only consolidation ===");
    let report = engine.consolidate(ConsolidationScope::EdgesOnly)?;
    println!(
        "decayed: {} edges, archived: {}",
        report.edges_decayed, report.edges_archived
    );

    engine.close()?;
    println!("\nconsolidation cycle demo done ✓");
    Ok(())
}

fn ctx_default() -> WriteContext {
    WriteContext {
        conversation_id: None,
        session_id: None,
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: hippmem_core::time::Timestamp(0),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}
