//! HIPPMEM multi-turn Agent session example.
//!
//! Simulates a multi-turn conversation between an AI Agent and a user:
//!   write → retrieve context → feedback → consolidate
//!
//! Run: cargo run --example agent_session

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{
    ConsolidationScope, Engine, EngineConfig, FeedbackInput, InspectQuery, RetrieveContext,
    RetrieveInput, UsageSignal, WriteMemoryInput,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("agent.redb"),
        ..Default::default()
    })?;

    let session_id = 1u64;
    let mut retrieval_id = 0u64;

    // ── Turn 1: user introduces background ──
    println!("=== Turn 1: user introduces background ===");
    let out = engine.write(WriteMemoryInput {
        content: "The user is Chen Ming, a software engineer working on an open-source memory engine project.".into(),
        content_type: Some(ContentType::UserStatement),
        context: WriteContext {
            session_id: Some(session_id),
            ..ctx_default()
        },
        importance_hint: Some(0.7),
        source_refs: vec![],
    })?;
    println!(
        "write: memory_id={} stage={:?} links={}",
        out.memory_id.0,
        out.stage_reached,
        out.created_links.len()
    );

    // ── Turn 2: user asks a technical question ──
    println!("\n=== Turn 2: retrieve + answer ===");
    retrieval_id += 1;
    let results = engine.retrieve(RetrieveInput {
        query: "What tech stack does this project use?".into(),
        context: RetrieveContext {
            session_id: Some(session_id),
            ..Default::default()
        },
        top_k: 3,
        max_hops: Some(2),
        retrieval_mode: RetrievalMode::Balanced,
    })?;

    println!("retrieved {} related memories:", results.results.len());
    for r in &results.results {
        println!(
            "  [{:.3}] {} (dims: {:?})",
            r.final_score,
            r.memory.content.raw.chars().take(60).collect::<String>(),
            r.matched_dimensions
        );
    }

    // Feedback: user confirmed the retrieval results
    let used_ids: Vec<_> = results.results.iter().map(|r| r.memory.id).collect();
    engine.feedback(FeedbackInput {
        retrieval_id,
        used_memory_ids: used_ids.clone(),
        signal: UsageSignal::UserConfirmedCorrect,
    })?;
    println!("feedback: user confirmed {} results", used_ids.len());

    // ── Turn 3: user adds preferences ──
    println!("\n=== Turn 3: add preferences ===");
    let out = engine.write(WriteMemoryInput {
        content: "The user prefers Rust, uses redb for embedded storage, and wants to avoid external database dependencies.".into(),
        content_type: Some(ContentType::Preference),
        context: WriteContext {
            session_id: Some(session_id),
            ..ctx_default()
        },
        importance_hint: Some(0.8),
        source_refs: vec![],
    })?;
    println!("write preference: links={}", out.created_links.len());

    // ── Consolidate ──
    println!("\n=== consolidate ===");
    let report = engine.consolidate(ConsolidationScope::Incremental)?;
    println!(
        "processed {} memories, decayed {} edges, elapsed {}ms",
        report.memories_processed, report.edges_decayed, report.elapsed_ms
    );

    // ── Final diagnostics ──
    println!("\n=== final state ===");
    let r = engine.inspect(InspectQuery::StoreStats)?;
    if let hippmem_engine::InspectReport::StoreStats(s) = r {
        println!(
            "memories: {} edges: {} backlog: {}",
            s.memory_count, s.edge_count, s.queue_backlog
        );
    }

    engine.close()?;
    println!("\nsession simulation done ✓");
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
