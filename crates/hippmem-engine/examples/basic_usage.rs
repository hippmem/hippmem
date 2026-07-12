//! HIPPMEM quick-start example.
//!
//! Run: cargo run --example basic_usage --features api-backends
//! (without an API key: cargo run --example basic_usage, uses the degraded backend)

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use a temporary directory for storage
    let dir = tempfile::tempdir()?;
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config)?;

    // Write a memory
    let out = engine.write(WriteMemoryInput {
        content: "The user prefers Rust for development and redb for embedded storage.".into(),
        content_type: Some(ContentType::Preference),
        context: WriteContext {
            conversation_id: Some(1),
            session_id: Some(1),
            ..ctx_default()
        },
        importance_hint: Some(0.8),
        source_refs: vec![],
    })?;
    println!("write succeeded: memory_id={}", out.memory_id.0);

    // Retrieve memories
    let results = engine.retrieve(RetrieveInput {
        query: "What language and database does the user prefer?".into(),
        context: RetrieveContext::default(),
        top_k: 3,
        max_hops: None,
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    })?;

    for r in &results.results {
        println!(
            "[{:.3}] {} (dims: {:?})",
            r.final_score,
            r.memory.content.raw.chars().take(50).collect::<String>(),
            r.matched_dimensions
        );
    }

    engine.close()?;
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
