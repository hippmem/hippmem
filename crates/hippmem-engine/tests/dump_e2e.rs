//! acceptance test: dump API.
//!
//! Tests full JSONL export: round-trip fidelity, file writing, empty database.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput};
use std::io::Read;
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

#[test]
fn dump_returns_jsonl_string() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    engine
        .write(WriteMemoryInput {
            content: "first memory".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "second memory".into(),
            content_type: Some(ContentType::Decision),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    let output = engine.dump(hippmem_engine::DumpInput::default()).unwrap();

    assert_eq!(output.count, 2);
    assert!(
        output.json.is_some(),
        "should return a json string when no file path is given"
    );

    let json = output.json.unwrap();
    assert!(!json.is_empty());

    // Verify JSONL format: one complete JSON object per line
    let lines: Vec<&str> = json.trim().lines().collect();
    assert_eq!(lines.len(), 2, "should have 2 JSON lines");

    for line in &lines {
        // Each line should parse as a serde_json Value
        let _: serde_json::Value =
            serde_json::from_str(line).expect("each line should be valid JSON");
    }

    engine.close().unwrap();
}

#[test]
fn dump_writes_to_file() {
    let dir = tempdir().unwrap();
    let output_path = dir.path().join("dump.jsonl");
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    engine
        .write(WriteMemoryInput {
            content: "test memory".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    let output = engine
        .dump(hippmem_engine::DumpInput {
            output_path: Some(output_path.clone()),
        })
        .unwrap();

    assert_eq!(output.count, 1);
    assert!(output.written_to.is_some());
    assert!(output_path.exists(), "file should be created");

    // Verify file content
    let mut file = std::fs::File::open(&output_path).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    assert!(
        content.contains("test memory"),
        "file should contain the original content"
    );
    assert!(!content.trim().is_empty());

    engine.close().unwrap();
}

#[test]
fn dump_empty_database() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let output = engine.dump(hippmem_engine::DumpInput::default()).unwrap();

    assert_eq!(output.count, 0);
    // Empty database: json should be empty (rather than panic)
    if let Some(json) = &output.json {
        assert!(json.trim().is_empty() || json == "\n");
    }

    engine.close().unwrap();
}
