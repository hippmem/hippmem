//! Inserts 100 test memories of various types into the live HIPPMEM database.
//!
//! Locale-specific seed data lives in `examples/fixtures/seed_100/<locale>.json`
//! per P7 (test-data locale symmetry). All 100 items organized in 8 sections.
//! Default locale: "zh". Override via first CLI arg.
//!
//! Run: cargo run --example seed_100 --features api-backends [locale]

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput};
use serde::Deserialize;

#[derive(Deserialize)]
struct SeedSection {
    content_type: String,
    conv_id: u64,
    importance: f32,
    items: Vec<String>,
}

#[derive(Deserialize)]
struct SeedFixture {
    sections: Vec<SeedSection>,
}

fn load_fixture(locale: &str) -> SeedFixture {
    let path = format!(
        "{}/examples/fixtures/seed_100/{locale}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale = locale
    );
    let data = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("failed to read seed_100/{locale}.json: {e}\nUsage: cargo run --example seed_100 --features api-backends [zh|en]")
    });
    serde_json::from_str(&data).expect("invalid fixture")
}

fn make_context(conv_id: u64, ts_ms: i64) -> WriteContext {
    WriteContext {
        conversation_id: Some(conv_id),
        session_id: Some(conv_id / 10),
        project_id: Some(1),
        task_id: None,
        user_id: Some(1),
        local_time: hippmem_core::time::Timestamp(ts_ms),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

fn parse_content_type(s: &str) -> ContentType {
    match s {
        "Decision" => ContentType::Decision,
        "Preference" => ContentType::Preference,
        "ProjectKnowledge" => ContentType::ProjectKnowledge,
        "TaskState" => ContentType::TaskState,
        "Correction" => ContentType::Correction,
        "AssistantObservation" => ContentType::AssistantObservation,
        _ => ContentType::UserStatement,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default locale; override via first CLI arg
    let locale = std::env::args().nth(1).unwrap_or_else(|| "zh".to_string());
    println!("using locale: {locale}");

    let store_path = std::path::PathBuf::from("hippmem_data");

    if store_path.exists() {
        println!(
            "clearing old database file: {:?} ({})",
            store_path,
            store_path.metadata().map(|m| m.len()).unwrap_or(0)
        );
        std::fs::remove_file(&store_path)?;
    }

    let config = EngineConfig {
        store_dir: store_path.clone(),
        ..Default::default()
    };
    let engine = Engine::open(config)?;
    println!("database opened: {:?}", store_path);

    let fixture = load_fixture(&locale);
    let base_ts: i64 = 1_700_000_000_000;
    let mut count = 0;

    for section in &fixture.sections {
        let ct = parse_content_type(&section.content_type);
        for item in &section.items {
            let out = engine.write(WriteMemoryInput {
                content: item.clone(),
                content_type: Some(ct),
                context: make_context(section.conv_id, base_ts + count as i64 * 1000),
                importance_hint: Some(section.importance + (count as f32 * 0.001)),
                source_refs: vec![],
            })?;
            count += 1;
            println!(
                "[{:03}] ✅ {} | id={}",
                count,
                item.chars().take(30).collect::<String>(),
                out.memory_id.0
            );
        }
    }

    engine.close()?;
    println!(
        "\n✅ done! inserted {} memories into {:?}",
        count, store_path
    );
    println!("data is persisted on disk and available for manual query testing.");
    Ok(())
}
