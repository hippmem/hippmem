//! Deep diagnostic: analyze channel score distribution for 6 P@1-miss queries.
//!
//! Run: cargo run -p hippmem-eval --example diagnose_p1_miss --features api-backends
//!
//! Backend is configured via environment variables:
//!   OPENAI_API_KEY           — required
//!   HIPPMEM_EMBEDDER_BASE_URL — embedder base URL (default: https://api.openai.com/v1)
//!   HIPPMEM_EMBEDDER_MODEL    — embedder model name (default: text-embedding-3-small)

use hippmem_core::config::EmbedderConfig;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};
use hippmem_eval::bench_corpus::{load_fixture, BenchDataset, CategoryQuerySet};
use hippmem_eval::fixture_loader::discover_bench_locales;
use std::collections::HashMap;
use tempfile::tempdir;

fn write_ctx() -> WriteContext {
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

fn retrieve_ctx() -> RetrieveContext {
    RetrieveContext::default()
}

// Uses hippmem_eval::fixture_loader::discover_bench_locales (shared with tests)

fn open_engine(dir: &tempfile::TempDir) -> Engine {
    let mut config = EngineConfig {
        store_dir: dir.path().join("eval.redb"),
        ..Default::default()
    };
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        if !api_key.is_empty() {
            let base_url = std::env::var("HIPPMEM_EMBEDDER_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            let model = std::env::var("HIPPMEM_EMBEDDER_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            config.embedder = EmbedderConfig::OpenAiCompatible {
                base_url,
                model,
                api_key: Some(api_key),
                dimensions: 1024,
            };
        }
    }
    Engine::open(config).expect("Engine::open")
}

fn short(s: &str) -> String {
    s.chars().take(80).collect()
}

fn get_failing_dataset(locale: &str) -> Vec<(ContentType, String, f32, String)> {
    let dataset: BenchDataset = load_fixture(locale, "diagnostic_dataset");
    dataset
        .entries
        .into_iter()
        .map(|e| (e.content_type, e.content, e.importance, e.category))
        .collect()
}

fn get_failing_queries(locale: &str) -> Vec<(String, Vec<String>, String)> {
    let set: CategoryQuerySet = load_fixture(locale, "diagnostic_queries");
    set.queries
        .into_iter()
        .map(|q| (q.query, q.expected_categories, q.description))
        .collect()
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  HIPPMEM diagnostic: channel score analysis for 6 P@1-miss queries");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let locales = discover_bench_locales();

    for locale in &locales {
        println!("\n  Locale: {locale}");
        println!("---------------------------------------------------------------");

        let dir = tempdir().unwrap();
        let engine = open_engine(&dir);

        let dataset = get_failing_dataset(locale);
        println!("\n  Dataset: {} memories", dataset.len());

        let mut content_to_cat = HashMap::new();
        let start = std::time::Instant::now();
        for (i, (ct, content, importance, category)) in dataset.iter().enumerate() {
            engine
                .write(WriteMemoryInput {
                    content: content.to_string(),
                    content_type: Some(*ct),
                    context: write_ctx(),
                    importance_hint: Some(*importance),
                    source_refs: vec![],
                })
                .unwrap();
            content_to_cat.insert(content.to_string(), category.to_string());
            if i % 10 == 9 {
                println!("  Written {} memories...", i + 1);
            }
        }
        let write_time = start.elapsed();
        println!(
            "  Wrote {} memories in: {:.1}s",
            dataset.len(),
            write_time.as_secs_f64()
        );

        let queries = get_failing_queries(locale);
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  Per-query diagnostic analysis (locale={locale})                    ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");

        for (query, expected_cats, desc) in &queries {
            println!("\n┌─────────────────────────────────────────────────────────────┐");
            println!("│ Query: \"{}\" ({})", query, desc);
            println!("│ Expected categories: {:?}", expected_cats);
            println!("├─────────────────────────────────────────────────────────────┤");

            let results = engine
                .retrieve(RetrieveInput {
                    query: query.to_string(),
                    context: retrieve_ctx(),
                    top_k: 10,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            let channels: Vec<String> = results
                .diagnostics
                .channel_contributions
                .iter()
                .map(|(ch, cnt)| format!("{:?}(x{})", ch, cnt))
                .collect();
            println!("│ Channel contributions: {}", channels.join(", "));

            println!("│ Seed count: {}", results.trace.seeds.len());
            for seed in results.trace.seeds.iter().take(10) {
                println!(
                    "│   Seed: id={} ch={:?} energy={:.4}",
                    seed.id.0, seed.channel, seed.initial_energy
                );
            }

            println!("│ Top-10 results:");
            for (i, r) in results.results.iter().take(10).enumerate() {
                let cat = content_to_cat
                    .get(&r.memory.content.raw)
                    .cloned()
                    .unwrap_or_default();
                let hit = if expected_cats.contains(&cat) {
                    "✅"
                } else {
                    "  "
                };
                let dims: Vec<String> = r
                    .matched_dimensions
                    .iter()
                    .map(|d| format!("{:?}", d))
                    .collect();
                let content_type = format!("{:?}", r.memory.content.content_type);
                println!(
                    "│ {}. {}[{:.4}] dims={} ct={} | {}",
                    i + 1,
                    hit,
                    r.final_score,
                    dims.join(","),
                    content_type,
                    short(&r.memory.content.raw),
                );
            }

            let mut found_rank = None;
            for (i, r) in results.results.iter().enumerate() {
                let cat = content_to_cat
                    .get(&r.memory.content.raw)
                    .cloned()
                    .unwrap_or_default();
                if expected_cats.contains(&cat) {
                    if found_rank.is_none() {
                        found_rank = Some(i + 1);
                    }
                    let dims: Vec<String> = r
                        .matched_dimensions
                        .iter()
                        .map(|d| format!("{:?}", d))
                        .collect();
                    println!(
                        "│    → Correct answer rank #{}: score={:.4} dims=[{}] \"{}\"",
                        i + 1,
                        r.final_score,
                        dims.join(","),
                        short(&r.memory.content.raw),
                    );
                }
            }
            if found_rank.is_none() {
                println!("│    ⚠️ Correct answer not in Top-10!");
            }
        }

        println!("\n  Total time: {:.1}s", write_time.as_secs_f64());
        engine.close().unwrap();
    }
}
