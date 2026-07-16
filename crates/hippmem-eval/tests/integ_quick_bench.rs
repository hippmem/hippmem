//! HIPPMEM vs mem0 quick comparison benchmark — compact version
//! 50 records, 15 queries, structured evaluation
//! Run: cargo test -p hippmem-eval --test quick_bench -- --nocapture

mod common;

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_engine::{Engine, EngineConfig, RetrieveInput, WriteMemoryInput};
use hippmem_eval::bench_corpus::{load_fixture, BenchDataset, CategoryQuerySet};
use std::collections::{HashMap, HashSet};
use tempfile::tempdir;

/// Load the 50-record memory dataset from `fixtures/bench/<locale>/quick_bench_dataset.json`.
fn get_memory_dataset(locale: &str) -> Vec<(ContentType, String, f32, String)> {
    let dataset: BenchDataset = load_fixture(locale, "quick_bench_dataset");
    dataset
        .entries
        .into_iter()
        .map(|e| (e.content_type, e.content, e.importance, e.category))
        .collect()
}

/// Load the evaluation query set from `fixtures/bench/<locale>/quick_bench_queries.json`.
fn get_queries(locale: &str) -> Vec<(String, Vec<String>, String)> {
    let set: CategoryQuerySet = load_fixture(locale, "quick_bench_queries");
    set.queries
        .into_iter()
        .map(|q| (q.query, q.expected_categories, q.description))
        .collect()
}

#[test]
fn test_quick_bench() {
    let locales = common::discover_bench_locales();
    for locale in &locales {
        let dir = tempdir().unwrap();
        let engine = Engine::open(EngineConfig {
            store_dir: dir.path().join("bench.redb"),
            ..Default::default()
        })
        .unwrap();

        // ── 50 structured records (45 valuable + 5 noise), loaded from JSON fixture ──
        let dataset = get_memory_dataset(locale);
        // ── 15 evaluation queries, loaded from JSON fixture ──
        let queries = get_queries(locale);

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!(
            "║  HIPPMEM 50-record small-batch benchmark (locale: {}) ║",
            locale
        );
        println!("╚══════════════════════════════════════════════════════════╝");
        println!(
            "  Dataset: {} records (45 valuable + 5 noise)",
            dataset.len()
        );

        // Write
        let mut content_to_cat: HashMap<String, String> = HashMap::new();
        println!("\n  Writing {} memories...", dataset.len());
        let start = std::time::Instant::now();
        for (ct, content, imp, cat) in &dataset {
            engine
                .write(WriteMemoryInput {
                    content: content.to_string(),
                    content_type: Some(*ct),
                    context: common::write_ctx(),
                    importance_hint: Some(*imp),
                    source_refs: vec![],
                })
                .unwrap();
            content_to_cat.insert(content.to_string(), cat.clone());
        }
        println!(
            "  Write done, elapsed {:.1}s",
            start.elapsed().as_secs_f64()
        );

        // Retrieval evaluation
        println!("\n  ═══ Running {} evaluation queries ═══", queries.len());
        let mut p1_sum = 0u32;
        let mut p3_sum = 0u32;
        let mut p5_sum = 0u32;
        let mut recall_sum = 0u32;
        let mut mrr_sum = 0.0f64;
        let mut top3_hits = 0u32;
        let mut valid = 0u32;

        println!("\n  {:<30} | P@1 | P@3 | MRR   | Top cats", "Query");
        println!("  {:-<30}-|-----|-----|-------|----------", "");

        for (query, expected_cats, desc) in &queries {
            let expected: HashSet<&str> = expected_cats.iter().map(|s| s.as_str()).collect();
            let has_expected = !expected.is_empty() && expected != HashSet::from(["noise"]);
            if !has_expected {
                continue;
            }
            valid += 1;

            let results = engine
                .retrieve(RetrieveInput {
                    query: query.to_string(),
                    context: common::retrieve_ctx(),
                    top_k: 5,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            let returned_cats: Vec<&str> = results
                .results
                .iter()
                .map(|r| {
                    content_to_cat
                        .get(r.memory.content.raw.as_str())
                        .map(|s| s.as_str())
                        .unwrap_or("?")
                })
                .collect();

            let p1 = returned_cats.first().is_some_and(|c| expected.contains(c));
            if p1 {
                p1_sum += 1;
            }
            let k3 = 3.min(returned_cats.len());
            let p3 = k3 > 0 && returned_cats[..k3].iter().any(|c| expected.contains(c));
            if p3 {
                p3_sum += 1;
                top3_hits += 1;
            }
            let k5 = 5.min(returned_cats.len());
            let p5 = k5 > 0 && returned_cats[..k5].iter().any(|c| expected.contains(c));
            if p5 {
                p5_sum += 1;
            }
            if returned_cats.iter().any(|c| expected.contains(c)) {
                recall_sum += 1;
            }

            let mut rank = 0;
            for (i, c) in returned_cats.iter().enumerate() {
                if expected.contains(c) {
                    rank = i + 1;
                    break;
                }
            }
            if rank > 0 {
                mrr_sum += 1.0 / rank as f64;
            }

            let mrr_val = if rank > 0 { 1.0 / rank as f64 } else { 0.0 };
            println!(
                "  {:<30} | {:^3} | {:^3} | {:>5.3} | {:?}",
                desc,
                if p1 { "✅" } else { "❌" },
                if p3 { "✅" } else { "❌" },
                mrr_val,
                &returned_cats[..returned_cats.len().min(5)],
            );
        }

        let elapsed = start.elapsed();
        let v = valid as f64;

        println!("\n  ═══════════════════════════════════════════");
        println!(
            "  HIPPMEM (fallback backend) evaluation summary (locale: {}):",
            locale
        );
        println!("  ═══════════════════════════════════════════");
        println!("    Precision@1:   {:.1}%", p1_sum as f64 / v * 100.0);
        println!("    Precision@3:   {:.1}%", p3_sum as f64 / v * 100.0);
        println!("    Precision@5:   {:.1}%", p5_sum as f64 / v * 100.0);
        println!("    Recall@5:      {:.1}%", recall_sum as f64 / v * 100.0);
        println!("    MRR:           {:.3}", mrr_sum / v);
        println!("    Top-3 hit rate: {:.1}%", top3_hits as f64 / v * 100.0);
        println!("    Total elapsed:  {:.1}s", elapsed.as_secs_f64());

        // Quality assertions
        assert!(p1_sum as f64 / v >= 0.20, "[{locale}] Precision@1 >= 20%");
        assert!(p3_sum as f64 / v >= 0.35, "[{locale}] Precision@3 >= 35%");
        assert!(mrr_sum / v >= 0.30, "[{locale}] MRR >= 0.30");

        engine.close().unwrap();
    }
}
