//! HIPPMEM vs mem0 comparison benchmark — small-batch test with 100 items
//!
//! Design principles:
//!   1. 100 memories covering multiple types (facts, preferences, decisions, project knowledge, events, etc.)
//!   2. 20 evaluation queries covering different retrieval scenarios
//!   3. Metrics: Precision@1/3/5, Recall@5, MRR, NDCG@5
//!   4. Only memory effectiveness is evaluated; architecture, business model, etc. are ignored
//!
//! Run: cargo test -p hippmem-eval --test bench_mem0_comparison --features api-backends -- --nocapture

mod common;

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput};
use hippmem_eval::bench_corpus::{load_fixture, BenchDataset, CategoryQuerySet};
use std::collections::{HashMap, HashSet};
use tempfile::tempdir;

// ═══════════════════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════════════════

fn short(s: &str) -> String {
    s.chars().take(70).collect()
}

/// 100-item memory dataset, loaded from `fixtures/bench/<locale>/mem0_comparison_dataset.json`.
///
/// Returns `(content_type, content, importance, category)` tuples. Content and
/// category are now owned `String` (sourced from JSON) rather than `&'static str`.
fn get_memory_dataset(locale: &str) -> Vec<(ContentType, String, f32, String)> {
    let dataset: BenchDataset = load_fixture(locale, "mem0_comparison_dataset");
    dataset
        .entries
        .into_iter()
        .map(|e| (e.content_type, e.content, e.importance, e.category))
        .collect()
}

/// Evaluation query set, loaded from `fixtures/bench/<locale>/mem0_comparison_queries.json`.
///
/// Returns `(query, expected_categories, description)` tuples with owned
/// `String` fields (sourced from JSON).
fn get_queries(locale: &str) -> Vec<(String, Vec<String>, String)> {
    let set: CategoryQuerySet = load_fixture(locale, "mem0_comparison_queries");
    set.queries
        .into_iter()
        .map(|q| (q.query, q.expected_categories, q.description))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Metric computation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Default)]
struct Metrics {
    precision_at_1: f64,
    precision_at_3: f64,
    precision_at_5: f64,
    recall_at_5: f64,
    mrr: f64,
    ndcg_at_5: f64,
    top3_hit_rate: f64,
}

fn dcg(relevance: &[f64], k: usize) -> f64 {
    relevance
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| rel / (i as f64 + 2.0).log2())
        .sum()
}

fn idcg(relevance: &[f64], k: usize) -> f64 {
    let mut sorted: Vec<f64> = relevance.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    dcg(&sorted, k)
}

fn ndcg(relevance: &[f64], k: usize) -> f64 {
    let d = dcg(relevance, k);
    let i = idcg(relevance, k);
    if i == 0.0 {
        0.0
    } else {
        d / i
    }
}

fn evaluate(
    results: &[Vec<(String, f64)>],
    queries: &[(String, Vec<String>, String)],
    cat_map: &HashMap<String, String>,
    top_k: usize,
) -> Metrics {
    let mut p1_sum = 0.0;
    let mut p3_sum = 0.0;
    let mut p5_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut mrr_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut top3_hit_count = 0;
    let mut valid_queries = 0;

    for (qi, (_, expected_cats, desc)) in queries.iter().enumerate() {
        if qi >= results.len() {
            break;
        }
        let expected: HashSet<&str> = expected_cats.iter().map(|s| s.as_str()).collect();
        let has_expected = !expected.is_empty() && expected != HashSet::from(["noise"]);
        if !has_expected {
            continue;
        }
        valid_queries += 1;

        let result_list = &results[qi];

        // Build category and relevance score for each returned result
        let returned_cats: Vec<&str> = result_list
            .iter()
            .map(|(content, _)| {
                cat_map
                    .iter()
                    .find(|(mem_content, _)| {
                        content.contains(mem_content.as_str())
                            || mem_content.contains(content.as_str())
                    })
                    .map(|(_, cat)| cat.as_str())
                    .unwrap_or("unknown")
            })
            .collect();

        let mut rels: Vec<f64> = returned_cats
            .iter()
            .map(|c| if expected.contains(c) { 1.0 } else { 0.0 })
            .collect();

        // P@1
        let p1 = if !returned_cats.is_empty() {
            expected.contains(returned_cats[0])
        } else {
            false
        };
        if p1 {
            p1_sum += 1.0;
        }

        // P@3
        let k3 = 3.min(returned_cats.len());
        if k3 > 0 {
            let hit = returned_cats[..k3].iter().any(|c| expected.contains(c));
            if hit {
                p3_sum += 1.0;
                top3_hit_count += 1;
            }
        }

        // P@5
        let k5 = 5.min(returned_cats.len());
        if k5 > 0 {
            let hit = returned_cats[..k5].iter().any(|c| expected.contains(c));
            if hit {
                p5_sum += 1.0;
            }
        }

        // Recall@5
        let hit_any = returned_cats.iter().any(|c| expected.contains(c));
        if hit_any {
            recall_sum += 1.0;
        }

        // MRR
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

        // NDCG@5
        while rels.len() < top_k {
            rels.push(0.0);
        }
        ndcg_sum += ndcg(&rels, top_k);

        println!(
            "  {:<35} | P@1={} | P@3={} | MRR={:.3} | cats(top5): {:?}",
            desc,
            if p1 { "✅" } else { "❌" },
            if returned_cats.iter().take(3).any(|c| expected.contains(c)) {
                "✅"
            } else {
                "❌"
            },
            if rank > 0 { 1.0 / rank as f64 } else { 0.0 },
            &returned_cats.iter().take(5).copied().collect::<Vec<_>>(),
        );
    }

    if valid_queries == 0 {
        return Metrics::default();
    }

    Metrics {
        precision_at_1: p1_sum / valid_queries as f64,
        precision_at_3: p3_sum / valid_queries as f64,
        precision_at_5: p5_sum / valid_queries as f64,
        recall_at_5: recall_sum / valid_queries as f64,
        mrr: mrr_sum / valid_queries as f64,
        ndcg_at_5: ndcg_sum / valid_queries as f64,
        top3_hit_rate: top3_hit_count as f64 / valid_queries as f64,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIPPMEM evaluation
// ═══════════════════════════════════════════════════════════════════════════════

fn run_hippmem_eval(
    engine: &Engine,
    dataset: &[(ContentType, String, f32, String)],
    locale: &str,
) -> Metrics {
    println!(
        "\n  ═══ Writing {} memories (batch mode) ═══",
        dataset.len()
    );

    // Batch write optimization: one API call embeds all texts, Tantivy commits every 20 items
    engine.set_fulltext_commit_every(20);

    let mut content_to_cat = HashMap::new();
    let inputs: Vec<WriteMemoryInput> = dataset
        .iter()
        .map(|(ct, content, importance, _category)| WriteMemoryInput {
            content: content.to_string(),
            content_type: Some(*ct),
            context: common::write_ctx(),
            importance_hint: Some(*importance),
            source_refs: vec![],
        })
        .collect();

    // Record category mapping
    for (_ct, content, _importance, category) in dataset.iter() {
        content_to_cat.insert(content.to_string(), category.to_string());
    }

    let t0 = std::time::Instant::now();
    let outputs = engine.write_batch(inputs).unwrap();
    let write_time = t0.elapsed();
    println!(
        "    All {} writes completed ({:.1}s, avg {:.0}ms/item)",
        outputs.len(),
        write_time.as_secs_f64(),
        write_time.as_secs_f64() * 1000.0 / outputs.len() as f64
    );

    // Force-commit remaining full-text index
    engine.flush_fulltext();

    // Retrieval evaluation
    let queries = get_queries(locale);
    println!("\n  ═══ Running {} evaluation queries ═══", queries.len());

    let mut all_results: Vec<Vec<(String, f64)>> = Vec::new();

    // Collect results for both modes for comparison
    for (query, _, desc) in &queries {
        println!("\n  ── {} ──", desc);
        println!("  Query: \"{}\"", query);

        // Balanced mode
        let results = engine
            .retrieve(hippmem_engine::RetrieveInput {
                query: query.to_string(),
                context: common::retrieve_ctx(),
                top_k: 5,
                max_hops: Some(2),
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();

        let result_list: Vec<(String, f64)> = results
            .results
            .iter()
            .map(|r| (r.memory.content.raw.clone(), r.final_score as f64))
            .collect();

        for (i, (content, score)) in result_list.iter().enumerate() {
            let cat = content_to_cat
                .get(content.as_str())
                .cloned()
                .unwrap_or_default();
            println!(
                "    {}. [{:.3}] {} | cat={}",
                i + 1,
                score,
                short(content),
                cat,
            );
        }
        all_results.push(result_list);
    }

    evaluate(&all_results, &queries, &content_to_cat, 5)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main test entry
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg(feature = "api-backends")]
fn test_hippmem_100_items_benchmark() {
    for locale in common::discover_bench_locales() {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  HIPPMEM 100-item small-batch memory effectiveness benchmark      ║");
        println!("║  Locale: {:<56} ║", locale);
        println!("║  Backend: API (OPENAI_API_KEY + HIPPMEM_EMBEDDER_BASE_URL)       ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");

        let dir = tempdir().unwrap();
        let engine = common::open_engine(&dir);

        let dataset = get_memory_dataset(&locale);
        println!(
            "\n  Dataset: {} memories (80 valuable + 20 noise)",
            dataset.len()
        );

        let start = std::time::Instant::now();
        let metrics = run_hippmem_eval(&engine, &dataset, &locale);
        let elapsed = start.elapsed();

        println!("\n  ═══════════════════════════════════════════");
        println!("  HIPPMEM evaluation results summary:");
        println!("  ═══════════════════════════════════════════");
        println!("    Precision@1:   {:.1}%", metrics.precision_at_1 * 100.0);
        println!("    Precision@3:   {:.1}%", metrics.precision_at_3 * 100.0);
        println!("    Precision@5:   {:.1}%", metrics.precision_at_5 * 100.0);
        println!("    Recall@5:      {:.1}%", metrics.recall_at_5 * 100.0);
        println!("    MRR:           {:.3}", metrics.mrr);
        println!("    NDCG@5:        {:.3}", metrics.ndcg_at_5);
        println!("    Top-3 hit rate: {:.1}%", metrics.top3_hit_rate * 100.0);
        println!("    Total time:     {:.1}s", elapsed.as_secs_f64());

        // Quality threshold assertions
        assert!(
            metrics.precision_at_1 >= 0.25,
            "Precision@1 should be >= 25%"
        );
        assert!(
            metrics.precision_at_3 >= 0.40,
            "Precision@3 should be >= 40%"
        );
        assert!(metrics.mrr >= 0.35, "MRR should be >= 0.35");

        engine.close().unwrap();
    }
}

/// Fallback backend mode (no network)
#[test]
fn test_hippmem_100_items_fallback() {
    for locale in common::discover_bench_locales() {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  HIPPMEM 100-item benchmark (fallback backend / offline mode)     ║");
        println!("║  Locale: {:<56} ║", locale);
        println!("╚══════════════════════════════════════════════════════════════════╝");

        let dir = tempdir().unwrap();
        let engine = Engine::open(EngineConfig {
            store_dir: dir.path().join("hippmem.redb"),
            ..Default::default()
        })
        .unwrap();

        let dataset = get_memory_dataset(&locale);
        let start = std::time::Instant::now();
        let metrics = run_hippmem_eval(&engine, &dataset, &locale);
        let elapsed = start.elapsed();

        println!("\n  ═══════════════════════════════════════════");
        println!("  HIPPMEM (fallback backend) evaluation results summary:");
        println!("  ═══════════════════════════════════════════");
        println!("    Precision@1:   {:.1}%", metrics.precision_at_1 * 100.0);
        println!("    Precision@3:   {:.1}%", metrics.precision_at_3 * 100.0);
        println!("    Precision@5:   {:.1}%", metrics.precision_at_5 * 100.0);
        println!("    Recall@5:      {:.1}%", metrics.recall_at_5 * 100.0);
        println!("    MRR:           {:.3}", metrics.mrr);
        println!("    NDCG@5:        {:.3}", metrics.ndcg_at_5);
        println!("    Top-3 hit rate: {:.1}%", metrics.top3_hit_rate * 100.0);
        println!("    Total time:     {:.1}s", elapsed.as_secs_f64());

        // Fallback backend quality thresholds (more lenient)
        assert!(
            metrics.precision_at_1 >= 0.15,
            "Precision@1 should be >= 15%"
        );
        assert!(
            metrics.precision_at_3 >= 0.25,
            "Precision@3 should be >= 25%"
        );
        assert!(metrics.mrr >= 0.20, "MRR should be >= 0.20");

        engine.close().unwrap();
    }
}
