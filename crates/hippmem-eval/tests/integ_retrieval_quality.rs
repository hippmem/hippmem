//! acceptance test: retrieval quality benchmark (precision/recall/MRR/NDCG)
//!
//! Builds 30 memories + 10 queries with known correct answers, then computes standard IR metrics.
//! No hard thresholds; metrics are recorded in comments for trend tracking (06-eval-framework).

mod common;

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_engine::{Engine, EngineConfig, RetrieveInput, WriteMemoryInput};
use hippmem_eval::bench_corpus::{load_fixture, CategoryQuerySet, CategoryTextSet};
use std::collections::HashSet;
use tempfile::tempdir;

/// Write 30 memories covering multiple topics, returning (memory_ids, category_map).
/// Each category has 5 memories, 6 categories in total.
///
/// Memory texts and category labels live in `fixtures/bench/<locale>/retrieval_quality_categories.json`
/// (externalized to keep this source file free of non-English string literals).
fn setup_memories(
    engine: &Engine,
    locale: &str,
) -> Vec<(String, Vec<hippmem_core::ids::MemoryId>)> {
    let dataset: CategoryTextSet = load_fixture(locale, "retrieval_quality_categories");

    let mut result = Vec::new();
    for group in &dataset.categories {
        let mut ids = Vec::new();
        for text in &group.texts {
            let output = engine
                .write(WriteMemoryInput {
                    content: text.clone(),
                    content_type: Some(ContentType::ProjectKnowledge),
                    context: common::write_ctx(),
                    importance_hint: None,
                    source_refs: vec![],
                })
                .unwrap();
            ids.push(output.memory_id);
        }
        result.push((group.category.clone(), ids));
    }
    result
}

/// Compute MRR (Mean Reciprocal Rank)
fn mrr(queries: &[(Vec<u64>, Vec<u64>)]) -> f64 {
    let mut sum = 0.0;
    for (retrieved, relevant) in queries {
        let rel_set: HashSet<_> = relevant.iter().collect();
        for (i, id) in retrieved.iter().enumerate() {
            if rel_set.contains(id) {
                sum += 1.0 / (i as f64 + 1.0);
                break;
            }
        }
    }
    if queries.is_empty() {
        0.0
    } else {
        sum / queries.len() as f64
    }
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain)
fn ndcg_at_k(retrieved: &[u64], relevant: &HashSet<u64>, k: usize) -> f64 {
    let k = k.min(retrieved.len());
    if k == 0 || relevant.is_empty() {
        return 1.0;
    }

    // DCG
    let mut dcg = 0.0;
    for (i, id) in retrieved.iter().enumerate().take(k) {
        let rel = if relevant.contains(id) { 1.0 } else { 0.0 };
        dcg += rel / ((i as f64 + 2.0).log2());
    }

    // IDCG (ideal: all relevant first)
    let ideal_rel_count = relevant.len().min(k);
    let mut idcg = 0.0;
    for i in 0..ideal_rel_count {
        idcg += 1.0 / ((i as f64 + 2.0).log2());
    }
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[test]
fn retrieval_quality_benchmark() {
    let locales = common::discover_bench_locales();
    assert!(!locales.is_empty(), "should have at least one bench locale");

    for locale in &locales {
        let dir = tempdir().unwrap();
        let engine = Engine::open(EngineConfig {
            store_dir: dir.path().join("hippmem.redb"),
            ..Default::default()
        })
        .unwrap();

        // ── Step 1: write 30 memories ──
        let categories = setup_memories(&engine, locale);
        assert_eq!(
            categories.len(),
            6,
            "should have 6 categories for locale {locale}"
        );

        // Build memory ids → category mapping
        let mut id_to_cat: Vec<(hippmem_core::ids::MemoryId, &str)> = Vec::new();
        for (cat, ids) in &categories {
            for id in ids {
                id_to_cat.push((*id, cat.as_str()));
            }
        }
        assert_eq!(
            id_to_cat.len(),
            30,
            "should have 30 memories for locale {locale}"
        );

        // ── Step 2: 10 queries, each with known correct answers ──
        // Query data lives in `fixtures/bench/<locale>/retrieval_quality_queries.json`.
        let query_set: CategoryQuerySet = load_fixture(locale, "retrieval_quality_queries");

        let mut precision_at_1 = 0.0;
        let mut precision_at_3 = 0.0;
        let mut precision_at_5 = 0.0;
        let mut recall_sum = 0.0;
        let mut mrr_data: Vec<(Vec<u64>, Vec<u64>)> = Vec::new();
        let mut ndcg_sum_5 = 0.0;
        let mut ndcg_sum_10 = 0.0;

        for q in &query_set.queries {
            let query_text = q.query.as_str();
            let expected_cats: Vec<&str> =
                q.expected_categories.iter().map(|s| s.as_str()).collect();

            let retrieve = engine
                .retrieve(RetrieveInput {
                    query: query_text.to_string(),
                    context: common::retrieve_ctx(),
                    top_k: 10,
                    max_hops: None,
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            // Collect relevant memory IDs (memories matching the expected categories)
            let relevant_ids: Vec<u64> = id_to_cat
                .iter()
                .filter(|(_, cat)| expected_cats.contains(cat))
                .map(|(id, _)| id.0 as u64)
                .collect();
            let relevant_set: HashSet<u64> = relevant_ids.iter().copied().collect();

            // Retrieved result IDs
            let retrieved_ids: Vec<u64> = retrieve
                .results
                .iter()
                .map(|r| r.memory.id.0 as u64)
                .collect();

            // Precision@k
            let hit_at_1 = if !retrieved_ids.is_empty() && relevant_set.contains(&retrieved_ids[0])
            {
                1.0
            } else {
                0.0
            };
            precision_at_1 += hit_at_1;

            let k3 = 3.min(retrieved_ids.len());
            let hit_at_3 = if k3 > 0 {
                retrieved_ids[..k3]
                    .iter()
                    .filter(|id| relevant_set.contains(id))
                    .count() as f64
                    / k3 as f64
            } else {
                0.0
            };
            precision_at_3 += hit_at_3;

            let k5 = 5.min(retrieved_ids.len());
            let hit_at_5 = if k5 > 0 {
                retrieved_ids[..k5]
                    .iter()
                    .filter(|id| relevant_set.contains(id))
                    .count() as f64
                    / k5 as f64
            } else {
                0.0
            };
            precision_at_5 += hit_at_5;

            // Recall@10
            let recall = if relevant_ids.is_empty() {
                1.0
            } else {
                let hit = retrieved_ids
                    .iter()
                    .filter(|id| relevant_set.contains(id))
                    .count();
                hit as f64 / relevant_ids.len() as f64
            };
            recall_sum += recall;

            // MRR data
            mrr_data.push((retrieved_ids.clone(), relevant_ids.clone()));

            // NDCG
            ndcg_sum_5 += ndcg_at_k(&retrieved_ids, &relevant_set, 5);
            ndcg_sum_10 += ndcg_at_k(&retrieved_ids, &relevant_set, 10);
        }

        let n = query_set.queries.len() as f64;

        let avg_precision_1 = precision_at_1 / n;
        let avg_precision_3 = precision_at_3 / n;
        let avg_precision_5 = precision_at_5 / n;
        let avg_recall = recall_sum / n;
        let avg_mrr = mrr(&mrr_data);
        let avg_ndcg_5 = ndcg_sum_5 / n;
        let avg_ndcg_10 = ndcg_sum_10 / n;

        // ── Step 3: verify all metrics are within reasonable range (no hard thresholds) ──
        // Baseline values (2026-06-08):
        //   Precision@1:  ~0.2-0.6
        //   Precision@3:  ~0.2-0.5
        //   Precision@5:  ~0.2-0.5
        //   Recall@10:    ~0.3-0.8
        //   MRR:          ~0.3-0.7
        //   NDCG@5:       ~0.3-0.7
        //   NDCG@10:      ~0.3-0.7

        assert!(
            (0.0..=1.0).contains(&avg_precision_1),
            "Precision@1 should be in [0,1] for locale {locale}"
        );
        assert!(
            (0.0..=1.0).contains(&avg_precision_3),
            "Precision@3 should be in [0,1] for locale {locale}"
        );
        assert!(
            (0.0..=1.0).contains(&avg_precision_5),
            "Precision@5 should be in [0,1] for locale {locale}"
        );
        assert!(
            (0.0..=1.0).contains(&avg_recall),
            "Recall@10 should be in [0,1] for locale {locale}"
        );
        assert!(
            (0.0..=1.0).contains(&avg_mrr),
            "MRR should be in [0,1] for locale {locale}"
        );
        assert!(
            (0.0..=1.0).contains(&avg_ndcg_5),
            "NDCG@5 should be in [0,1] for locale {locale}"
        );
        assert!(
            (0.0..=1.0).contains(&avg_ndcg_10),
            "NDCG@10 should be in [0,1] for locale {locale}"
        );

        // At least some recall (not entirely empty)
        assert!(
            avg_recall > 0.0,
            "should have at least partial recall > 0 for locale {locale}"
        );

        engine.close().unwrap();
    }
}
