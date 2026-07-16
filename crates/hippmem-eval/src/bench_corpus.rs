//! Bench corpus format types: Rust deserialization targets for benchmark fixtures.
//!
//! These types correspond to the JSON structure of `fixtures/bench/<locale>/*.json`.
//! Benchmark data uses locale leaf directories: `fixtures/bench/{en,zh}/`.

use hippmem_core::model::ContentType;
use serde::{Deserialize, Serialize};

// ── Dataset (memory entries) ──

/// A benchmark memory dataset: a list of entries to write into the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchDataset {
    /// Dataset entries, written in order.
    pub entries: Vec<BenchEntry>,
}

/// One memory entry in a benchmark dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchEntry {
    /// Content type of the memory.
    pub content_type: ContentType,
    /// Raw content text of the memory.
    pub content: String,
    /// Importance hint (0.0–1.0).
    pub importance: f32,
    /// Category label used to match queries to expected hits (e.g. `"tech_decisions"`).
    pub category: String,
}

// ── Query set: category-matching form ──

/// A set of benchmark queries that expect hits in given categories.
///
/// Used by `bench_mem0_comparison`, `integ_quick_bench`, and `integ_retrieval_quality`:
/// each query is checked by whether the returned memories fall into the
/// expected categories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryQuerySet {
    /// Queries in the set.
    pub queries: Vec<CategoryQuery>,
}

/// A query expecting hits in specific categories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryQuery {
    /// Query text.
    pub query: String,
    /// Category labels the top results should belong to.
    pub expected_categories: Vec<String>,
    /// Human-readable description of what the query tests.
    pub description: String,
}

// ── Category text set: (category, [texts]) form ──

/// A set of category-labeled text groups, used by benchmarks that write
/// memories in fixed category buckets without specifying content_type/importance
/// (e.g. `retrieval_quality`).
///
/// Each group contributes N memories sharing a category label; queries later
/// match against the category to compute precision/recall/MRR/NDCG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryTextSet {
    /// Category groups, written in order.
    pub categories: Vec<CategoryTexts>,
}

/// A category group: a label plus the texts that belong to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryTexts {
    /// Category label used to match queries to expected hits.
    pub category: String,
    /// Texts in this category; each becomes one memory.
    pub texts: Vec<String>,
}

// ── Query set: keyword-matching form ──

/// A set of benchmark queries that expect a specific keyword in the results.
///
/// Used by `comprehensive_100`: each query is checked by whether any returned
/// memory contains the expected keyword.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeywordQuerySet {
    /// Queries in the set.
    pub queries: Vec<KeywordQuery>,
}

/// A query expecting a specific keyword in at least `min_results` returned memories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeywordQuery {
    /// Query text.
    pub query: String,
    /// Keyword that should appear in the returned memory content.
    pub expected_keyword: String,
    /// Minimum number of results expected to match.
    pub min_results: usize,
}

// ── Loader ──

/// Loads a typed bench fixture. Delegates to [`crate::fixture_loader::load_bench_fixture`]
/// which is the single source of truth for the `fixtures/bench/` path.
pub fn load_fixture<T: serde::de::DeserializeOwned>(locale: &str, name: &str) -> T {
    crate::fixture_loader::load_bench_fixture(locale, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BenchEntry round-trips through JSON with ContentType in PascalCase form.
    #[test]
    fn bench_entry_roundtrip() {
        let entry = BenchEntry {
            content_type: ContentType::Decision,
            content: "decide to use redb".into(),
            importance: 0.8,
            category: "tech_decisions".into(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(
            json.contains("\"content_type\":\"Decision\""),
            "json was: {json}"
        );
        let rt: BenchEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, rt);
    }
}
