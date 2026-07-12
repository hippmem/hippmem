//! acceptance test: Tantivy full-text index and BM25 recall
//!
//! Locale-specific test data lives in `tests/fixtures/bm25/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_store::fulltext::FulltextIndex;
use std::fs;
use tempfile::tempdir;

/// Discover available locale fixtures.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/bm25", env!("CARGO_MANIFEST_DIR"));
    let mut locales = vec![];
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".json") {
                locales.push(name.trim_end_matches(".json").to_string());
            }
        }
    }
    locales.sort();
    if locales.is_empty() {
        panic!("no locale fixtures found in bm25/");
    }
    locales
}

/// Load BM25 fixture for a specific locale.
fn load_fixture(locale: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/bm25/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = fs::read_to_string(&path).expect("failed to read fixture");
    serde_json::from_str(&data).expect("invalid fixture")
}

/// With a fixed English corpus, BM25 query results are deterministic and consistently ordered.
#[test]
fn bm25_english_deterministic() {
    let dir = tempdir().expect("temp dir");
    let index_path = dir.path().join("fulltext");
    let mut idx = FulltextIndex::create(&index_path).expect("create index");

    // Index three documents
    idx.add_document(1u128, "RocksDB is a key-value store", "en")
        .expect("add document 1");
    idx.add_document(2u128, "redb is a pure Rust database", "en")
        .expect("add document 2");
    idx.add_document(3u128, "Tantivy is a full-text search library", "en")
        .expect("add document 3");

    idx.commit().expect("commit");

    // Query "database store"
    let results = idx.search("database store", 5).expect("search");
    // Both relevant documents should appear in the results (BM25 scoring guarantees determinism but not exact order)
    assert!(results.len() >= 2, "should have at least 2 results");
    let ids: Vec<u128> = results.iter().map(|r| r.0).collect();
    assert!(ids.contains(&1u128), "RocksDB should be in the results");
    assert!(ids.contains(&2u128), "redb should be in the results");
    assert!(!ids.contains(&3u128), "Tantivy document should not match");
    // Score should be greater than zero
    assert!(results[0].1 > 0.0);
}

/// CJK tokenization: BM25 recall is deterministic across all active locales.
///
/// Adding a new locale requires: registering a tokenizer for the locale +
/// adding a fixture file with documents + queries. No test code changes needed.
#[test]
fn bm25_cjk_deterministic() {
    for locale in discover_fixture_locales() {
        let dir = tempdir().expect("temp dir");
        let index_path = dir.path().join("fulltext");
        let mut idx = FulltextIndex::create(&index_path).expect("create index");

        let fixture = load_fixture(&locale);

        // Index documents with the correct locale tag
        for doc in fixture["documents"].as_array().unwrap() {
            let id = doc["id"].as_u64().unwrap() as u128;
            let text = doc["text"].as_str().unwrap();
            idx.add_document(id, text, &locale).expect("add");
        }

        // ── Architectural placeholders for future locales (ja, ko) ──
        // When a new locale's tokenizer is registered, adding its fixture file
        // is sufficient — the locale loop above will automatically index and test it.

        idx.commit().expect("commit");

        // Query with the locale-specific search term
        let query = fixture["queries"][0].as_str().unwrap();
        let results = idx.search(query, 5).expect("search");
        assert!(!results.is_empty(), "[{locale}] should have results");
        assert_eq!(
            results[0].0, 20u128,
            "[{locale}] Tantivy document should rank first"
        );
    } // end locale loop
}

/// Search results include BM25 scores.
#[test]
fn bm25_returns_scores() {
    let dir = tempdir().expect("temp dir");
    let index_path = dir.path().join("fulltext");
    let mut idx = FulltextIndex::create(&index_path).expect("create index");

    idx.add_document(1u128, "hello world", "en").expect("add");
    idx.commit().expect("commit");

    let results = idx.search("hello", 10).expect("search");
    assert_eq!(results.len(), 1);
    assert!(results[0].1 > 0.0, "score should be greater than zero");
}

/// Querying an empty corpus returns empty.
#[test]
fn bm25_empty_index_returns_empty() {
    let dir = tempdir().expect("temp dir");
    let index_path = dir.path().join("fulltext");
    let idx = FulltextIndex::create(&index_path).expect("create index");

    let results = idx.search("anything", 10).expect("search");
    assert!(results.is_empty());
}

/// Deterministic ordering for a single-word corpus: the document with more
/// occurrences of the query word ranks first.
#[test]
fn bm25_deterministic_single_word() {
    let dir = tempdir().expect("temp dir");
    let index_path = dir.path().join("fulltext");
    let mut idx = FulltextIndex::create(&index_path).expect("create index");

    // doc 100: "rust" appears 3 times
    idx.add_document(100u128, "rust rust rust", "en")
        .expect("add");
    // doc 200: "rust" appears 1 time
    idx.add_document(200u128, "rust is nice", "en")
        .expect("add");

    idx.commit().expect("commit");

    let results = idx.search("rust", 5).expect("search");
    assert_eq!(results.len(), 2);
    // The document with more occurrences of the word gets a higher BM25 score
    assert_eq!(
        results[0].0, 100u128,
        "the document with 'rust' 3 times should rank first"
    );
    assert_eq!(results[1].0, 200u128);
}

/// Reopens an existing index and queries it.
#[test]
fn bm25_reopen_and_search() {
    let dir = tempdir().expect("temp dir");
    let index_path = dir.path().join("fulltext");

    {
        let mut idx = FulltextIndex::create(&index_path).expect("create");
        idx.add_document(42u128, "persistent test document", "en")
            .expect("add");
        idx.commit().expect("commit");
    }

    {
        let idx = FulltextIndex::open(&index_path).expect("reopen");
        let results = idx.search("persistent", 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42u128);
    }
}
