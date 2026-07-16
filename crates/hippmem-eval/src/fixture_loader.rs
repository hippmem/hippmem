//! Locale-aware fixture discovery and loading.
//!
//! Shared between library tests and examples. The three discovery functions
//! glob the fixtures directory at runtime, so adding a new locale requires
//! only creating the fixture file — no code changes.

use std::fs;
use std::path::PathBuf;

// ── Path helpers (single source of truth for all fixture paths) ──

fn fixtures_base() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

fn bench_dir() -> PathBuf {
    fixtures_base().join("bench")
}

fn corpus_dir() -> PathBuf {
    fixtures_base().join("corpus")
}

fn test_fixtures_dir(purpose: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(purpose)
}

// ── Discovery ──

/// Discover available locales by listing subdirectories under `fixtures/bench/`.
pub fn discover_bench_locales() -> Vec<String> {
    let dir = bench_dir();
    let mut locales = vec![];
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    locales.push(name);
                }
            }
        }
    }
    locales.sort();
    if locales.is_empty() {
        panic!("no locale directories found in {}", dir.display());
    }
    locales
}

/// Discover available locales by listing *.json files in `tests/fixtures/<purpose>/`.
pub fn discover_test_locales(purpose: &str) -> Vec<String> {
    let dir = test_fixtures_dir(purpose);
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
        panic!("no locale fixtures found in {}", dir.display());
    }
    locales
}

/// Discover available locales by listing subdirectories under `fixtures/corpus/`.
pub fn discover_corpus_locales() -> Vec<String> {
    let dir = corpus_dir();
    let mut locales = vec![];
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    locales.push(name);
                }
            }
        }
    }
    locales.sort();
    if locales.is_empty() {
        panic!("no locale directories found in {}", dir.display());
    }
    locales
}

// ── Loading ──

/// Load a test fixture for a given purpose and locale.
pub fn load_test_fixture(purpose: &str, locale: &str) -> serde_json::Value {
    let path = test_fixtures_dir(purpose).join(format!("{locale}.json"));
    let data = fs::read_to_string(&path).expect("failed to read test fixture");
    serde_json::from_str(&data).expect("invalid test fixture")
}

/// Load an eval corpus case for a given locale and case name.
pub fn load_corpus_case(locale: &str, name: &str) -> String {
    let path = corpus_dir().join(locale).join(format!("{name}.json"));
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read corpus case {}: {e}", path.display()))
}

/// Load a typed bench fixture from `fixtures/bench/{locale}/{name}.json`.
///
/// This is the single source of truth for the bench fixture path. Callers that
/// need typed deserialization (e.g. `BenchDataset`, `CategoryQuerySet`) use this;
/// raw JSON loading goes through the other loaders above.
pub fn load_bench_fixture<T: serde::de::DeserializeOwned>(locale: &str, name: &str) -> T {
    let path = bench_dir().join(locale).join(format!("{name}.json"));
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read bench fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("cannot parse bench fixture {name}: {e}"))
}
