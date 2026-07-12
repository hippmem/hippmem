//! acceptance test: DeterministicSummarizer
//!
//! Locale-specific test texts live in `tests/fixtures/summarize/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_core::ids::MemoryId;
use hippmem_model::deterministic::summarize::DeterministicSummarizer;
use hippmem_model::traits::{SummarizeInput, Summarizer};
use std::fs;

/// Discover available locale fixtures.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/summarize", env!("CARGO_MANIFEST_DIR"));
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
        panic!("no locale fixtures found in summarize/");
    }
    locales
}

/// Load summarize texts for a specific locale.
fn load_texts(locale: &str) -> Vec<String> {
    let path = format!(
        "{}/tests/fixtures/summarize/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = fs::read_to_string(&path).expect("failed to read fixture");
    let fixture: serde_json::Value = serde_json::from_str(&data).expect("invalid fixture");
    fixture["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

/// Produces a non-empty summary text. All locales.
#[test]
fn produces_non_empty_summary() {
    for locale in discover_fixture_locales() {
        let s = DeterministicSummarizer;
        let texts = load_texts(&locale);
        let sources: Vec<SummarizeInput> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| SummarizeInput {
                id: MemoryId(i as u128 + 1),
                text: t.clone(),
            })
            .collect();
        let out = s.summarize_sync(&sources).unwrap();
        assert!(
            !out.summary.is_empty(),
            "[{locale}] summary should not be empty"
        );
    }
}

/// covers contains all input ids.
#[test]
fn covers_all_input_ids() {
    let s = DeterministicSummarizer;
    let sources = vec![
        SummarizeInput {
            id: MemoryId(1),
            text: "A B C".into(),
        },
        SummarizeInput {
            id: MemoryId(2),
            text: "D E F".into(),
        },
        SummarizeInput {
            id: MemoryId(3),
            text: "G H I".into(),
        },
    ];
    let out = s.summarize_sync(&sources).unwrap();
    assert_eq!(out.covers.len(), 3);
    assert!(out.confidence.value() <= 0.5);
}

/// backend_id is correct.
#[test]
fn backend_id_correct() {
    let s = DeterministicSummarizer;
    assert_eq!(s.backend_id(), "deterministic-extractive");
}
