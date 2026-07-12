//! Tokenization tests for HIPPMEM's multilingual NLP pipeline.
//!
//! HIPPMEM supports multiple languages through locale-parametrized tokenizers:
//! - zh (Chinese): jieba segmentation
//! - ja (Japanese): planned — MeCab / Sudachi
//! - ko (Korean): planned — Mecab-ko
//!
//! Adding a new language requires: registering a tokenizer + adding a
//! locale-tagged fixture at `tests/fixtures/tokenization/<locale>.json`.
//! No test code changes needed.

use hippmem_core::hash::{stable_hash64, tokenize};
use serde::Deserialize;

// ── Fixture types ──

#[derive(Deserialize)]
struct TokenizationCase {
    label: String,
    text: String,
    #[serde(default)]
    expected: Vec<String>,
    #[serde(default)]
    expected_any: Vec<String>,
}

#[derive(Deserialize)]
struct TokenizationFixture {
    cases: Vec<TokenizationCase>,
}

/// Discover available tokenization fixtures by globbing the fixtures directory.
fn discover_tokenization_locales() -> Vec<String> {
    let dir_path = format!("{}/tests/fixtures/tokenization", env!("CARGO_MANIFEST_DIR"));
    let dir = match std::fs::read_dir(&dir_path) {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    let mut locales = vec![];
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".json") {
            locales.push(name.trim_end_matches(".json").to_string());
        }
    }
    locales.sort();
    locales
}

fn load_tokenization_fixture(locale: &str) -> Vec<TokenizationCase> {
    let fixture_path = format!(
        "{}/tests/fixtures/tokenization/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", fixture_path, e));
    let fixture: TokenizationFixture = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("invalid fixture {}: {}", fixture_path, e));
    fixture.cases
}

// ── stable_hash64 tests ──

#[test]
fn stable_hash64_same_text_same_hash() {
    let text = "Hello, HIPPMEM!";
    let h1 = stable_hash64(text);
    let h2 = stable_hash64(text);
    let h3 = stable_hash64(text);
    assert_eq!(
        h1, h2,
        "hashing the same text multiple times should be identical"
    );
    assert_eq!(
        h2, h3,
        "hashing the same text multiple times should be identical"
    );
}

#[test]
fn stable_hash64_different_text_different_hash() {
    let h1 = stable_hash64("hello world");
    let h2 = stable_hash64("hello worlb"); // one character different
    assert_ne!(h1, h2, "different text should produce different hashes");
}

#[test]
fn stable_hash64_empty_string() {
    let _ = stable_hash64("");
    // An empty string should not panic (reaching this point means it passed)
}

#[test]
fn stable_hash64_non_empty_always_produces_value() {
    let h = stable_hash64("a");
    assert!(
        h > 0,
        "non-empty text hash should be non-zero (with overwhelming probability)"
    );
}

// ── tokenize tests ──

/// Locale-driven tokenization test: discovers and loads cases from
/// `tests/fixtures/tokenization_<locale>.json` for each locale that has a fixture.
/// Adding a language = adding its fixture file. No test code changes needed.
#[test]
fn tokenize_locale_fixture_driven() {
    let locales = discover_tokenization_locales();
    assert!(
        !locales.is_empty(),
        "should have at least one tokenization fixture"
    );
    for locale in &locales {
        let cases = load_tokenization_fixture(locale);
        for case in &cases {
            let tokens = tokenize(&case.text, locale);
            assert!(
                !tokens.is_empty(),
                "[{}] '{}': tokenization should not be empty",
                locale,
                case.label
            );
            for expected in &case.expected {
                assert!(
                    tokens.contains(&expected.to_string()),
                    "[{}] '{}': should contain '{}', got {:?}",
                    locale,
                    case.label,
                    expected,
                    tokens
                );
            }
            if !case.expected_any.is_empty() {
                let hit = case
                    .expected_any
                    .iter()
                    .any(|e| tokens.contains(&e.to_string()));
                assert!(
                    hit,
                    "[{}] '{}': should contain at least one of {:?}, got {:?}",
                    locale, case.label, case.expected_any, tokens
                );
            }
        }
    }
}

#[test]
fn tokenize_english_basic() {
    let tokens = tokenize("hello world from HIPPMEM", "en");
    assert!(
        !tokens.is_empty(),
        "English tokenization should not be empty"
    );
    assert!(tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
    assert!(
        tokens.contains(&"hippmem".to_string()),
        "should be lowercased"
    );
}

#[test]
fn tokenize_english_punctuation_handling() {
    let tokens = tokenize("hello, world! how are you?", "en");
    assert!(!tokens.is_empty());
    // Commas and exclamation marks should not be part of any token
    assert!(
        !tokens.contains(&",".to_string()),
        "punctuation should not become a standalone token"
    );
    assert!(tokens.contains(&"hello".to_string()));
}

#[test]
fn tokenize_returns_lowercase_for_english() {
    let tokens = tokenize("Rust Memory Engine", "en");
    assert!(
        tokens.iter().any(|t| t == "rust"),
        "English should be lowercased: {:?}",
        tokens
    );
    assert!(
        !tokens.contains(&"Rust".to_string()),
        "English should not retain uppercase: {:?}",
        tokens
    );
}
