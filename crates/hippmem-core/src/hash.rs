//! Stable hashing and tokenization utilities.
//!
//! Corresponds to ADR-018 (tokenization), ADR-019 (xxh3 64-bit stable hash), 03#1 (signature generation basis).

use jieba_rs::Jieba;
use std::sync::LazyLock;

/// Global Jieba instance (lazy-loaded, shared within the process).
/// Avoids re-initializing the dictionary and HMM data on every tokenization (~200ms → ~1ms).
pub static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

/// Computes a stable xxh3 64-bit hash of the text (fixed seed 0).
///
/// **Deterministic**: the same text always yields the same hash across processes and versions.
/// The algorithm and seed are permanently fixed (ADR-019); changing them requires a schema migration + ADR.
pub fn stable_hash64(text: &str) -> u64 {
    xxhash_rust::xxh3::xxh3_64(text.as_bytes())
}

/// Tokenizes by language: Chinese goes through jieba; English/code goes through whitespace + punctuation splitting and lowercasing.
///
/// `language` takes values such as `"zh"`, `"zh-CN"`, `"en"`, `"ja"`, etc.
/// Unknown languages fall back to the English tokenization strategy.
pub fn tokenize(text: &str, language: &str) -> Vec<String> {
    match language {
        "zh" | "zh-CN" | "zh-TW" | "zh-HK" => tokenize_chinese(text),
        _ => tokenize_english(text),
    }
}

/// Tokenizes Chinese text with jieba-rs (HMM enabled).
/// Uses the global JIEBA instance to avoid repeated initialization.
fn tokenize_chinese(text: &str) -> Vec<String> {
    JIEBA
        .cut(text, true) // true = enable HMM new-word discovery
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// jieba part-of-speech tagging (uses the global JIEBA instance).
pub fn tag_chinese(text: &str) -> Vec<(String, String)> {
    JIEBA
        .tag(text, true)
        .into_iter()
        .map(|t| (t.word.to_string(), t.tag.to_string()))
        .collect()
}

/// English/code tokenization: splits on whitespace + ASCII punctuation, lowercases, filters empties.
fn tokenize_english(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_hash_deterministic() {
        let h1 = stable_hash64("test text");
        let h2 = stable_hash64("test text");
        assert_eq!(h1, h2);
    }
}
