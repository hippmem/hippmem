//! Deterministic summarizer: extractive summary — keyword coverage + position scoring (08 §4.4).

use crate::error::ModelResult;
use crate::traits::{SummarizeInput, Summarizer, SummaryOutput};
use hippmem_core::score::UnitScore;
use std::collections::HashSet;

/// Deterministic summarizer: extractive summary, no generation, no hallucination.
#[derive(Default)]
pub struct DeterministicSummarizer;

impl DeterministicSummarizer {
    /// Synchronous version (for tests).
    pub fn summarize_sync(&self, sources: &[SummarizeInput]) -> ModelResult<SummaryOutput> {
        let all_text: String = sources
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let sentences = split_sentences(&all_text);

        // Extract all keywords
        let all_tokens = hippmem_core::hash::tokenize(&all_text, "zh");
        let keywords: HashSet<&str> = all_tokens
            .iter()
            .filter(|t| t.len() >= 2)
            .take(20)
            .map(|t| t.as_str())
            .collect();

        // Score sentences (keyword coverage + position weight)
        let n = sentences.len().max(1);
        let mut scored: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let toks = hippmem_core::hash::tokenize(s, "zh");
                let hits = toks
                    .iter()
                    .filter(|t| keywords.contains(t.as_str()))
                    .count();
                let pos_weight = 1.0 - (i as f32 / n as f32) * 0.5; // earlier sentences weigh more
                let score = hits as f32 * pos_weight;
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take the top 3 sentences and concatenate
        let top_count = 3.min(sentences.len());
        let summary = scored[..top_count]
            .iter()
            .map(|(i, _)| sentences[*i].as_str())
            .collect::<Vec<_>>()
            .join("");

        let covers: Vec<_> = sources.iter().map(|s| s.id).collect();

        // Confidence is based on the number of sources: more sources -> higher
        // confidence, capped at 0.6
        //   - 2 sources -> 0.10 (< 0.35, gated by the caller via confidence threshold)
        //   - 7 sources -> 0.35 (= threshold, just passes)
        //   - 12 sources -> 0.60 (minimum threshold that triggers a Consolidation summary)
        let base_conf = (sources.len() as f32 * 0.05).min(0.6);
        let confidence = UnitScore::new(base_conf);

        Ok(SummaryOutput {
            summary,
            covers,
            confidence,
        })
    }
}

#[async_trait::async_trait]
impl Summarizer for DeterministicSummarizer {
    async fn summarize(&self, sources: &[SummarizeInput]) -> ModelResult<SummaryOutput> {
        self.summarize_sync(sources)
    }

    fn backend_id(&self) -> &str {
        "deterministic-extractive"
    }
}

/// Simple sentence splitter: splits on Chinese and English punctuation.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?' | '\n') {
            if cur.trim().len() > 1 {
                sentences.push(cur.trim().to_string());
            }
            cur.clear();
        }
    }
    if cur.trim().len() > 1 {
        sentences.push(cur.trim().to_string());
    }
    if sentences.is_empty() {
        sentences.push(text.to_string());
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::ids::MemoryId;

    #[test]
    fn produces_non_empty_summary() {
        let s = DeterministicSummarizer;
        let sources = vec![
            SummarizeInput {
                id: MemoryId(1),
                text: "Rust is a systems programming language. It is safe.".into(),
            },
            SummarizeInput {
                id: MemoryId(2),
                text: "Today I learned about Rust's ownership system.".into(),
            },
        ];
        let out = s.summarize_sync(&sources).unwrap();
        assert!(!out.summary.is_empty());
        assert_eq!(out.covers.len(), 2);
        assert!(out.confidence.value() <= 0.5);
    }
}
