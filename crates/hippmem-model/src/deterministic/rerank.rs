//! Deterministic reranker: BM25-style term-overlap scoring (08 §4.3).

use crate::error::ModelResult;
use crate::traits::Reranker;
use std::collections::HashMap;

/// Deterministic reranker: scores by query-candidate term overlap.
#[derive(Default)]
pub struct DeterministicReranker;

impl DeterministicReranker {
    /// Synchronous version (for tests).
    pub fn rerank_sync(&self, query: &str, candidates: &[String]) -> ModelResult<Vec<f32>> {
        let q_tokens: Vec<String> = tokenize_all(query);
        if q_tokens.is_empty() {
            return Ok(vec![0.0; candidates.len()]);
        }
        Ok(candidates
            .iter()
            .map(|c| overlap_score(&q_tokens, &tokenize_all(c)))
            .collect())
    }
}

#[async_trait::async_trait]
impl Reranker for DeterministicReranker {
    async fn rerank(&self, query: &str, candidates: &[String]) -> ModelResult<Vec<f32>> {
        self.rerank_sync(query, candidates)
    }

    fn backend_id(&self) -> &str {
        "deterministic-bm25-overlap"
    }
}

fn tokenize_all(text: &str) -> Vec<String> {
    let mut t = hippmem_core::hash::tokenize(text, "zh");
    t.extend(hippmem_core::hash::tokenize(text, "en"));
    t.sort();
    t.dedup();
    t
}

fn overlap_score(query_tokens: &[String], doc_tokens: &[String]) -> f32 {
    let q_set: HashMap<&str, usize> = query_tokens.iter().map(|t| (t.as_str(), 1)).collect();
    let overlap: usize = doc_tokens
        .iter()
        .filter(|t| q_set.contains_key(t.as_str()))
        .count();
    // Jaccard-like: overlap / (query.len + doc.len - overlap)
    let denom = query_tokens.len() + doc_tokens.len() - overlap;
    if denom == 0 {
        0.0
    } else {
        overlap as f32 / denom.max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_candidate_scores_higher() {
        let r = DeterministicReranker;
        let s = r
            .rerank_sync(
                "rust programming",
                &["hello world".into(), "rust is great".into()],
            )
            .unwrap();
        assert!(s[1] > s[0]);
    }
}
