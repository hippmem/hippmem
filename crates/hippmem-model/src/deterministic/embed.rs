//! Deterministic embedder: feature-hash embedding (08 §4.1).
//!
//! Performs feature-hash embedding on the text — after tokenization each token
//! is projected to dim dimensions via multiple independent hashes, accumulated,
//! then L2-normalized. The same text always yields the same vector.

use crate::error::ModelResult;
use crate::traits::Embedder;
use xxhash_rust::xxh3::xxh3_64;

/// Default embedding dimension (08 §4.1).
pub const DEFAULT_EMBED_DIM: usize = 256;

/// Deterministic embedder: pure computation, no network, no randomness.
///
/// Implements the `Embedder` trait, used in CI and offline environments.
pub struct DeterministicEmbedder {
    dim: usize,
}

impl Default for DeterministicEmbedder {
    fn default() -> Self {
        Self {
            dim: DEFAULT_EMBED_DIM,
        }
    }
}

impl DeterministicEmbedder {
    /// Create an embedder with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Synchronous embedding (back-compat, delegates to the trait method).
    pub fn embed_sync(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>> {
        <Self as Embedder>::embed_sync(self, texts)
    }

    /// Generate an embedding vector for a single text.
    fn embed_one(&self, text: &str) -> Vec<f32> {
        let tokens = hippmem_core::hash::tokenize(text, "zh");
        let tokens_en = hippmem_core::hash::tokenize(text, "en");

        let mut acc = vec![0.0f32; self.dim];

        // Merge Chinese/English tokenization results and accumulate
        for token in tokens.iter().chain(tokens_en.iter()) {
            if token.is_empty() {
                continue;
            }
            let base = xxh3_64(token.as_bytes());
            for (d, acc_val) in acc.iter_mut().enumerate().take(self.dim) {
                // Use (base ^ d) as the hash for this dimension
                let mut input = Vec::with_capacity(16);
                input.extend_from_slice(&base.to_le_bytes());
                input.extend_from_slice(&(d as u64).to_le_bytes());
                let dim_hash = xxh3_64(&input);
                // Map to [-1, 1]
                let val = ((dim_hash % 2000u64) as f32 - 1000.0) / 1000.0;
                *acc_val += val;
            }
        }

        // All-zero text (only stopwords or empty string) -> return zero vector
        let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return acc; // zero vector
        }

        // L2 normalization
        for v in &mut acc {
            *v /= norm;
        }
        acc
    }
}

#[async_trait::async_trait]
impl Embedder for DeterministicEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn embed_sync(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn backend_id(&self) -> &str {
        "deterministic-hash"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_implements_default() {
        let e = DeterministicEmbedder::default();
        assert_eq!(e.dim(), 256);
    }

    #[test]
    fn embedder_deterministic() {
        let e = DeterministicEmbedder::default();
        let texts: Vec<String> = vec!["hello world".into(), "rust".into()];
        let v1 = e.embed_sync(&texts).unwrap();
        let v2 = e.embed_sync(&texts).unwrap();
        for (a, b) in v1.iter().zip(v2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-6);
            }
        }
    }
}
