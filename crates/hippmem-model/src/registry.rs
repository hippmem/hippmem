//! Model registry and backend selection: 08 §5.

use crate::traits::{Embedder, Extractor, Reranker, Summarizer};
use std::sync::Arc;

/// Assembled four-model handle, held by `hippmem-engine`.
///
/// Upper-layer algorithms depend only on traits, not on concrete implementations.
pub struct ModelRegistry {
    /// Embedder.
    pub embedder: Arc<dyn Embedder>,
    /// Extractor.
    pub extractor: Arc<dyn Extractor>,
    /// Reranker.
    pub reranker: Arc<dyn Reranker>,
    /// Summarizer.
    pub summarizer: Arc<dyn Summarizer>,
}

/// Backend choice for each capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// Use the API backend (requires network + key).
    Api,
    /// Use the deterministic fallback backend.
    Deterministic,
    /// Auto: use Api if a key is present, otherwise fall back to Deterministic.
    Auto,
}

/// Backend selection for the three non-Embedder capabilities (see 08 §5).
///
/// Embedder selection has migrated to `EmbedderConfig`; see
/// `hippmem_core::config::EmbedderConfig`.
#[derive(Debug, Clone)]
pub struct BackendSelection {
    /// Extractor backend.
    pub extractor: BackendChoice,
    /// Reranker backend.
    pub reranker: BackendChoice,
    /// Summarizer backend.
    pub summarizer: BackendChoice,
}

impl Default for BackendSelection {
    /// Defaults to all `Auto`.
    fn default() -> Self {
        Self {
            extractor: BackendChoice::Auto,
            reranker: BackendChoice::Auto,
            summarizer: BackendChoice::Auto,
        }
    }
}

/// Backend info actually used by a retrieval (written to
/// `RetrievalDiagnostics.backend_used`).
#[derive(Debug, Clone)]
pub struct BackendUsage {
    /// Embedder backend identifier (e.g. "openai-text-embedding-3-small").
    pub embedder: String,
    /// Reranker backend identifier (None means rerank is not enabled).
    pub reranker: Option<String>,
}

// ── Embedder factory functions (V4) ──

use hippmem_core::config::EmbedderConfig;

/// Build the corresponding `Embedder` implementation from the config.
///
/// - `Deterministic` -> always available, pure computation, zero dependencies.
/// - `OpenAiCompatible` -> requires the `api-backends` feature, otherwise returns `Unavailable`.
/// - `Onnx` -> reserved, currently always returns `Unavailable`.
pub fn build_embedder(
    config: &EmbedderConfig,
) -> crate::error::ModelResult<std::sync::Arc<dyn crate::traits::Embedder>> {
    match config {
        EmbedderConfig::Deterministic { dimensions } => Ok(std::sync::Arc::new(
            crate::deterministic::embed::DeterministicEmbedder::new(*dimensions),
        )),
        EmbedderConfig::OpenAiCompatible {
            base_url,
            model,
            api_key,
            dimensions,
        } => {
            // Suppress cfg-dependent dead_code warnings: some branches may not use all fields
            let _ = (base_url, model, api_key, dimensions);
            #[cfg(feature = "api-backends")]
            {
                let key = match api_key {
                    Some(k) if !k.is_empty() => k.clone(),
                    _ => std::env::var("OPENAI_API_KEY").unwrap_or_default(),
                };
                if key.is_empty() {
                    return Err(crate::error::ModelError::Auth(model.clone()));
                }
                let embedder = crate::api::openai::OpenAiEmbedder::new_with_base_url(
                    key,
                    base_url,
                    model,
                    *dimensions,
                )?;
                Ok(std::sync::Arc::new(embedder))
            }
            #[cfg(not(feature = "api-backends"))]
            {
                Err(crate::error::ModelError::Unavailable(
                    "api-backends feature not enabled".to_string(),
                ))
            }
        }
        EmbedderConfig::Onnx { .. } => Err(crate::error::ModelError::Unavailable(
            "onnx backend not yet implemented".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::config::EmbedderConfig;

    #[test]
    fn build_embedder_deterministic() {
        let cfg = EmbedderConfig::default(); // Deterministic { dimensions: 256 }
        let embedder = build_embedder(&cfg).unwrap();
        assert_eq!(embedder.dim(), 256);
        assert_eq!(embedder.backend_id(), "deterministic-hash");
    }

    #[test]
    fn build_embedder_deterministic_custom_dim() {
        let cfg = EmbedderConfig::Deterministic { dimensions: 512 };
        let embedder = build_embedder(&cfg).unwrap();
        assert_eq!(embedder.dim(), 512);
    }

    #[test]
    #[cfg(not(feature = "api-backends"))]
    fn openai_compatible_requires_feature() {
        let cfg = EmbedderConfig::OpenAiCompatible {
            base_url: "https://api.openai.com/v1".into(),
            model: "text-embedding-3-small".into(),
            api_key: Some("sk-test".into()),
            dimensions: 1536,
        };
        let result = build_embedder(&cfg);
        match &result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(
                    err_msg.contains("api-backends"),
                    "error message should mention the api-backends feature, got: {err_msg}"
                );
            }
            Ok(_) => panic!("should return an error when api-backends is not enabled"),
        }
    }

    #[test]
    fn onnx_returns_unavailable() {
        let cfg = EmbedderConfig::Onnx {
            model_name: "test-model".into(),
            model_cache_dir: std::path::PathBuf::from("/tmp"),
            dimensions: 512,
        };
        let result = build_embedder(&cfg);
        match &result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(
                    err_msg.contains("onnx"),
                    "error message should mention onnx, got: {err_msg}"
                );
            }
            Ok(_) => panic!("should return an error when ONNX is not implemented"),
        }
    }
}
