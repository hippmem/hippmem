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
    /// Embedder backend identifier (e.g. "neural-text-embedding-3-small").
    pub embedder: String,
    /// Reranker backend identifier (None means rerank is not enabled).
    pub reranker: Option<String>,
}

// ── Extractor factory (08 §5) ──

/// Build the corresponding `Extractor` implementation from the backend choice.
///
/// - `Deterministic` -> rule-based, always available, offline.
/// - `Api` -> Anthropic Claude structured extraction (requires
///   `ANTHROPIC_API_KEY`; fails fast when missing).
/// - `Auto` -> Api when `ANTHROPIC_API_KEY` is present, Deterministic
///   otherwise (the "default enhancement, degraded guarantee" semantics of
///   08 §1 — strongest effect when a key exists, offline otherwise).
pub fn build_extractor(choice: BackendChoice) -> Result<std::sync::Arc<dyn Extractor>, String> {
    use crate::api::openai_extract::OpenAiExtractor;
    use crate::deterministic::extract::DeterministicExtractor;
    match choice {
        BackendChoice::Deterministic => Ok(std::sync::Arc::new(DeterministicExtractor)),
        // Vendor-neutral API backend: OpenAI-compatible chat-completions
        // service configured via HIPPMEM_EXTRACTOR_* (fallback OPENAI_API_KEY).
        BackendChoice::Api => {
            let extractor =
                OpenAiExtractor::from_env().map_err(|e| format!("extractor Api backend: {e}"))?;
            Ok(std::sync::Arc::new(extractor))
        }
        BackendChoice::Auto => {
            let has_key = !std::env::var("HIPPMEM_EXTRACTOR_API_KEY")
                .ok()
                .filter(|k| !k.is_empty())
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .unwrap_or_default()
                .is_empty();
            if has_key {
                let extractor = OpenAiExtractor::from_env()
                    .map_err(|e| format!("extractor Auto backend: {e}"))?;
                Ok(std::sync::Arc::new(extractor))
            } else {
                Ok(std::sync::Arc::new(DeterministicExtractor))
            }
        }
    }
}

// ── Embedder factory functions (V4) ──

use hippmem_core::config::{default_embed_dim, default_embedder_base_url, EmbedderConfig};

/// Build the corresponding `Embedder` implementation from the config.
///
/// - `Hash` -> always available, pure computation, offline.
/// - `Neural` -> always compiled (no feature flag), requires API key or
///   `OPENAI_API_KEY` env var.
/// - `Onnx` -> reserved, currently always returns `Unavailable`.
pub fn build_embedder(
    config: &EmbedderConfig,
) -> crate::error::ModelResult<std::sync::Arc<dyn crate::traits::Embedder>> {
    // Auto: strongest-first default (2026-08-26) — neural when a key is
    // present, deterministic hash otherwise (no-key behavior unchanged).
    if matches!(config, EmbedderConfig::Auto) {
        let has_key = !std::env::var("OPENAI_API_KEY")
            .unwrap_or_default()
            .is_empty();
        if has_key {
            let base_url = std::env::var("HIPPMEM_EMBEDDING_BASE_URL")
                .unwrap_or_else(|_| default_embedder_base_url());
            let model = std::env::var("HIPPMEM_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            return build_embedder(&EmbedderConfig::Neural {
                base_url,
                model,
                api_key: None,
                dimensions: default_embed_dim(),
            });
        }
        return build_embedder(&EmbedderConfig::Hash {
            dimensions: default_embed_dim(),
        });
    }
    match config {
        EmbedderConfig::Auto => unreachable!("handled above"),
        EmbedderConfig::Hash { dimensions } => Ok(std::sync::Arc::new(
            crate::deterministic::embed::DeterministicEmbedder::new(*dimensions),
        )),
        EmbedderConfig::Neural {
            base_url,
            model,
            api_key,
            dimensions,
        } => {
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
    fn build_embedder_hash_explicit() {
        let cfg = EmbedderConfig::Hash { dimensions: 256 };
        let embedder = build_embedder(&cfg).unwrap();
        assert_eq!(embedder.dim(), 256);
        assert_eq!(embedder.backend_id(), "deterministic-hash");
    }

    /// Auto is the strongest-first default (D13): neural when
    /// OPENAI_API_KEY is present, deterministic hash otherwise.
    #[test]
    fn build_embedder_auto_semantics() {
        let _guard = crate::test_env::ENV_LOCK.lock().unwrap();
        let prev = std::env::var("OPENAI_API_KEY").ok();
        std::env::remove_var("OPENAI_API_KEY");

        let cfg = EmbedderConfig::Auto;
        let hash_embedder = build_embedder(&cfg).unwrap();
        assert_eq!(hash_embedder.backend_id(), "deterministic-hash");

        std::env::set_var("OPENAI_API_KEY", "sk-test");
        let neural_embedder = build_embedder(&cfg).unwrap();
        assert_eq!(neural_embedder.backend_id(), "text-embedding-3-small");

        match prev {
            Some(v) => std::env::set_var("OPENAI_API_KEY", v),
            None => std::env::remove_var("OPENAI_API_KEY"),
        }
    }

    #[test]
    fn build_embedder_hash_custom_dim() {
        let cfg = EmbedderConfig::Hash { dimensions: 512 };
        let embedder = build_embedder(&cfg).unwrap();
        assert_eq!(embedder.dim(), 512);
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
