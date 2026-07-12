//! OpenAI Embedder client (08 §3).
//!
//! Default model: text-embedding-3-small (dim=1536).

use crate::error::{ModelError, ModelResult};
use crate::traits::Embedder;

/// OpenAI-compatible embedder: calls any compatible service via an
/// OpenAI-compatible REST API.
///
/// Supports OpenAI / DashScope / vLLM and other services implementing the
/// OpenAI API format.
pub struct OpenAiEmbedder {
    api_key: String,
    base_url: String,
    model: String,
    dim: usize,
    client: reqwest::Client,
}

impl OpenAiEmbedder {
    /// Defaults: text-embedding-3-small, dim=1536, base_url=OpenAI.
    pub fn new(api_key: String) -> Self {
        Self::new_with_base_url(
            api_key,
            "https://api.openai.com/v1",
            "text-embedding-3-small",
            1536,
        )
        .expect("new() is given an explicit key, so it won't trigger an Auth error")
    }

    /// Custom model and dimension (base_url uses the default OpenAI endpoint).
    pub fn with_model(api_key: String, model: &str, dim: usize) -> Self {
        Self::new_with_base_url(api_key, "https://api.openai.com/v1", model, dim)
            .expect("with_model() is given an explicit key, so it won't trigger an Auth error")
    }

    /// Full constructor: specify api_key, base_url, model, dim.
    ///
    /// If `api_key` is an empty string, it is read from the `OPENAI_API_KEY`
    /// environment variable.
    /// If neither is present, returns `ModelError::Auth`.
    pub fn new_with_base_url(
        api_key: String,
        base_url: &str,
        model: &str,
        dim: usize,
    ) -> ModelResult<Self> {
        let api_key = if api_key.is_empty() {
            std::env::var("OPENAI_API_KEY").unwrap_or_default()
        } else {
            api_key
        };
        if api_key.is_empty() {
            return Err(ModelError::Auth(model.to_string()));
        }
        Ok(Self {
            api_key,
            base_url: base_url.to_string(),
            model: model.to_string(),
            dim,
            client: reqwest::Client::new(),
        })
    }

    /// Synchronous embedding (back-compat, delegates to the trait method).
    pub fn embed_sync(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>> {
        <Self as Embedder>::embed_sync(self, texts)
    }
}

#[async_trait::async_trait]
impl Embedder for OpenAiEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let url = format!("{}/embeddings", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| ModelError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            return match status {
                401 | 403 => Err(ModelError::Auth(self.model.clone())),
                429 => Err(ModelError::RateLimited),
                _ => Err(ModelError::Unavailable(format!("HTTP {status}"))),
            };
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ModelError::Parse(e.to_string()))?;

        let embeddings: Vec<Vec<f32>> = data["data"]
            .as_array()
            .ok_or_else(|| ModelError::Parse("missing data array".into()))?
            .iter()
            .map(|v| {
                v["embedding"]
                    .as_array()
                    .ok_or_else(|| ModelError::Parse("missing embedding".into()))
                    .map(|arr| {
                        arr.iter()
                            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                            .collect()
                    })
            })
            .collect::<Result<_, _>>()?;

        if embeddings.len() != texts.len() {
            return Err(ModelError::Parse(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            )));
        }
        Ok(embeddings)
    }

    fn embed_sync(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>> {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.embed(texts))
    }

    fn backend_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time check: OpenAiEmbedder implements Embedder.
    #[test]
    fn openai_embedder_implements_trait() {
        let e = OpenAiEmbedder::new("sk-test".into());
        assert_eq!(e.dim(), 1536);
        assert_eq!(e.backend_id(), "text-embedding-3-small");
    }

    /// Error handling: a Network error should be caught when there is no real network.
    #[test]
    fn embed_without_network_returns_error() {
        let e = OpenAiEmbedder::new("sk-test".into());
        let result = e.embed_sync(&["hello".into()]);
        // Without network it should return a Network error (not panic)
        assert!(result.is_err());
    }
}
