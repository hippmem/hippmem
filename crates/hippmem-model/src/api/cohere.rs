//! Cohere API backend (08 §3). feature-gated: api-backends.

use crate::error::ModelResult;
use crate::traits::Reranker;

/// Cohere reranker: reranks candidates via the Cohere Rerank API.
pub struct CohereReranker {
    _api_key: String,
}

impl CohereReranker {
    pub fn new(api_key: String) -> Self {
        Self { _api_key: api_key }
    }
}

#[async_trait::async_trait]
impl Reranker for CohereReranker {
    async fn rerank(&self, _query: &str, _candidates: &[String]) -> ModelResult<Vec<f32>> {
        Err(crate::error::ModelError::Network(
            "Cohere reranker not yet implemented".into(),
        ))
    }
    fn backend_id(&self) -> &str {
        "cohere"
    }
}
