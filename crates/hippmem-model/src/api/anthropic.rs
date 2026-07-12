//! Anthropic API backend (08 §3). feature-gated: api-backends.

use crate::error::ModelResult;
use crate::traits::Extractor;

/// Anthropic extractor: extracts structured understanding via the Claude API.
pub struct AnthropicExtractor {
    _api_key: String,
}

impl AnthropicExtractor {
    pub fn new(api_key: String) -> Self {
        Self { _api_key: api_key }
    }
}

#[async_trait::async_trait]
impl Extractor for AnthropicExtractor {
    async fn extract_immediate(
        &self,
        _c: &hippmem_core::model::unit::MemoryContent,
    ) -> ModelResult<crate::traits::ImmediateExtraction> {
        Err(crate::error::ModelError::Network(
            "Anthropic extractor not yet implemented".into(),
        ))
    }
    async fn extract_strong(
        &self,
        _c: &hippmem_core::model::unit::MemoryContent,
    ) -> ModelResult<crate::traits::StrongExtraction> {
        Err(crate::error::ModelError::Network(
            "Anthropic strong extractor not yet implemented".into(),
        ))
    }
    fn backend_id(&self) -> &str {
        "anthropic"
    }
}
