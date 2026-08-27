//! OpenAI-compatible extractor (08 §3): structured understanding via any
//! service implementing the OpenAI chat-completions protocol.
//!
//! Vendor-neutral by design (2026-08-26 decision): the base URL may point
//! at OpenAI, DashScope, vLLM, a local proxy, or any compatible service.

use crate::error::{ModelError, ModelResult};
use crate::traits::{Extractor, ImmediateExtraction};
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::understanding::{
    CausalClaim, CausalKind, EntityMention, EntityType, TopicTag,
};
use hippmem_core::model::unit::{Language, MemoryContent};
use hippmem_core::score::UnitScore;
use std::time::Duration;

/// Default model: a capable general model. Override via
/// `HIPPMEM_EXTRACTOR_MODEL` or the constructor.
pub const DEFAULT_EXTRACTOR_MODEL: &str = "gpt-4o";
/// Default endpoint (OpenAI-compatible chat completions).
pub const DEFAULT_EXTRACTOR_BASE_URL: &str = "https://api.openai.com/v1";

/// System prompt: asks the model for a strict JSON extraction result.
const EXTRACT_PROMPT: &str = r#"You are a structured understanding extractor for an associative memory engine.
Given a user text, output ONLY a JSON object with exactly these keys:
- "entities": list of {"text": <surface form as it appears>, "canonical": <most stable name>, "entity_type": "person"|"project"|"library"|"file"|"org"|"concept"|"other"}
- "topics": list of {"label": <short topic>, "confidence": <0..1>}
- "explicit_causals": list of {"cause": <cause>, "effect": <effect>}
- "importance": <0..1, how memorable/reusable this memory is>
- "language": "zh"|"en"|"code"|"mixed"
- "content_type": "user_statement"|"assistant_observation"|"tool_result"|"decision"|"preference"|"event"|"task_state"|null

Rules:
- Extract entities that matter for future recall: people, projects, places, organizations, products, key concepts.
- canonical: the most stable name for the entity; do not guess aggressive aliases.
- Empty fields use empty lists. No markdown, no commentary. JSON only."#;

/// OpenAI-compatible extractor: calls any chat-completions service.
pub struct OpenAiExtractor {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiExtractor {
    /// Defaults: gpt-4o, OpenAI endpoint.
    pub fn new(api_key: String) -> Self {
        Self::new_with_base_url(api_key, DEFAULT_EXTRACTOR_BASE_URL, DEFAULT_EXTRACTOR_MODEL)
            .expect("new() is given an explicit key, so it won't trigger an Auth error")
    }

    /// Full constructor. An empty `api_key` is resolved from
    /// `HIPPMEM_EXTRACTOR_API_KEY`, then `OPENAI_API_KEY`; missing → Auth error.
    pub fn new_with_base_url(api_key: String, base_url: &str, model: &str) -> ModelResult<Self> {
        let api_key = if api_key.is_empty() {
            std::env::var("HIPPMEM_EXTRACTOR_API_KEY")
                .ok()
                .filter(|k| !k.is_empty())
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .unwrap_or_default()
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
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(|e| ModelError::Network(format!("reqwest client build: {e}")))?,
        })
    }

    /// Builds from environment (registry entry point):
    /// `HIPPMEM_EXTRACTOR_API_KEY` (fallback `OPENAI_API_KEY`),
    /// `HIPPMEM_EXTRACTOR_BASE_URL`, `HIPPMEM_EXTRACTOR_MODEL`.
    pub fn from_env() -> ModelResult<Self> {
        let base_url = std::env::var("HIPPMEM_EXTRACTOR_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_EXTRACTOR_BASE_URL.to_string());
        let model = std::env::var("HIPPMEM_EXTRACTOR_MODEL")
            .unwrap_or_else(|_| DEFAULT_EXTRACTOR_MODEL.to_string());
        Self::new_with_base_url(String::new(), &base_url, &model)
    }
}

fn parse_entity_type(s: &str) -> EntityType {
    match s.to_ascii_lowercase().as_str() {
        "person" => EntityType::Person,
        "project" => EntityType::Project,
        "library" => EntityType::Library,
        "file" => EntityType::File,
        "org" | "organization" => EntityType::Org,
        "concept" => EntityType::Concept,
        _ => EntityType::Other,
    }
}

fn parse_language(s: &str) -> Language {
    match s.to_ascii_lowercase().as_str() {
        "zh" | "zh-cn" | "chinese" => Language::Zh,
        "en" | "english" => Language::En,
        "code" | "code snippet" => Language::Code,
        _ => Language::Mixed,
    }
}

fn parse_content_type(v: &serde_json::Value) -> Option<ContentType> {
    let s = v.as_str()?;
    match s.to_ascii_lowercase().as_str() {
        "user_statement" | "userstatement" => Some(ContentType::UserStatement),
        "assistant_observation" | "assistantobservation" => Some(ContentType::AssistantObservation),
        "tool_result" | "toolresult" => Some(ContentType::ToolResult),
        "decision" => Some(ContentType::Decision),
        "preference" => Some(ContentType::Preference),
        "event" => Some(ContentType::Event),
        "task_state" | "taskstate" => Some(ContentType::TaskState),
        _ => None,
    }
}

/// Extracts the JSON object substring from a model reply (tolerates
/// markdown fences or surrounding prose).
fn extract_json_substring(reply: &str) -> Option<&str> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&reply[start..=end])
}

fn parse_extraction(reply: &str) -> ModelResult<ImmediateExtraction> {
    let json_str = extract_json_substring(reply)
        .ok_or_else(|| ModelError::Parse("no JSON object in model reply".into()))?;
    let v: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| ModelError::Parse(format!("invalid extraction JSON: {e}")))?;

    let entities = v["entities"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|e| EntityMention {
                    text: e["text"].as_str().unwrap_or_default().to_string(),
                    canonical: e["canonical"].as_str().unwrap_or_default().to_string(),
                    entity_type: parse_entity_type(e["entity_type"].as_str().unwrap_or_default()),
                    span: None,
                    confidence: UnitScore::new(e["confidence"].as_f64().unwrap_or(0.8) as f32),
                })
                .filter(|m| !m.text.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let topics = v["topics"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|t| TopicTag {
                    label: t["label"].as_str().unwrap_or_default().to_string(),
                    confidence: UnitScore::new(t["confidence"].as_f64().unwrap_or(0.5) as f32),
                })
                .filter(|t| !t.label.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let explicit_causals = v["explicit_causals"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|c| CausalClaim {
                    cause: c["cause"].as_str().unwrap_or_default().to_string(),
                    effect: c["effect"].as_str().unwrap_or_default().to_string(),
                    kind: CausalKind::Explicit,
                    evidence_span: None,
                    confidence: UnitScore::new(c["confidence"].as_f64().unwrap_or(0.8) as f32),
                })
                .filter(|c| !c.cause.is_empty() && !c.effect.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let language = parse_language(v["language"].as_str().unwrap_or_default());
    let content_type = parse_content_type(&v["content_type"]);
    let importance = UnitScore::new(v["importance"].as_f64().unwrap_or(0.0) as f32);

    Ok(ImmediateExtraction {
        entities,
        topics,
        explicit_causals,
        language,
        content_type,
        importance,
    })
}

#[async_trait::async_trait]
impl Extractor for OpenAiExtractor {
    async fn extract_immediate(&self, content: &MemoryContent) -> ModelResult<ImmediateExtraction> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": content.raw},
            ],
            "temperature": 0,
        });

        let url = format!("{}/chat/completions", self.base_url);
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
        let content = data["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|c| c["message"]["content"].as_str())
            .ok_or_else(|| ModelError::Parse("missing choices[0].message.content".into()))?;

        parse_extraction(content)
    }

    fn extract_immediate_sync(
        &self,
        content: &hippmem_core::model::unit::MemoryContent,
    ) -> ModelResult<ImmediateExtraction> {
        tokio::runtime::Runtime::new()
            .map_err(|e| ModelError::Network(e.to_string()))?
            .block_on(self.extract_immediate(content))
    }

    /// Strong dimensions are deferred (the write path emits
    /// `StrongDimsDeferred` and reruns consolidation later).
    async fn extract_strong(
        &self,
        _content: &MemoryContent,
    ) -> ModelResult<crate::traits::StrongExtraction> {
        Err(ModelError::Unavailable(
            "strong extraction deferred for neural backend".into(),
        ))
    }

    fn backend_id(&self) -> &str {
        "openai-compatible"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_extraction_handles_fenced_reply() {
        let reply = "```json\n{\"entities\": [{\"text\": \"小明\", \"canonical\": \"小明\", \"entity_type\": \"person\"}], \"topics\": [{\"label\": \"住址\", \"confidence\": 0.8}], \"explicit_causals\": [], \"importance\": 0.7, \"language\": \"zh\", \"content_type\": \"user_statement\"}\n```";
        let out = parse_extraction(reply).expect("fenced JSON parses");
        assert_eq!(out.entities.len(), 1);
        assert_eq!(out.entities[0].text, "小明");
        assert_eq!(out.entities[0].entity_type, EntityType::Person);
        assert_eq!(out.topics[0].label, "住址");
        assert_eq!(out.language, Language::Zh);
        assert_eq!(out.content_type, Some(ContentType::UserStatement));
        assert!((out.importance.value() - 0.7).abs() < 0.01);
    }

    #[test]
    fn parse_extraction_tolerates_missing_fields() {
        let reply = "{\"entities\": [], \"importance\": 0.5}";
        let out = parse_extraction(reply).expect("sparse JSON parses");
        assert!(out.entities.is_empty());
        assert!(out.topics.is_empty());
        assert_eq!(out.language, Language::Mixed, "missing language -> Mixed");
        assert_eq!(out.content_type, None);
    }

    #[test]
    fn parse_extraction_rejects_non_json() {
        assert!(parse_extraction("sorry, I cannot do that").is_err());
        assert!(parse_extraction("").is_err());
    }

    #[test]
    fn entity_type_parsing_is_lenient() {
        assert_eq!(parse_entity_type("PERSON"), EntityType::Person);
        assert_eq!(parse_entity_type("organization"), EntityType::Org);
        assert_eq!(parse_entity_type("whatever"), EntityType::Other);
    }

    #[test]
    fn empty_key_resolves_from_env_and_falls_back() {
        let _lock = crate::test_env::ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::clear();
        let r = OpenAiExtractor::new_with_base_url(
            String::new(),
            "https://example.com/v1",
            "test-model",
        );
        assert!(r.is_err(), "no key anywhere -> Auth error");
    }

    #[test]
    fn key_resolved_from_env_when_constructor_key_empty() {
        let _lock = crate::test_env::ENV_LOCK.lock().unwrap();
        let mut guard = EnvGuard::clear();
        guard.set("OPENAI_API_KEY", "sk-env-test");
        let r = OpenAiExtractor::new_with_base_url(
            String::new(),
            "https://example.com/v1",
            "test-model",
        );
        assert!(r.is_ok(), "env fallback key should be picked up");
    }
    /// Test helper: clears the extractor-related env vars on creation and
    /// restores their previous values when dropped (parallel-test safe).
    struct EnvGuard {
        prev_key: Option<String>,
        prev_openai: Option<String>,
    }

    impl EnvGuard {
        fn clear() -> Self {
            let prev_key = std::env::var("HIPPMEM_EXTRACTOR_API_KEY").ok();
            let prev_openai = std::env::var("OPENAI_API_KEY").ok();
            std::env::remove_var("HIPPMEM_EXTRACTOR_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            Self {
                prev_key,
                prev_openai,
            }
        }

        fn set(&mut self, key: &str, value: &str) {
            std::env::set_var(key, value);
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev_key {
                Some(v) => std::env::set_var("HIPPMEM_EXTRACTOR_API_KEY", v),
                None => std::env::remove_var("HIPPMEM_EXTRACTOR_API_KEY"),
            }
            match &self.prev_openai {
                Some(v) => std::env::set_var("OPENAI_API_KEY", v),
                None => std::env::remove_var("OPENAI_API_KEY"),
            }
        }
    }
}
