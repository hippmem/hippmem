#![allow(dead_code)]
//! Shared test utilities for hippmem-eval integration tests.
//!
//! Provides standard contexts, locale discovery, fixture loading, and API backend
//! configuration. Each test file uses `mod common;` to access these helpers.
//!
//! Locale discovery and fixture loading functions delegate to
//! `hippmem_eval::fixture_loader` so that examples can also use them.

use hippmem_core::config::EmbedderConfig;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext};

// ── Re-export locale discovery / fixture loading from the library ──

#[allow(unused_imports)]
pub use hippmem_eval::fixture_loader::{
    discover_bench_locales, discover_corpus_locales, discover_test_locales, load_corpus_case,
    load_test_fixture,
};

// ── Standard contexts ──

/// Standard write context for eval tests (session_id=1, fixed timestamp).
pub fn write_ctx() -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: hippmem_core::time::Timestamp(1_700_000_000_000),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

/// Empty retrieve context.
pub fn retrieve_ctx() -> RetrieveContext {
    RetrieveContext::default()
}

// ── API backend configuration ──

/// Build an EmbedderConfig from environment variables, if an API key is available.
///
/// Environment variables:
/// - `OPENAI_API_KEY` (required)
/// - `HIPPMEM_EMBEDDER_BASE_URL` (default: `https://api.openai.com/v1`)
/// - `HIPPMEM_EMBEDDER_MODEL` (default: `text-embedding-3-small`)
pub fn api_embedder_config() -> Option<EmbedderConfig> {
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    if api_key.is_empty() {
        return None;
    }
    let base_url = std::env::var("HIPPMEM_EMBEDDER_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("HIPPMEM_EMBEDDER_MODEL")
        .unwrap_or_else(|_| "text-embedding-3-small".to_string());
    Some(EmbedderConfig::OpenAiCompatible {
        base_url,
        model,
        api_key: Some(api_key),
        dimensions: 1024,
    })
}

/// Open an engine, preferring the API backend if configured, otherwise using the
/// deterministic fallback backend (no network dependency).
pub fn open_engine(dir: &tempfile::TempDir) -> Engine {
    let mut config = EngineConfig {
        store_dir: dir.path().join("eval.redb"),
        ..Default::default()
    };
    if let Some(embedder) = api_embedder_config() {
        config.embedder = embedder;
    }
    Engine::open(config).expect("Engine::open")
}
