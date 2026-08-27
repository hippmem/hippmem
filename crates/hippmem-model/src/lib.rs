//! HIPPMEM · Model backend layer
//!
//! Defines the four traits Embedder/Extractor/Reranker/Summarizer and their
//! associated types, plus the ModelRegistry backend registration and selection hub.
//!
//! Corresponds to 08 §2 / §5.

pub mod api;
pub mod deterministic;
pub mod error;
pub mod lang;
pub mod registry;
pub mod traits;

#[cfg(test)]
pub(crate) mod test_env {
    /// Serializes env-var mutation tests (parallel-safe): env is process
    /// global, so a test setting OPENAI_API_KEY would otherwise leak into
    /// concurrent tests reading it.
    pub static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
