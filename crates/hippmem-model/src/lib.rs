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
