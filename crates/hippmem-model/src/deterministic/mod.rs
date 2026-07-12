//! Deterministic fallback backend: no network, no randomness, reproducible.
//!
//! Corresponds to 08 §4.

pub mod embed;
pub mod extract;
pub mod rerank;
pub mod summarize;
