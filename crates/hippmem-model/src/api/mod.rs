//! API backends: OpenAI/Anthropic/Cohere clients (08 §3).
//!
//! Gated by the `api-backends` feature; not compiled by default in CI.

#[cfg(feature = "api-backends")]
pub mod anthropic;
#[cfg(feature = "api-backends")]
pub mod cohere;
#[cfg(feature = "api-backends")]
pub mod openai;
