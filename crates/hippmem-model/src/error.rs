//! Model layer error types: 08 §6.

/// Model operation error.
#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    /// Network or timeout error.
    #[error("network/timeout: {0}")]
    Network(String),

    /// Authentication / missing key.
    #[error("auth/missing key for backend {0}")]
    Auth(String),

    /// Rate limited.
    #[error("rate limited")]
    RateLimited,

    /// Failed to parse model output.
    #[error("parse model output: {0}")]
    Parse(String),

    /// Backend unavailable.
    #[error("backend unavailable: {0}")]
    Unavailable(String),
}

/// Model layer Result alias.
pub type ModelResult<T> = Result<T, ModelError>;
