//! Basic error types and Result alias.
//!
//! Uses thiserror to define the core crate's structured error enum (ADR-009).
//! Library crates **MUST NOT** use `unwrap()`/`expect()`/`panic!` for recoverable errors.

use thiserror::Error;

/// Error enum for the HIPPMEM core crate.
///
/// All recoverable errors are propagated through this enum.
/// Note: this file does not contain serialization-layer errors (e.g. bincode::Error, see ADR-005);
/// such conversions are added as `From` impls in modules that need persistence.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum CoreError {
    /// Data schema version is too new for the current implementation to read.
    /// Corresponds to 02#11: returns a structured error rather than panicking when an unknown version is encountered during deserialization.
    #[error("schema version too new: data version {found}, current supported version {current}")]
    SchemaTooNew {
        /// Schema version number in the data.
        found: u16,
        /// Highest schema version supported by the current library.
        current: u16,
    },

    /// Data validation failed (an invariant was broken).
    #[error("validation failed: {0}")]
    Validation(String),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Internal logic error (a state that should not occur but is recoverable).
    #[error("internal error: {0}")]
    Internal(String),
}

impl CoreError {
    /// The data's version number when constructing SchemaTooNew.
    pub fn found_version(&self) -> u16 {
        match self {
            Self::SchemaTooNew { found, .. } => *found,
            _ => 0,
        }
    }

    /// The currently supported version number when constructing SchemaTooNew.
    pub fn current_version(&self) -> u16 {
        match self {
            Self::SchemaTooNew { current, .. } => *current,
            _ => 0,
        }
    }
}

/// Result alias for the HIPPMEM core crate.
pub type CoreResult<T> = Result<T, CoreError>;
