//! HIPPMEM core data model module.
//!
//! Contains MemoryUnit and all its sub-types. File-level responsibilities:
//! - `unit`: MemoryUnit, MemoryContent, WriteContext, Provenance, and lifecycle/stage enums
//! - `understanding`: MemoryUnderstanding and all understanding frames
//! - `links`: AssociationKeys, AssociationLink, ActivationState, and recall/retrieval related types

pub mod enums;
pub mod links;
pub mod understanding;
pub mod unit;

// Re-export the most commonly used types to the model level
pub use links::{AssociationKeys, AssociationLink, LinkDirection, LinkEvidence, LinkType};
pub use understanding::MemoryUnderstanding;
pub use unit::{
    ContentType, Language, MemoryContent, MemoryLifecycle, MemoryStage, MemoryUnit, Provenance,
    SourceKind, SourceRef, TextSpan, WriteContext,
};
