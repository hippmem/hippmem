//! Full enum re-export: gathers all enums here for one-stop import and testing.
//!
//! Authoritative definitions of each enum live in:
//! - unit.rs: ContentType, Language, SourceKind, MemoryStage, MemoryLifecycle, GeneratedBy
//! - understanding.rs: EntityType, GoalStatus, Polarity, EmotionKind, CausalKind
//! - links.rs: LinkType, LinkDirection, MatchDimension, ObservationState, RecallChannel, RetrievalMode, MemoryWarning

// Imported from unit.rs
pub use crate::model::unit::{
    ContentType, GeneratedBy, Language, MemoryLifecycle, MemoryStage, SourceKind,
};
// Imported from understanding.rs
pub use crate::model::understanding::{CausalKind, EmotionKind, EntityType, GoalStatus, Polarity};
// Imported from links.rs
pub use crate::model::links::{
    LinkDirection, LinkType, MatchDimension, MemoryWarning, ObservationState, RecallChannel,
    RetrievalMode,
};
