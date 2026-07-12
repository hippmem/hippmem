//! MemoryUnit core types: MemoryUnit, MemoryContent, WriteContext, Provenance, and lifecycle/stage enums.
//!
//! Corresponds to 02#memoryunit, 02#memorycontent, 02#writecontext, 02#memorylifecycle, 02#provenance.

use crate::ids::MemoryId;
use crate::model::links::{ActivationState, AssociationKeys, AssociationLink};
use crate::model::understanding::MemoryUnderstanding;
use crate::score::UnitScore;
use crate::time::Timestamp;
use serde::{Deserialize, Serialize};

/// Memory unit: HIPPMEM's core data object — not an ordinary document, but a network node of a three-layer structure of "content + associations + activation history".
///
/// Corresponds to 02#memoryunit, traceable to whitepaper §5.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryUnit {
    /// Serialization version number, currently = 1 (ADR-005).
    pub schema_version: u16,
    /// Unique memory ID (ULID, u128).
    pub id: MemoryId,
    /// Creation time (UTC milliseconds).
    pub created_at: Timestamp,
    /// Last modification time (invariant: >= created_at).
    pub updated_at: Timestamp,
    /// Memory content (raw/summary/normalized).
    pub content: MemoryContent,
    /// Environmental context at write time.
    pub context: WriteContext,
    /// Structured understanding result.
    pub understanding: MemoryUnderstanding,
    /// Multi-dimensional recall keys.
    pub association_keys: AssociationKeys,
    /// Out-edges of this memory (invariant: deduplicated by (target_id, link_type), no self-loops).
    pub links: Vec<AssociationLink>,
    /// Activation history.
    pub activation: ActivationState,
    /// Lifecycle state.
    pub lifecycle: MemoryLifecycle,
    /// Provenance.
    pub provenance: Provenance,
    /// Current stage of the staged memory.
    pub stage: MemoryStage,
}

/// Memory content: carries raw text, summary, normalized text, language, and content type.
///
/// Corresponds to 02#memorycontent, traceable to whitepaper §5.2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryContent {
    /// Raw text (immutable).
    pub raw: String,
    /// Summary (filled in the enriched/consolidated stage).
    pub summary: Option<String>,
    /// Normalized text (noise removal / case normalization; filled in the indexed stage).
    pub normalized: Option<String>,
    /// Language.
    pub language: Language,
    /// Content type.
    pub content_type: ContentType,
}

/// Language enum. Corresponds to the Chinese/English tokenization path selection in ADR-018.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// Chinese
    Zh,
    /// English
    En,
    /// Code snippet
    Code,
    /// Mixed Chinese/English
    Mixed,
    /// Reserved: BCP-47 numeric code
    Other(u16),
}

/// Content type enum. Each type has a different decay-protection level and importance baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    /// User statement
    UserStatement,
    /// Assistant observation
    AssistantObservation,
    /// Tool result
    ToolResult,
    /// Decision
    Decision,
    /// Preference
    Preference,
    /// Event
    Event,
    /// Task state
    TaskState,
    /// Project knowledge
    ProjectKnowledge,
    /// Reflection
    Reflection,
    /// Correction
    Correction,
}

// ── WriteContext ──

/// Write context: environmental information such as session, task, and project at write time.
/// All fields are retained (constitution C3); the first version may leave them empty.
///
/// Corresponds to 02#writecontext, traceable to whitepaper §5.3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteContext {
    /// Conversation ID.
    pub conversation_id: Option<u64>,
    /// Session instance ID.
    pub session_id: Option<u64>,
    /// Project ID.
    pub project_id: Option<u64>,
    /// Task ID.
    pub task_id: Option<u64>,
    /// User ID.
    pub user_id: Option<u64>,
    /// Local time at write.
    pub local_time: Timestamp,
    /// Preceding adjacent memory IDs (for temporal-proximity recall).
    pub preceding_memory_ids: Vec<MemoryId>,
    /// Source references.
    pub source_refs: Vec<SourceRef>,
}

/// Source reference: describes where a memory's content came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    /// Source type.
    pub kind: SourceKind,
    /// URI / file path / external system ID / memory ID string.
    pub locator: String,
    /// Position in the source text (optional).
    pub span: Option<TextSpan>,
}

/// Source type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceKind {
    /// Conversation
    Conversation,
    /// File
    File,
    /// Tool
    Tool,
    /// External system
    ExternalSystem,
    /// Points to another memory
    MemoryRef,
    /// Other
    Other,
}

/// Byte offset span in text (start <= end).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSpan {
    /// Start byte offset.
    pub start: u32,
    /// End byte offset.
    pub end: u32,
}

// ── MemoryStage / MemoryLifecycle ──

/// Staged memory stage. Unidirectional: Raw → Indexed → Enriched → Consolidated.
///
/// Corresponds to 02#memorystage, traceable to whitepaper §6.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryStage {
    /// Raw content persisted
    Raw,
    /// Basic immediate dimensions completed
    Indexed,
    /// Strong semantic dimensions filled in
    Enriched,
    /// Summarization/merging/long-term evolution completed
    Consolidated,
}

/// Memory lifecycle state machine.
///
/// Corresponds to 02#memorylifecycle, traceable to whitepaper §5.1, §8.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLifecycle {
    /// Active
    Active,
    /// Compressed into a summary memory
    Compressed {
        /// Target summary memory ID.
        into: MemoryId,
    },
    /// Archived
    Archived,
    /// Superseded by a new memory
    Superseded {
        /// The superseding memory ID.
        by: MemoryId,
    },
    /// Outdated but retained for historical significance
    Deprecated,
    /// Explicitly negated by the user
    Negated {
        /// The negator (correcting memory) ID.
        by: MemoryId,
    },
}

// ── Provenance ──

/// Provenance: the source, evidence, generation method, and reliability of a memory.
///
/// Corresponds to 02#provenance, traceable to whitepaper §5.1, risk 5.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Source type.
    pub origin: SourceKind,
    /// Who produced this understanding/memory.
    pub generated_by: GeneratedBy,
    /// Reliability.
    pub reliability: UnitScore,
    /// Evidence references.
    pub evidence_refs: Vec<SourceRef>,
    /// Marks from each consolidation/correction.
    pub revision_history: Vec<RevisionMark>,
}

/// Generator: who produced this understanding/memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneratedBy {
    /// Direct user input
    UserDirect,
    /// Which extraction backend (e.g. "anthropic-claude" / "deterministic")
    Extractor {
        /// Backend identifier name.
        backend: String,
    },
    /// Generated by background consolidation (e.g. summary)
    Consolidation,
    /// Rule-based generation
    Rule,
}

/// Revision mark: the time and reason of each consolidation/correction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionMark {
    /// Revision time.
    pub at: Timestamp,
    /// Revision reason.
    pub reason: String,
    /// Reviser.
    pub by: GeneratedBy,
}
