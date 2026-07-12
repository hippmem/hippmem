//! AssociationKeys, AssociationLink, ActivationState and recall/retrieval related types.
//!
//! Corresponds to 02#associationkeys, 02#associationlink, 02#activationstate, 02#recallchannel.

use crate::ids::{
    CausalKey, EmotionKey, EntityKey, EventKey, GoalKey, MemoryId, TemporalKey, TopicKey, VectorId,
};
use crate::model::unit::TextSpan;
use crate::score::UnitScore;
use crate::time::Timestamp;
use serde::{Deserialize, Serialize};

// ── AssociationKeys ──

/// Association keys: multi-dimensional index keys used to quickly discover candidate associations.
///
/// Corresponds to 02#associationkeys, traceable to whitepaper §5.5.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssociationKeys {
    /// Entity key list (hash of entities[].canonical).
    pub entity_keys: Vec<EntityKey>,
    /// Time bucket key list.
    pub temporal_keys: Vec<TemporalKey>,
    /// Lexical signature (SimHash).
    pub lexical_signature: LexicalSignature,
    /// Semantic signature (multi-layer fingerprint).
    pub semantic_signature: SemanticSignature,
    /// Topic key list.
    pub topic_keys: Vec<TopicKey>,
    /// Emotion key list.
    pub emotion_keys: Vec<EmotionKey>,
    /// Goal key list.
    pub goal_keys: Vec<GoalKey>,
    /// Event key list.
    pub event_keys: Vec<EventKey>,
    /// Causal key list.
    pub causal_keys: Vec<CausalKey>,
}

/// Lexical signature: SimHash family, 4×u64 = 256 bits. Used for fast literal similarity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalSignature {
    /// SimHash 256 bits (4 u64s).
    pub simhash: [u64; 4],
}

/// Semantic signature: multi-layer fingerprint (lexical SimHash + dense vector + binary code + topic MinHash).
///
/// Traceable to whitepaper §10.2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticSignature {
    /// Lexical SimHash (same source as LexicalSignature; stored redundantly for recall convenience).
    pub lexical_simhash: [u64; 4],
    /// Dense vector handle (generated in the enriched stage or synchronously; None when using the fallback backend).
    pub dense_embedding_ref: Option<VectorId>,
    /// 128-bit binary semantic code (2 u64s) for fast approximate matching.
    pub binary_code: [u64; 2],
    /// Topic MinHash, 16×u32, for coarse clustering.
    pub topic_minhash: [u32; 16],
}

// ── AssociationLink / LinkType ──

/// Association edge: a native association edge between MemoryUnits, carrying type, direction, strength, confidence, and evidence.
/// A "first-class citizen" type of HIPPMEM.
///
/// Corresponds to 02#associationlink, traceable to whitepaper §5.6, constitution 2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssociationLink {
    /// Target memory ID.
    pub target_id: MemoryId,
    /// Edge type.
    pub link_type: LinkType,
    /// Direction.
    pub direction: LinkDirection,
    /// Edge strength (may be rewritten by Hebbian/decay).
    pub strength: UnitScore,
    /// Confidence that this association holds.
    pub confidence: UnitScore,
    /// Edge evidence (why it was built, constitution C9).
    pub evidence: LinkEvidence,
    /// Formation time.
    pub formed_at: Timestamp,
    /// Last activation time.
    pub last_activated_at: Option<Timestamp>,
    /// Activation count.
    pub activation_count: u32,
    /// Observation zone state.
    pub observation: ObservationState,
}

/// Edge type enum (14 variants). Includes the 12 from whitepaper §5.6 plus Supersedes/Deprecated from §8.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinkType {
    /// Entity overlap: shares the same entity.
    EntityOverlap,
    /// Temporal adjacency: adjacent in time.
    TemporalAdjacent,
    /// Semantic similarity: semantic neighbors.
    SemanticSimilar,
    /// Topic related: same topic cluster.
    TopicRelated,
    /// Same goal: serves the same goal.
    SameGoal,
    /// Same event: belongs to the same event chain.
    SameEvent,
    /// Causal: directed, suitable for tracing along direction.
    Causal,
    /// Emotional resonance: similar emotional state or transition.
    EmotionalResonance,
    /// Contradiction: two memories conflict.
    Contradiction,
    /// Correction: a new memory explicitly corrects an old one.
    Correction,
    /// Elaboration: one memory is an expansion/supplement of the other.
    Elaboration,
    /// Co-activation: historically recalled together many times.
    CoActivation,
    /// Supersedes: a new memory supersedes an old one in time/authority.
    Supersedes,
    /// Deprecated: an old memory is outdated but retained for historical significance.
    Deprecated,
}

/// Edge direction enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinkDirection {
    /// Undirected
    Undirected,
    /// Forward (from → to)
    Forward,
    /// Backward (to → from)
    Backward,
}

/// Edge evidence: why this edge was established (constitution C9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkEvidence {
    /// Which dimensions hit.
    pub contributing_dimensions: Vec<MatchDimension>,
    /// Per-dimension contribution scores.
    pub score_breakdown: Vec<(MatchDimension, f32)>,
    /// Evidence spans (required for causal/contradiction/correction).
    pub text_spans: Vec<TextSpan>,
    /// Supplementary note.
    pub note: Option<String>,
}

/// Hit dimension: used in evidence and matched_dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatchDimension {
    /// Entity
    Entity,
    /// Semantic
    Semantic,
    /// Temporal
    Temporal,
    /// Topic
    Topic,
    /// Goal
    Goal,
    /// Event
    Event,
    /// Emotion
    Emotion,
    /// Causal
    Causal,
    /// Co-context
    CoContext,
    /// Importance
    Importance,
}

/// Observation zone state.
///
/// Traceable to whitepaper §6.3/§8.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationState {
    /// Confirmed edge
    Confirmed,
    /// Under observation (low-confidence but potentially valuable candidate association)
    Observing {
        /// Time of entering the observation zone.
        since: Timestamp,
    },
}

// ── ActivationState ──

/// Activation state: records the history of a memory being retrieved, co-activated, reinforced, and decayed.
///
/// Corresponds to 02#activationstate, traceable to whitepaper §2.1, §5.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationState {
    /// Last retrieval time.
    pub last_retrieved_at: Option<Timestamp>,
    /// Retrieval count.
    pub retrieval_count: u32,
    /// Co-activation counts with other memories (bounded).
    pub co_activations: Vec<CoActivationCount>,
    /// Cumulative usage-value score, drives Hebbian/decay.
    pub usage_score: UnitScore,
}

/// Cumulative co-activation count with a given memory. Used by Hebbian to create a CoActivation edge when there is "no edge but multiple co-activations".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoActivationCount {
    /// The co-activated memory ID.
    pub with: MemoryId,
    /// Co-activation count.
    pub count: u32,
    /// Last co-activation time.
    pub last_at: Timestamp,
}

/// A single-step record of energy arriving at a memory during one retrieval (an element of activation_trace).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationStep {
    /// Source memory (None = seed/direct recall).
    pub from: Option<MemoryId>,
    /// Target memory.
    pub to: MemoryId,
    /// Link type traversed (None for seeds).
    pub via_link: Option<LinkType>,
    /// Which recall channel the seed came from.
    pub channel: Option<RecallChannel>,
    /// Hop count.
    pub hop: u8,
    /// Energy entering this node.
    pub energy_in: f32,
    /// Energy leaving this node.
    pub energy_out: f32,
}

// ── Recall and retrieval related types ──

/// Recall channel: an independent candidate-memory discovery pathway with its own scoring and observable contribution.
///
/// Corresponds to 02#recallchannel, traceable to whitepaper §7.1, constitution 6/9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecallChannel {
    /// BM25 full-text retrieval
    Bm25,
    /// Entity inverted index
    EntityInverted,
    /// Dense semantic vector
    SemanticDense,
    /// Binary code approximation
    SemanticBinary,
    /// Temporal proximity
    Temporal,
    /// Topic clustering
    TopicCluster,
    /// Goal matching
    Goal,
    /// Event matching
    Event,
    /// Causal tracing
    Causal,
    /// Recent activation
    RecentActivation,
    /// Graph spreading itself (used for contribution attribution)
    GraphSpreading,
}

/// Retrieval mode: determines the channel set, hop count, and whether to rerank.
///
/// Corresponds to 02#retrievalmode, traceable to whitepaper §11.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetrievalMode {
    /// Few channels, 1 hop, no reranker
    Fast,
    /// Default: multi-channel, 2 hops, rule/light reranker
    Balanced,
    /// All channels, up to 3 hops, reranker
    Deep,
    /// Same as Deep + outputs the full trace and channel contribution
    Diagnostic,
}

/// A single retrieval result.
///
/// Corresponds to 02#retrievalresult, traceable to whitepaper §7.3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// The recalled memory.
    pub memory: MemoryUnit,
    /// Final score.
    pub final_score: f32,
    /// Activation trace (how it was spread to).
    pub activation_trace: Vec<ActivationStep>,
    /// Hit dimensions.
    pub matched_dimensions: Vec<MatchDimension>,
    /// Risk warnings.
    pub warnings: Vec<MemoryWarning>,
}

/// Memory risk warning: explicitly flags potential issues in retrieval results (constitution C4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryWarning {
    /// A newer correction exists
    HasCorrection {
        /// The correcting memory ID.
        by: MemoryId,
    },
    /// A contradiction exists
    HasContradiction {
        /// The contradicting memory ID.
        with: MemoryId,
    },
    /// Has been superseded
    Superseded {
        /// The superseding memory ID.
        by: MemoryId,
    },
    /// Deprecated
    Deprecated,
    /// Overall low confidence
    LowConfidence,
    /// Low freshness
    StaleFreshness,
}

// Forward reference: MemoryUnit is defined in unit.rs; imported here
use crate::model::unit::MemoryUnit;
