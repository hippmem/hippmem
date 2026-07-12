//! MemoryUnderstanding and all understanding frames.
//!
//! Corresponds to 02#memoryunderstanding, 02#understanding-frames.

use crate::ids::MemoryId;
use crate::model::unit::TextSpan;
use crate::score::UnitScore;
use crate::time::Timestamp;
use serde::{Deserialize, Serialize};

/// Memory understanding: the structured understanding result that an algorithm or model produces for content.
///
/// Dimension layering:
/// - Basic immediate dimensions (indexed stage): entities, topics, explicit causal_claims
/// - Strong semantic dimensions (enriched stage): goals, preferences, emotions, decisions, implicit causal_claims, contradictions
///
/// Corresponds to 02#memoryunderstanding, traceable to whitepaper §5.4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryUnderstanding {
    /// Entity mention list.
    pub entities: Vec<EntityMention>,
    /// Event frame list.
    pub events: Vec<EventFrame>,
    /// Goal frame list (strong semantic dimension).
    pub goals: Vec<GoalFrame>,
    /// Decision frame list (strong semantic dimension).
    pub decisions: Vec<DecisionFrame>,
    /// Preference frame list (strong semantic dimension).
    pub preferences: Vec<PreferenceFrame>,
    /// Emotion frame list (strong semantic dimension).
    pub emotions: Vec<EmotionFrame>,
    /// Causal claim list.
    pub causal_claims: Vec<CausalClaim>,
    /// Contradiction hint list (strong semantic dimension).
    pub contradictions: Vec<ContradictionHint>,
    /// Preliminary topics (basic immediate dimension).
    pub topics: Vec<TopicTag>,
    /// Overall importance.
    pub importance: UnitScore,
    /// Confidence in the overall understanding.
    pub confidence: UnitScore,
}

// ── EntityMention ──

/// Entity mention: a person/project/library/file/organization/concept mentioned in text, along with its position and type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityMention {
    /// The entity's surface form in the source text.
    pub text: String,
    /// Canonical name (used to generate entity_key).
    pub canonical: String,
    /// Entity type.
    pub entity_type: EntityType,
    /// Position in the source text.
    pub span: Option<TextSpan>,
    /// Confidence.
    pub confidence: UnitScore,
}

/// Entity type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Person
    Person,
    /// Project
    Project,
    /// Library
    Library,
    /// File
    File,
    /// Organization
    Org,
    /// Concept
    Concept,
    /// Other
    Other,
}

// ── EventFrame ──

/// Event frame: the time, participants, action, and outcome of an event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventFrame {
    /// Action/predicate.
    pub action: String,
    /// Canonical names of participating entities.
    pub participants: Vec<String>,
    /// Occurrence time.
    pub occurred_at: Option<Timestamp>,
    /// Outcome.
    pub outcome: Option<String>,
    /// Confidence.
    pub confidence: UnitScore,
}

// ── GoalFrame ──

/// Goal frame: user/project goal, constraints, and status. A strong semantic dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalFrame {
    /// Goal description.
    pub description: String,
    /// Goal status.
    pub status: GoalStatus,
    /// Constraints.
    pub constraints: Vec<String>,
    /// Confidence.
    pub confidence: UnitScore,
}

/// Goal status enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GoalStatus {
    /// In progress
    Active,
    /// Achieved
    Achieved,
    /// Abandoned
    Abandoned,
    /// Blocked
    Blocked,
    /// Unknown
    Unknown,
}

// ── DecisionFrame ──

/// Decision frame: the content, rationale, time, and whether reverted of a decision. A strong semantic dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionFrame {
    /// What was decided.
    pub decision: String,
    /// Decision rationale.
    pub rationale: Option<String>,
    /// Decision time.
    pub decided_at: Option<Timestamp>,
    /// Whether it has been reverted.
    pub reverted: bool,
    /// Confidence.
    pub confidence: UnitScore,
}

// ── PreferenceFrame ──

/// Preference frame: preference object, polarity (like/dislike), strength, and validity. A strong semantic dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreferenceFrame {
    /// Preference object.
    pub object: String,
    /// Polarity direction.
    pub polarity: Polarity,
    /// Strength.
    pub strength: UnitScore,
    /// Whether still valid (can be negated by a Correction).
    pub still_valid: bool,
    /// Confidence.
    pub confidence: UnitScore,
}

/// Polarity direction enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Polarity {
    /// Like
    Like,
    /// Dislike
    Dislike,
    /// Neutral
    Neutral,
}

// ── EmotionFrame ──

/// Emotion frame: emotion kind, intensity, and trigger object. A strong semantic dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionFrame {
    /// Emotion kind.
    pub emotion: EmotionKind,
    /// Intensity.
    pub intensity: UnitScore,
    /// Trigger object.
    pub trigger: Option<String>,
    /// Confidence.
    pub confidence: UnitScore,
}

/// Fixed emotion category, mapped to emotion_keys (u8). The first version uses basic categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionKind {
    /// Joy
    Joy,
    /// Sadness
    Sadness,
    /// Anger
    Anger,
    /// Fear
    Fear,
    /// Surprise
    Surprise,
    /// Disgust
    Disgust,
    /// Frustration
    Frustration,
    /// Anxiety
    Anxiety,
    /// Satisfaction
    Satisfaction,
    /// Neutral
    Neutral,
    /// Other
    Other,
}

// ── CausalClaim ──

/// Causal claim: a directed cause→effect assertion with confidence and evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalClaim {
    /// Cause.
    pub cause: String,
    /// Effect.
    pub effect: String,
    /// Causal kind (explicit/implicit).
    pub kind: CausalKind,
    /// Position of the evidence in the source text.
    pub evidence_span: Option<TextSpan>,
    /// Confidence.
    pub confidence: UnitScore,
}

/// Causal kind: explicit (conjunction hit) / implicit (model inferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalKind {
    /// Explicit causality (conjunction hit)
    Explicit,
    /// Implicit causality (model inferred)
    Implicit,
}

// ── ContradictionHint ──

/// Contradiction hint: a hint pointing to two potentially conflicting pieces of information. A strong semantic dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContradictionHint {
    /// The claim in this memory.
    pub statement_a: String,
    /// The suspected conflicting old memory (may be unknown at write time).
    pub conflicts_with: Option<MemoryId>,
    /// Conflict description.
    pub note: String,
    /// Confidence.
    pub confidence: UnitScore,
}

// ── TopicTag ──

/// Topic tag: a preliminary topic marker (basic immediate dimension).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicTag {
    /// Topic label (canonical).
    pub label: String,
    /// Confidence.
    pub confidence: UnitScore,
}
