//! Eval corpus format types: Rust deserialization targets for 06 §2 and §4.
//!
//! These types correspond to the JSON structure of `fixtures/corpus/<locale>/*.json`;
//! the runner loads them and converts to Engine API call parameters.

use hippmem_core::model::enums::{MatchDimension, RetrievalMode};
use hippmem_core::model::{ContentType, LinkType};
use serde::{Deserialize, Serialize};

// ── Top-level case ──

/// An eval case: corresponds to the JSON format in 06 §2.
///
/// Contains the write sequence, query, and expected output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCase {
    /// Unique case identifier, e.g. `"fact-recall-001"`.
    pub case_id: String,
    /// Task type: one of the ten categories (see 06 §4).
    pub task_type: TaskType,
    /// Memory write sequence, in order.
    pub writes: Vec<EvalWrite>,
    /// Query parameters.
    pub query: EvalQuery,
    /// Expected output (used for metric computation).
    pub ground_truth: GroundTruth,
}

// ── Task types ──

/// Eval task type: the ten task categories defined in 06 §4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Fact recall
    FactRecall,
    /// Preference recall
    PreferenceRecall,
    /// Project continuity
    ProjectContinuity,
    /// Causal trace
    CausalTrace,
    /// Contradiction detection
    ContradictionDetection,
    /// State change
    StateChange,
    /// Implicit association (no keyword overlap)
    ImplicitAssociation,
    /// Noise resistance
    NoiseResistance,
    /// Long-tail recall
    LongTailRecall,
    /// Explanation quality
    ExplanationQuality,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TaskType::FactRecall => "FactRecall",
            TaskType::PreferenceRecall => "PreferenceRecall",
            TaskType::ProjectContinuity => "ProjectContinuity",
            TaskType::CausalTrace => "CausalTrace",
            TaskType::ContradictionDetection => "ContradictionDetection",
            TaskType::StateChange => "StateChange",
            TaskType::ImplicitAssociation => "ImplicitAssociation",
            TaskType::NoiseResistance => "NoiseResistance",
            TaskType::LongTailRecall => "LongTailRecall",
            TaskType::ExplanationQuality => "ExplanationQuality",
        };
        write!(f, "{s}")
    }
}

// ── Write entries ──

/// A single write instruction in an eval case: corresponds to 06 §2 `writes[]`.
///
/// Simpler than `WriteContext`: only the context fields needed for eval;
/// the runner fills in runtime info such as `Timestamp` and `MemoryId`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalWrite {
    /// Unique ID within the case; supports `"noise_*"` + `n` batch declaration.
    pub local_id: String,
    /// Memory text content.
    pub content: String,
    /// Content type: corresponds to the `ContentType` enum.
    #[serde(default)]
    pub content_type: Option<ContentType>,
    /// Write context (simplified).
    #[serde(default)]
    pub context: EvalWriteContext,
    /// Batch noise count: used together with the `local_id` wildcard.
    #[serde(default)]
    pub n: Option<u32>,
}

/// Simplified write context for eval: only optional ID fields.
///
/// The runner expands it into a full `WriteContext` before calling `engine.write()`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvalWriteContext {
    /// Session ID.
    pub session_id: Option<u64>,
    /// Conversation instance ID.
    pub conversation_id: Option<u64>,
    /// Project ID.
    pub project_id: Option<u64>,
    /// Task ID.
    pub task_id: Option<u64>,
    /// User ID.
    pub user_id: Option<u64>,
}

// ── Query ──

/// Eval query: corresponds to 06 §2 `query`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalQuery {
    /// Query text.
    pub text: String,
    /// Retrieval mode.
    pub mode: RetrievalMode,
    /// Number of results to return.
    pub top_k: usize,
    /// Query context (simplified).
    #[serde(default)]
    pub context: EvalQueryContext,
}

/// Simplified query context for eval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvalQueryContext {
    /// Session ID.
    pub session_id: Option<u64>,
    /// Conversation instance ID.
    pub conversation_id: Option<u64>,
    /// Project ID.
    pub project_id: Option<u64>,
    /// Task ID.
    pub task_id: Option<u64>,
    /// User ID.
    pub user_id: Option<u64>,
}

// ── Expected output ──

/// Expected output: corresponds to 06 §2 `ground_truth`.
///
/// The runner compares it against the system output to compute
/// Recall/Precision and other metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundTruth {
    /// `local_id` list that should be recalled.
    #[serde(default)]
    pub relevant: Vec<String>,
    /// `local_id` list that are acceptable but not required.
    #[serde(default)]
    pub also_acceptable: Vec<String>,
    /// Dimensions the explanation path should hit.
    #[serde(default)]
    pub expected_dimensions: Vec<MatchDimension>,
    /// Expected risk warnings.
    #[serde(default)]
    pub expected_warnings: Vec<ExpectedWarning>,
    /// Edges that should exist (explanation accuracy).
    #[serde(default)]
    pub expected_edges: Vec<ExpectedEdge>,
}

/// Expected association edge: references memories by `local_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedEdge {
    /// Source memory `local_id`.
    pub from: String,
    /// Target memory `local_id`.
    pub to: String,
    /// Link type.
    pub link_type: LinkType,
}

/// Expected risk warning type: corresponds to 06 §8.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpectedWarning {
    /// Has a newer correction
    HasCorrection,
    /// Has a contradiction
    HasContradiction,
    /// Superseded
    Superseded,
    /// Deprecated
    Deprecated,
    /// Low confidence
    LowConfidence,
    /// Freshness expired
    StaleFreshness,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure all fields of Default for EvalWriteContext are None.
    #[test]
    fn eval_write_context_default_is_empty() {
        let ctx = EvalWriteContext::default();
        assert!(ctx.session_id.is_none());
        assert!(ctx.conversation_id.is_none());
    }

    /// TaskType Display is consistent with serde (round-trip preserves information).
    #[test]
    fn task_type_display_roundtrip() {
        for variant in &[
            TaskType::FactRecall,
            TaskType::PreferenceRecall,
            TaskType::ProjectContinuity,
            TaskType::CausalTrace,
            TaskType::ContradictionDetection,
            TaskType::StateChange,
            TaskType::ImplicitAssociation,
            TaskType::NoiseResistance,
            TaskType::LongTailRecall,
            TaskType::ExplanationQuality,
        ] {
            let s = variant.to_string();
            let json = format!("\"{s}\"");
            let rt: TaskType = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, rt);
        }
    }
}
