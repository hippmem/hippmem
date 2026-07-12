//! Four model traits and associated types: 08 §2.

use crate::error::ModelResult;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::understanding::{
    CausalClaim, ContradictionHint, DecisionFrame, EmotionFrame, EntityMention, GoalFrame,
    PreferenceFrame, TopicTag,
};
use hippmem_core::model::unit::{ContentType, Language, MemoryContent};
use hippmem_core::score::UnitScore;

// ── Embedder ──

/// Text -> dense vector.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Returns the vector dimension (fixed per backend).
    fn dim(&self) -> usize;

    /// Batch embedding. Takes N text segments, returns N equal-length vectors.
    async fn embed(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>>;

    /// Synchronous embedding (for tests / simple scenarios).
    /// Each backend implements its own; no default impl to avoid implicitly
    /// pulling in a tokio runtime dependency.
    fn embed_sync(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>>;

    /// Backend identifier, written to provenance.
    fn backend_id(&self) -> &str;
}

// ── Extractor ──

/// Text -> structured understanding.
#[async_trait::async_trait]
pub trait Extractor: Send + Sync {
    /// Extract basic immediate dimensions (entities / topics / explicit causals).
    /// All backends MUST provide this.
    async fn extract_immediate(&self, content: &MemoryContent) -> ModelResult<ImmediateExtraction>;

    /// Extract strong semantic dimensions (goals / preferences / emotions /
    /// decisions / implicit causals / contradictions).
    async fn extract_strong(&self, content: &MemoryContent) -> ModelResult<StrongExtraction>;

    /// Backend identifier.
    fn backend_id(&self) -> &str;
}

// ── Reranker ──

/// (query, candidate) -> rerank score.
#[async_trait::async_trait]
pub trait Reranker: Send + Sync {
    /// Takes a query and candidate texts, returns scores of the same length as
    /// candidates (higher means more relevant).
    async fn rerank(&self, query: &str, candidates: &[String]) -> ModelResult<Vec<f32>>;

    /// Backend identifier.
    fn backend_id(&self) -> &str;
}

// ── Summarizer ──

/// Multiple memories -> summary.
#[async_trait::async_trait]
pub trait Summarizer: Send + Sync {
    /// Takes several memory raw texts, produces a summary text plus coverage info.
    async fn summarize(&self, sources: &[SummarizeInput]) -> ModelResult<SummaryOutput>;

    /// Backend identifier.
    fn backend_id(&self) -> &str;
}

// ── Associated types ──

/// Basic immediate dimension extraction result (08 §2).
#[derive(Debug, Clone)]
pub struct ImmediateExtraction {
    /// Entity mention list.
    pub entities: Vec<EntityMention>,
    /// Preliminary topic tags.
    pub topics: Vec<TopicTag>,
    /// Explicit causal claims.
    pub explicit_causals: Vec<CausalClaim>,
    /// Detected language.
    pub language: Language,
    /// Backend's judgement of content type (may be None).
    pub content_type: Option<ContentType>,
    /// Overall importance.
    pub importance: UnitScore,
}

/// Strong semantic dimension extraction result (08 §2).
#[derive(Debug, Clone)]
pub struct StrongExtraction {
    /// Goal frame list.
    pub goals: Vec<GoalFrame>,
    /// Preference frame list.
    pub preferences: Vec<PreferenceFrame>,
    /// Emotion frame list.
    pub emotions: Vec<EmotionFrame>,
    /// Decision frame list.
    pub decisions: Vec<DecisionFrame>,
    /// Implicit causal claims.
    pub implicit_causals: Vec<CausalClaim>,
    /// Contradiction hint list.
    pub contradictions: Vec<ContradictionHint>,
    /// Overall confidence; the fallback backend gives a lower value.
    pub confidence: UnitScore,
}

/// Summary input (08 §2).
#[derive(Debug, Clone)]
pub struct SummarizeInput {
    /// ID of the memory being summarized.
    pub id: MemoryId,
    /// Memory raw text.
    pub text: String,
}

/// Summary output (08 §2).
#[derive(Debug, Clone)]
pub struct SummaryOutput {
    /// Summary text.
    pub summary: String,
    /// List of covered memory IDs.
    pub covers: Vec<MemoryId>,
    /// Summary confidence.
    pub confidence: UnitScore,
}
