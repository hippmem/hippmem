//! HIPPMEM · Native Association Memory Engine — unified external Rust API orchestration layer.
//!
//! Integrates [`hippmem_core`], [`hippmem_store`], [`hippmem_model`],
//! [`hippmem_write`], [`hippmem_retrieval`], [`hippmem_consolidation`],
//! providing seven core APIs: `write/retrieve/explain/consolidate/inspect/feedback`.
//!
//! Corresponds to 05-api-contract, 09-engine-assembly.

pub mod consolidate_api;
pub mod dump_api;
pub mod explain_api;
pub mod feedback_api;
pub mod inspect_api;
pub mod list_api;
pub mod retrieve_api;
pub mod runtime;
pub mod traverse_api;
pub mod write_api;

use hippmem_core::config::{AlgoParams, EmbedderConfig};
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::AssociationLink;
use hippmem_core::model::understanding::MemoryUnderstanding;
use hippmem_core::model::unit::{MemoryStage, WriteContext};
use hippmem_model::registry::{build_embedder, BackendSelection};
use hippmem_model::traits::Embedder;
use hippmem_store::fulltext::FulltextIndex;
use hippmem_store::semantic::binary::BinaryCodeIndex;
use hippmem_store::semantic::hnsw::FlatVectorIndex;
use hippmem_store::store::{RedbStore, Store};
use parking_lot::RwLock;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;

// ── EngineError ──

/// Engine external error type: converts lower-layer errors into a unified external error code.
///
/// Corresponds to 05 §7. MUST NOT expose underlying library types (constitution C2).
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Underlying storage error.
    #[error("store: {0}")]
    Store(String),

    /// Memory not found.
    #[error("not found: {0:?}")]
    NotFound(MemoryId),

    /// Invalid input parameter.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Incompatible schema version.
    #[error("schema too new: {0}")]
    SchemaTooNew(u16),

    /// Model invocation failed (non-fatal).
    #[error("model: {0}")]
    Model(String),

    /// Backend unavailable (API key missing or network error).
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    /// Internal error.
    #[error("internal: {0}")]
    Internal(String),
}

/// Engine-layer Result alias.
pub type EngineResult<T> = Result<T, EngineError>;

// ── Write types (05 §1) ──

/// Input for writing a memory.
pub struct WriteMemoryInput {
    pub content: String,
    pub content_type: Option<hippmem_core::model::enums::ContentType>,
    pub context: WriteContext,
    pub importance_hint: Option<f32>,
    pub source_refs: Vec<hippmem_core::model::unit::SourceRef>,
}

/// Output of writing a memory.
pub struct WriteMemoryOutput {
    pub memory_id: MemoryId,
    pub stage_reached: MemoryStage,
    pub created_links: Vec<AssociationLink>,
    pub understanding: MemoryUnderstanding,
    pub warnings: Vec<WriteWarning>,
}

// ── Retrieval types (05 §2) ──

use hippmem_core::model::links::{RecallChannel, RetrievalResult};

/// Retrieval context.
#[derive(Debug, Clone, Default)]
pub struct RetrieveContext {
    pub conversation_id: Option<u64>,
    pub session_id: Option<u64>,
    pub project_id: Option<u64>,
    pub task_id: Option<u64>,
    pub user_id: Option<u64>,
    pub recent_memory_ids: Vec<MemoryId>,
}

/// Retrieval input.
pub struct RetrieveInput {
    pub query: String,
    pub context: RetrieveContext,
    pub top_k: usize,
    pub max_hops: Option<usize>,
    pub retrieval_mode: hippmem_core::model::links::RetrievalMode,
}

/// Retrieval output.
pub struct RetrieveOutput {
    pub results: Vec<RetrievalResult>,
    pub trace: RetrievalTrace,
    pub diagnostics: RetrievalDiagnostics,
}

/// Retrieval trace.
pub struct RetrievalTrace {
    pub seeds: Vec<SeedRecord>,
    pub steps: Vec<hippmem_core::model::links::ActivationStep>,
    pub hops_used: u8,
    pub merged_count: usize,
}

/// Seed record.
pub struct SeedRecord {
    pub id: MemoryId,
    pub channel: RecallChannel,
    pub initial_energy: f32,
    /// V9: in-channel rank (0 = best), for RRF diagnostics
    pub rank_in_channel: Option<usize>,
}

/// Retrieval diagnostics.
pub struct RetrievalDiagnostics {
    pub channel_contributions: Vec<(RecallChannel, u32)>,
    pub reranked: bool,
    pub pruned_branches: u32,
    pub backend_used: BackendUsage,
    pub latency_ms: u32,
}

/// Backend usage info.
pub struct BackendUsage {
    pub embedder: String,
    pub reranker: Option<String>,
}

// ── Feedback types (05 §6) ──

/// Usage feedback input.
pub struct FeedbackInput {
    pub retrieval_id: u64,
    pub used_memory_ids: Vec<MemoryId>,
    pub signal: UsageSignal,
}

// ── Consolidation types (05 §5) ──

/// Consolidation scope.
#[derive(Debug, Clone)]
pub enum ConsolidationScope {
    Full,
    Incremental,
    ByMemoryType(hippmem_core::model::enums::ContentType),
    ByTimeRange {
        from: hippmem_core::time::Timestamp,
        to: hippmem_core::time::Timestamp,
    },
    Reindex,
    EdgesOnly,
}

// ── Explain/diagnostics types (05 §4 §6) ──

use hippmem_core::model::links::LinkType;
use hippmem_core::model::unit::{MemoryLifecycle, MemoryUnit};

/// Explain output.
pub struct Explanation {
    pub memory_id: MemoryId,
    pub content_summary: String,
    pub current_importance: f32,
    pub linked: Vec<LinkSummary>,
    pub corrections: Vec<MemoryId>,
    pub contradictions: Vec<MemoryId>,
    pub recent_activations: u32,
}

pub struct LinkSummary {
    pub target: MemoryId,
    pub link_type: LinkType,
    pub strength: f32,
}

pub enum InspectQuery {
    Memory(MemoryId),
    Edges(MemoryId),
    Channel(RecallChannel),
    StoreStats,
    QueueStatus,
    StrongestEdges { limit: usize },
    Contradictions { limit: usize },
}

pub enum InspectReport {
    Memory(Box<MemoryInspect>),
    StoreStats(StoreStats),
    QueueStatus(QueueStatus),
}

pub struct MemoryInspect {
    pub unit: MemoryUnit,
    pub out_edges: Vec<EdgeView>,
    pub in_edges: Vec<EdgeView>,
    pub stage: MemoryStage,
    pub lifecycle: MemoryLifecycle,
}

#[derive(Debug, Clone)]
pub struct EdgeView {
    pub from: MemoryId,
    pub to: MemoryId,
    pub link_type: LinkType,
    pub strength: f32,
    pub confidence: f32,
    pub activation_count: u32,
    pub evidence: String,
}

pub struct StoreStats {
    pub memory_count: u64,
    pub edge_count: u64,
    pub observing_edge_count: u64,
    pub per_index_size: Vec<(RecallChannel, u64)>,
    pub queue_backlog: u64,
    pub store_bytes: u64,
}

pub struct QueueStatus {
    pub pending_enrich: u64,
    pub pending_consolidate: u64,
    pub in_flight: u64,
    pub oldest_pending_age_ms: u64,
}

/// Consolidation report.
pub struct ConsolidationReport {
    pub memories_processed: u64,
    pub edges_decayed: u64,
    pub edges_archived: u64,
    pub edges_merged: u64,
    pub observation_promoted: u64,
    pub summaries_created: u64,
    pub contradictions_found: u64,
    pub reindexed: bool,
    pub elapsed_ms: u64,
}

/// Usage signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageSignal {
    Referenced,
    UserConfirmedCorrect,
    TaskSucceeded,
    UserRejected,
}

/// Write warning.
#[derive(Debug, Clone, PartialEq)]
pub enum WriteWarning {
    /// Degraded extractor was used
    ExtractorDegraded,
    /// Dense vector generation deferred
    EmbeddingDeferred,
    /// Strong semantic dimensions deferred to enrich
    StrongDimsDeferred,
    /// Model invocation failed and was degraded
    ModelError { detail: String },
}

// ── List API types ──

/// Input parameters for paginated memory listing.
#[derive(Debug, Clone)]
pub struct ListInput {
    /// Page size, default 20, max 100.
    pub limit: usize,
    /// Cursor: pass the MemoryId (u128 value) of the last item on the previous page to get the next page.
    pub cursor: Option<u128>,
    /// Filter by ContentType; None = no filter.
    pub content_type: Option<hippmem_core::model::enums::ContentType>,
}

impl Default for ListInput {
    fn default() -> Self {
        Self {
            limit: 20,
            cursor: None,
            content_type: None,
        }
    }
}

/// Output of paginated memory listing.
#[derive(Debug, Clone, Serialize)]
pub struct ListOutput {
    pub items: Vec<ListItem>,
    /// Cursor for the next page; None means this is the last page.
    pub next_cursor: Option<u128>,
    /// Total memory count (approximate).
    pub total: u64,
}

/// Summary of a single memory in the list.
#[derive(Debug, Clone, Serialize)]
pub struct ListItem {
    pub id: hippmem_core::ids::MemoryId,
    /// Content preview: first 100 chars of raw.
    pub content_preview: String,
    pub content_type: hippmem_core::model::enums::ContentType,
    pub created_at: hippmem_core::time::Timestamp,
    pub importance: f32,
    pub stage: hippmem_core::model::unit::MemoryStage,
    pub lifecycle: hippmem_core::model::unit::MemoryLifecycle,
    /// Number of outgoing edges of this memory.
    pub edge_count: usize,
}

// ── Dump API types ──

/// Full export input parameters.
#[derive(Debug, Clone, Default)]
pub struct DumpInput {
    /// Output file path; None = return a JSON string.
    pub output_path: Option<std::path::PathBuf>,
}

/// Full export output.
#[derive(Debug, Clone, Serialize)]
pub struct DumpOutput {
    pub count: u64,
    /// Path echo when written to a file.
    pub written_to: Option<std::path::PathBuf>,
    /// JSONL string returned when output_path is None.
    pub json: Option<String>,
}

// ── Traverse API types ──

/// Graph traversal input parameters.
#[derive(Debug, Clone)]
pub struct TraverseInput {
    /// Start memory ID.
    pub start_id: hippmem_core::ids::MemoryId,
    /// BFS max depth, default 2, max 5.
    pub max_depth: u8,
    /// Traversal direction.
    pub direction: TraverseDirection,
    /// Filter edges by LinkType; None = no filter.
    pub link_types: Option<Vec<hippmem_core::model::links::LinkType>>,
}

impl TraverseInput {
    /// Creates default traversal params from the specified ID (depth=2, outgoing, no filter).
    pub fn new(start_id: hippmem_core::ids::MemoryId) -> Self {
        Self {
            start_id,
            max_depth: 2,
            direction: TraverseDirection::Outgoing,
            link_types: None,
        }
    }
}

impl Default for TraverseInput {
    fn default() -> Self {
        Self {
            start_id: hippmem_core::ids::MemoryId(0),
            max_depth: 2,
            direction: TraverseDirection::Outgoing,
            link_types: None,
        }
    }
}

/// Traversal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraverseDirection {
    /// Only outgoing edges.
    Outgoing,
    /// Only incoming edges.
    Incoming,
    /// Both directions.
    Both,
}

/// Graph traversal output.
#[derive(Debug, Clone)]
pub struct TraverseOutput {
    /// Nodes visited by BFS (excluding the start node).
    pub nodes: Vec<TraverseNode>,
    /// Edges traversed.
    pub edges: Vec<EdgeView>,
}

/// Node in BFS traversal.
#[derive(Debug, Clone)]
pub struct TraverseNode {
    pub id: hippmem_core::ids::MemoryId,
    /// BFS depth: 1 = direct neighbor, 2 = neighbor of neighbor...
    pub depth: u8,
    pub content_preview: String,
    pub content_type: hippmem_core::model::enums::ContentType,
    pub importance: f32,
}

// ── Conversion from lower-layer errors ──

impl From<hippmem_store::store::StoreError> for EngineError {
    fn from(e: hippmem_store::store::StoreError) -> Self {
        EngineError::Store(e.to_string())
    }
}

// ── EngineConfig ──

/// Background worker configuration.
#[derive(Debug, Clone)]
pub struct BackgroundConfig {
    /// Strong-semantic enrich concurrency, default 2.
    pub enrich_workers: usize,
    /// Consolidation concurrency, default 1.
    pub consolidate_workers: usize,
    /// Background queue capacity (bounded), default 4096.
    pub queue_capacity: usize,
    /// Periodic consolidation trigger interval (ms), default 3_600_000 (1h).
    pub consolidate_interval_ms: u64,
    /// Whether to enable enrich, default true.
    pub enrich_enabled: bool,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            enrich_workers: 2,
            consolidate_workers: 1,
            queue_capacity: 4096,
            consolidate_interval_ms: 3_600_000,
            enrich_enabled: true,
        }
    }
}

/// Engine construction configuration.
///
/// Corresponds to 05 §0. Configures persistence path, algorithm params,
/// model backend selection, and background workers.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Storage directory (the redb file will be created under this directory).
    pub store_dir: PathBuf,
    /// Algorithm params, defaults to `AlgoParams::default()`.
    pub algo: AlgoParams,
    /// Embedder backend config, defaults to deterministic 256d SimHash (matches V3 behavior).
    pub embedder: EmbedderConfig,
    /// Backend selection (extractor/reranker/summarizer), all default to `Auto`.
    pub backend: BackendSelection,
    /// Background worker configuration.
    pub background: BackgroundConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            store_dir: PathBuf::from("./hippmem_data"),
            algo: AlgoParams::default(),
            embedder: EmbedderConfig::default(),
            backend: BackendSelection::default(),
            background: BackgroundConfig::default(),
        }
    }
}

// ── Engine ──

/// HIPPMEM unified orchestration facade.
///
/// Holds the persistent storage, model registry, and algorithm params,
/// and exposes seven core APIs externally.
///
/// Corresponds to 05 §0, 09 §1.
pub struct Engine {
    /// Persistent storage (redb).
    store: Arc<RedbStore>,
    /// Algorithm params (hot-swappable).
    #[allow(dead_code)]
    params: Arc<RwLock<AlgoParams>>,
    /// Embedder backend (config-driven, default deterministic 256d SimHash).
    embedder: Arc<dyn Embedder>,
    /// Backend config (extractor/reranker/summarizer; embedder migrated to `self.embedder`).
    #[allow(dead_code)]
    backend: BackendSelection,
    /// Tantivy fulltext index (fulltext/ subdirectory of store dir; internal Mutex supports &self writes).
    fulltext_index: parking_lot::Mutex<FulltextIndex>,
    /// Tantivy fulltext index directory path (used for Reindex rebuild).
    fulltext_dir: PathBuf,
    /// Binary code index (in-memory Hamming distance recall, 03 §4.5 SemanticBinary channel).
    binary_code_index: parking_lot::Mutex<BinaryCodeIndex>,
    /// Dense vector index (in-memory brute-force L2 KNN, 03 §4.5 SemanticDense channel).
    dense_vector_index: parking_lot::Mutex<FlatVectorIndex>,
}

impl Engine {
    /// Opens/creates a HIPPMEM memory store.
    ///
    /// Corresponds to 05 §0 `Engine::open`.
    /// - Automatically creates the `store_dir` parent directory.
    /// - If a redb file already exists at the specified path, opens the existing store.
    /// - Builds the Embedder backend from `config.embedder` (default deterministic 256d, constitution C5).
    pub fn open(config: EngineConfig) -> EngineResult<Self> {
        // Automatically create parent directory
        if let Some(parent) = config.store_dir.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                EngineError::Store(format!("cannot create storage directory: {}", e))
            })?;
        }

        // Build Embedder backend (config-driven)
        let embedder =
            build_embedder(&config.embedder).map_err(|e| EngineError::Model(e.to_string()))?;

        // Open/create redb storage
        let store = RedbStore::open(&config.store_dir)?;

        // Create/open Tantivy fulltext index (same dir as redb, fulltext/ subdirectory)
        let fulltext_dir = config
            .store_dir
            .parent()
            .map(|p| p.join("fulltext"))
            .unwrap_or_else(|| PathBuf::from("hippmem_data").join("fulltext"));
        let fulltext_index = FulltextIndex::open(&fulltext_dir)
            .or_else(|_| FulltextIndex::create(&fulltext_dir))
            .map_err(|e| {
                EngineError::Store(format!("Tantivy index initialization failed: {}", e))
            })?;

        Ok(Self {
            store: Arc::new(store),
            params: Arc::new(RwLock::new(config.algo)),
            embedder,
            backend: config.backend,
            fulltext_index: parking_lot::Mutex::new(fulltext_index),
            fulltext_dir,
            binary_code_index: parking_lot::Mutex::new(BinaryCodeIndex::new()),
            dense_vector_index: parking_lot::Mutex::new(FlatVectorIndex::new()),
        })
    }

    /// Graceful shutdown.
    ///
    /// Corresponds to 05 §0 `Engine::close`.
    /// Currently only drops the store (redb auto-flushes); in the future it will wait for background workers to exit.
    pub fn close(self) -> EngineResult<()> {
        // Tantivy: commit unwritten documents and close
        if let Err(e) = self.fulltext_index.lock().flush() {
            // Non-fatal; tracing warn, does not block close
            eprintln!("Tantivy flush failed: {}", e);
        }
        // store is dropped; redb auto-flushes and closes
        drop(self.store);
        Ok(())
    }

    /// Sets the fulltext index batch commit interval (auto commit every N entries).
    /// Only used in batch write scenarios; production defaults to per-entry commit.
    pub fn set_fulltext_commit_every(&self, n: usize) {
        self.fulltext_index.lock().set_commit_every(n);
    }

    /// Force-commits all unwritten documents in the fulltext index.
    pub fn flush_fulltext(&self) {
        let _ = self.fulltext_index.lock().flush();
    }
}
