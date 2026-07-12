# API Contract

> This document defines HIPPMEM's public Rust API: the `Engine` entry point, its methods, input/output types, error codes, and the precise synchronous/asynchronous boundary. It is the canonical reference for the in-process API; the gRPC service is a thin wrapper that maps these types to proto messages.

All public API methods live on `Engine` in the `hippmem-engine` crate. Element types (`MemoryUnit`, `ContentType`, `LinkType`, etc.) are defined in `hippmem-core`.

---

## 1. The `Engine` Entry Point

```rust
pub struct Engine { /* Store + ModelRegistry + AlgoParams + runtime */ }

impl Engine {
    pub async fn open(config: EngineConfig) -> EngineResult<Self>;
    pub async fn write(&self, input: WriteMemoryInput) -> EngineResult<WriteMemoryOutput>;
    pub async fn retrieve(&self, input: RetrieveInput) -> EngineResult<RetrieveOutput>;
    pub async fn explain(&self, id: MemoryId, ctx: Option<RetrieveContext>) -> EngineResult<Explanation>;
    pub async fn consolidate(&self, scope: ConsolidationScope) -> EngineResult<ConsolidationReport>;
    pub async fn inspect(&self, query: InspectQuery) -> EngineResult<InspectReport>;
    pub async fn feedback(&self, input: FeedbackInput) -> EngineResult<()>;
    pub async fn close(self) -> EngineResult<()>;
}
```

`open` creates or opens a store, assembles model backends (Auto mode: use API backends when keys are present, otherwise the deterministic fallback), and starts background workers. `close` flushes the background queue to a safe point, persists any remaining tasks for the next start, and closes the store.

### `EngineConfig`

```rust
pub struct EngineConfig {
    pub store_dir: PathBuf,
    pub algo: AlgoParams,            // algorithm parameter table; defaults are constants
    pub backend: BackendSelection,   // per-capability Api / Deterministic / Auto
    pub background: BackgroundConfig,
}

pub struct BackgroundConfig {
    pub enrich_workers: usize,           // default 2
    pub consolidate_workers: usize,      // default 1
    pub queue_capacity: usize,           // default 4096 (bounded, backpressure)
    pub consolidate_interval_ms: u64,    // default 3_600_000 (1h)
    pub enrich_enabled: bool,            // default true
    pub on_queue_full: QueueFullPolicy,
}

pub enum QueueFullPolicy {
    BlockProducer,    // default: briefly back-pressure the writer
    DropLowValue,     // drop low-importance non-strong-semantic tasks
}
```

The queue policy distinguishes strong-semantic enrich tasks (which must not be dropped, only delayed) from low-value tasks (which may be dropped under pressure).

---

## 2. `write` — Write a Memory

```rust
pub struct WriteMemoryInput {
    pub content: String,
    pub content_type: Option<ContentType>,   // defaulted by Extractor, then UserStatement
    pub context: WriteContext,               // required struct, fields optional
    pub importance_hint: Option<f32>,        // [0,1]
    pub source_refs: Vec<SourceRef>,
}

pub struct WriteMemoryOutput {
    pub memory_id: MemoryId,
    pub stage_reached: MemoryStage,           // Indexed when returning synchronously
    pub created_links: Vec<AssociationLink>,
    pub understanding: MemoryUnderstanding,
    pub warnings: Vec<WriteWarning>,
}

pub enum WriteWarning {
    ExtractorDegraded,            // used the fallback backend
    EmbeddingDeferred,            // dense vector computed later
    StrongDimsDeferred,           // strong semantic dims pushed to background enrich
    ModelError { detail: String },
}
```

### Semantics

- The synchronous path completes through the `Indexed` stage (basic immediate dimensions + initial association links + index writes), then returns.
- Strong semantic dimensions (goals, preferences, emotions, decisions, implicit causals, contradictions) and any deferred dense vector are computed by background enrich workers and merged back into the `MemoryUnit`.
- A model call failure **must not** cause `write` to fail: the engine falls back to the deterministic backend or leaves the field empty and adds a `WriteWarning`. The raw content is always persisted first.
- `memory_log` appends the raw record immediately, so a crash never loses the original memory.

---

## 3. `retrieve` — Multi-Channel Recall

```rust
pub struct RetrieveInput {
    pub query: String,
    pub context: RetrieveContext,
    pub top_k: usize,                  // default 10
    pub max_hops: Option<usize>,       // None => determined by mode
    pub retrieval_mode: RetrievalMode, // default Balanced
}

pub struct RetrieveContext {
    pub conversation_id: Option<u64>,
    pub session_id: Option<u64>,
    pub project_id: Option<u64>,
    pub task_id: Option<u64>,
    pub user_id: Option<u64>,
    pub recent_memory_ids: Vec<MemoryId>,
}

pub struct RetrieveOutput {
    pub results: Vec<RetrievalResult>,
    pub trace: RetrievalTrace,
    pub diagnostics: RetrievalDiagnostics,
}

pub struct RetrievalTrace {
    pub retrieval_id: RetrievalId,        // for feedback to point back at this retrieval
    pub seeds: Vec<SeedRecord>,
    pub steps: Vec<ActivationStep>,
    pub hops_used: u8,
    pub merged_count: usize,
}

pub struct RetrievalDiagnostics {
    pub channel_contributions: Vec<(RecallChannel, ChannelStat)>,
    pub reranked: bool,
    pub pruned_branches: u32,
    pub backend_used: BackendUsage,
    pub latency_ms: u32,
}
```

### Semantics

- Multi-channel seeds (BM25, entity inverted, temporal proximity, semantic) are recalled, merged, and deduplicated.
- Seeds receive `initial_energy`, then spreading activation traverses the association graph with configurable `max_hops`, `fan_out`, `decay_factor`, and type modifiers.
- Results carry `activation_trace`, `matched_dimensions`, and `warnings` (contradictions, staleness, supersession).
- Per-channel contribution is queryable via `channel_contributions` — every result can be traced back to which channel(s) recalled its seed.
- `Diagnostic` mode emits the full trace.

`RetrievalId` is a `u64` newtype — a process-local monotonically increasing retrieval sequence number generated by an injected counter. It is distinct from `MemoryId` (which is a ULID/u128) and exists so `feedback` can point back at a specific retrieval.

---

## 4. `explain` — Explain a Memory

```rust
pub struct Explanation {
    pub memory: MemoryUnit,
    pub origin: Provenance,
    pub current_importance: f32,
    pub linked: Vec<LinkSummary>,
    pub corrections: Vec<MemoryId>,
    pub contradictions: Vec<MemoryId>,
    pub recent_activations: Vec<ActivationStep>,
}

pub struct LinkSummary {
    pub target: MemoryId,
    pub link_type: LinkType,
    pub strength: f32,
    pub direction: LinkDirection,
}
```

`explain` answers five questions about a memory: where it came from, how important it is, what it links to, whether it has been corrected or contradicted, and why it was recently activated. It is a pure read against the store.

---

## 5. `consolidate` — Consolidation and Evolution

```rust
pub enum ConsolidationScope {
    Full,
    Incremental,
    ByMemoryType(ContentType),
    ByTimeRange { from: Timestamp, to: Timestamp },
    Reindex,          // rebuild all indexes from memory_log (crash recovery / migration)
    EdgesOnly,        // compaction only (decay / cleanup / merge)
}

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
```

`consolidate` dispatches or triggers background work (Hebbian reinforcement, decay, compaction, summary, contradiction discovery). Depending on the scope it either returns a lightweight acknowledgement immediately or waits for completion (with a 30s timeout guard, after which it returns `Err(EngineError::Internal("consolidate timeout"))`). `Reindex` is idempotent — multiple runs produce identical results.

---

## 6. `inspect` — Diagnostics

```rust
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
    Memory(MemoryInspect),
    Edges(EdgesInspect),
    Channel(ChannelInspect),
    StoreStats(StoreStats),
    QueueStatus(QueueStatus),
    StrongestEdges(Vec<EdgeView>),
    Contradictions(Vec<ContradictionView>),
}
```

Each `InspectQuery` variant maps one-to-one to an `InspectReport` variant. All report types derive `Serialize`, so they can be emitted as JSON via `serde_json`. `inspect` is a pure read against the store and the background queue's metrics.

Notable report types:

- `StoreStats` — `memory_count`, `edge_count`, `observing_edge_count`, per-index size, `queue_backlog`, `store_bytes`.
- `QueueStatus` — `pending_enrich`, `pending_consolidate`, `in_flight`, `oldest_pending_age_ms`.
- `EdgeView` — full diagnostic view of an edge, including `evidence` (why the edge exists), `confidence`, `activation_count`, and `observation` state.
- `ContradictionView` — pairs of memories with a `Contradiction` / `Correction` / `Supersedes` link and a note.

---

## 7. `feedback` — Usage Signal

```rust
pub struct FeedbackInput {
    pub retrieval_id: Option<RetrievalId>,
    pub used_memory_ids: Vec<MemoryId>,
    pub signal: UsageSignal,
}

pub enum UsageSignal {
    Referenced,
    UserConfirmedCorrect,
    TaskSucceeded,
    UserRejected,
}
```

`feedback` records a usage signal tied to a prior retrieval. `UserConfirmedCorrect` and `TaskSucceeded` enqueue a Hebbian candidate so the corresponding edges are reinforced on the next consolidation cycle. `UserRejected` marks the related memories/edges for down-weighting (handled by the consolidate worker). The signal is recorded synchronously; reinforcement happens in the background.

---

## 8. Synchronous / Asynchronous Boundary

| API | Synchronous (before return) | Background |
|-----|------------------------------|------------|
| `write` | raw persistence + Indexed (basic dims + initial edges + indexes) | enrich (strong dims), deferred dense vector, consolidation |
| `retrieve` | entire path (seeds → spread → rerank → warnings) | post-retrieval Hebbian candidate |
| `explain` | entire path (read store) | — |
| `consolidate` | dispatch / trigger; returns acknowledgement or waits (scope-dependent) | bulk consolidation execution |
| `feedback` | record usage signal | Hebbian reinforcement applied on next cycle |
| `inspect` | entire path (read store + queue state) | — |

Performance targets (non-binding): `write` synchronous segment < 50 ms; `retrieve` Balanced < 300 ms; `retrieve` with rerank < 1 s.

---

## 9. Error Codes

```rust
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("store: {0}")]        Store(String),
    #[error("not found: {0:?}")]  NotFound(MemoryId),
    #[error("invalid input: {0}")] InvalidInput(String),
    #[error("schema too new: {0}")] SchemaTooNew(u16),
    #[error("model: {0}")]        Model(String),
    #[error("backend unavailable: {0}")] BackendUnavailable(String),
    #[error("internal: {0}")]     Internal(String),
}

pub type EngineResult<T> = Result<T, EngineError>;
```

Lower-layer `StoreError` and `ModelError` are converted at the engine boundary; lower-layer library types never appear in `EngineError`. The gRPC layer maps `EngineError` to status codes:

| `EngineError` | gRPC status |
|---------------|-------------|
| `NotFound` | `NOT_FOUND` |
| `InvalidInput` | `INVALID_ARGUMENT` |
| `BackendUnavailable` | `UNAVAILABLE` |
| `SchemaTooNew` | `FAILED_PRECONDITION` |
| `Store` / `Model` / `Internal` | `INTERNAL` |

---

## 10. Concurrency and Crash Recovery

- **Concurrent writes.** Multiple `write` calls may run concurrently. `memory_log` appends and overlay updates occur inside a redb transaction, so consistency is guaranteed. `MemoryId` uniqueness is guaranteed by ULID.
- **Write/read isolation.** `retrieve` reads committed state. A memory currently being enriched in the background participates in recall at its current stage (basic dimensions are already usable).
- **Idempotency.** `write` is not idempotent (each call produces a new id). `consolidate(Reindex)` is idempotent — multiple runs yield identical results.
- **Crash recovery.** On restart, `Engine::open` validates redb consistency. If indexes (Tantivy/HNSW) lag behind `memory_log`, it either prompts or auto-triggers `Reindex` (configurable). Original memories are never lost on crash (append-only log + transactions).
- **Graceful shutdown.** `close` flushes the background queue to a safe point and persists any remaining tasks, which resume on the next start.

---

## 11. Thread Safety and Lifetime

- `Engine` is `Send + Sync`. Internally it holds `Arc<dyn Store + Send + Sync>`, `Arc<ModelRegistry>`, and `Arc<RwLock<AlgoParams>>`, so it can be shared across threads (and across gRPC requests) via `Arc<Engine>`.
- `AlgoParams` is hot-swappable: the `RwLock` allows multiple concurrent readers while evaluation tuning writes a single updated copy.
- The engine owns its tokio runtime — callers do not need to bring their own `#[tokio::main]`. Background workers share the store and model registry via `Arc`.
- `MemoryId` is a `u128` newtype (ULID); `RetrievalId` is a `u64` newtype. Both are `Copy`, `Hash`, and `Serialize`.
