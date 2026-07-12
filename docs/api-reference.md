# API Reference

> This document lists the signatures, input/output types, error codes, and minimal examples for all 7 public methods of `Engine`.
> Pair it with the [User Guide](user-guide.md) for concepts and the [Cookbook](cookbook.md) for real-world scenarios.

---

## Engine Lifecycle

### `Engine::open(config: EngineConfig) -> EngineResult<Engine>`

Creates or opens a memory engine instance. The storage directory is created automatically if it does not exist.

**EngineConfig fields:**

| Field | Type | Default | Description |
|------|------|--------|------|
| `store_dir` | `PathBuf` | `"./hippmem_data"` | Storage directory (redb file + Tantivy index) |
| `algo` | `AlgoParams` | `AlgoParams::default()` | All algorithm parameters, see [Configuration Reference](configuration.md) |
| `backend` | `BackendSelection` | `BackendSelection::default()` | Model backend selection (`Auto` / `DeterministicOnly` / explicit API) |
| `background` | `BackgroundConfig` | `BackgroundConfig::default()` | Background worker configuration |

```rust
use hippmem_engine::{Engine, EngineConfig};
use std::path::PathBuf;

let config = EngineConfig {
    store_dir: PathBuf::from("./my_memory"),
    ..Default::default()
};
let engine = Engine::open(config)?;
```

### `engine.close(self) -> EngineResult<()>`

Closes the engine, waits for background workers to exit gracefully, and flushes all unpersisted data.

```rust
engine.close()?;
```

---

## Write

### `engine.write(input: WriteMemoryInput) -> EngineResult<WriteMemoryOutput>`

Synchronously writes a single memory. When it returns, the memory is at the `Indexed` stage (inverted index + full-text index built + association links discovered).
Strong semantic dimensions (goal/preference/emotion/decision) are populated asynchronously during the background `enrich` stage.

**WriteMemoryInput fields:**

| Field | Type | Description |
|------|------|------|
| `content` | `String` | Raw text content of the memory |
| `content_type` | `Option<ContentType>` | Content type, see enum below |
| `context` | `WriteContext` | Write context (session/project/user IDs, etc.) |
| `importance_hint` | `Option<f32>` | Importance hint, range [0, 1] |
| `source_refs` | `Vec<SourceRef>` | Source references (external docs, URLs, etc.) |

**ContentType variants:**

| Variant | Description |
|------|------|
| `UserStatement` | User statement |
| `Decision` | Decision |
| `Preference` | Preference |
| `Event` | Event |
| `TaskState` | Task state |
| `ProjectKnowledge` | Project knowledge |
| `Reflection` | Reflection / summary |
| `Correction` | Correction |

**WriteMemoryOutput fields:**

| Field | Type | Description |
|------|------|------|
| `memory_id` | `MemoryId` | ID of the new memory |
| `stage_reached` | `MemoryStage` | Stage reached (normally `Indexed`) |
| `created_links` | `Vec<AssociationLink>` | Association links created by this write |
| `understanding` | `MemoryUnderstanding` | Structured understanding extracted immediately |
| `warnings` | `Vec<WriteWarning>` | Write warnings, if any |

**WriteWarning variants:**

| Variant | Meaning |
|------|------|
| `ExtractorDegraded` | A degraded extractor was used (no API backend) |
| `EmbeddingDeferred` | Dense vector generation deferred |
| `StrongDimsDeferred` | Strong semantic dimensions (goal/preference/emotion/decision) deferred to enrich |
| `ModelError { detail }` | Model call failed and was degraded |

**Error codes:**

| Error variant | HTTP analog | gRPC mapping |
|----------|----------|----------|
| `InvalidInput` | 400 | `InvalidArgument` |
| `Store` | 500 | `Internal` |
| `BackendUnavailable` | 503 | `Unavailable` |

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput};
use hippmem_core::model::enums::ContentType;

let engine = Engine::open(EngineConfig::default())?;

let output = engine.write(WriteMemoryInput {
    content: "Rewrote the data import pipeline in Rust, achieving a 3x performance improvement.".into(),
    content_type: Some(ContentType::Decision),
    context: Default::default(),
    importance_hint: Some(0.7),
    source_refs: vec![],
})?;

println!("memory_id={} stage={:?} links={}",
    output.memory_id.0,
    output.stage_reached,
    output.created_links.len()
);
```

---

## Retrieval

### `engine.retrieve(input: RetrieveInput) -> EngineResult<RetrieveOutput>`

Retrieves relevant memories. Internal pipeline: multi-channel seed recall -> spreading activation -> reranking -> explanation path.

**RetrieveInput fields:**

| Field | Type | Description |
|------|------|------|
| `query` | `String` | Query text |
| `context` | `RetrieveContext` | Retrieval context (session/project IDs, etc.), used for temporal/spatial preferences |
| `top_k` | `usize` | Maximum number of results to return |
| `max_hops` | `Option<usize>` | Maximum spreading hops (`None` uses the default of 2) |
| `retrieval_mode` | `RetrievalMode` | Retrieval mode: `Balanced` / `Fast` / `Deep` / `Diagnostic` |

**RetrievalMode variants:**

| Variant | Description |
|------|------|
| `Balanced` | Default: multi-channel + 2-hop spreading, balances speed and quality |
| `Fast` | Single hop, faster response |
| `Deep` | 3 hops, more comprehensive recall (slower) |
| `Diagnostic` | Returns full diagnostics, for debugging and evaluation |

**RetrieveOutput fields:**

| Field | Type | Description |
|------|------|------|
| `results` | `Vec<RetrievalResult>` | Sorted result list |
| `trace` | `RetrievalTrace` | Retrieval trace (seeds -> spreading steps) |
| `diagnostics` | `RetrievalDiagnostics` | Diagnostics (channel contributions / latency, etc.) |

**RetrievalResult fields (from `hippmem_core`):**

| Field | Type | Description |
|------|------|------|
| `memory` | `MemoryUnit` | The memory unit |
| `final_score` | `f32` | Final score [0, 1] |
| `matched_dimensions` | `Vec<MatchDimension>` | List of matched dimensions |
| `activation_trace` | `Vec<ActivationStep>` | Activation trace (how it was found) |
| `channel_contributions` | `Vec<(RecallChannel, f32)>` | Contribution score per channel |
| `warnings` | `Vec<MemoryWarning>` | Risk warnings, if any |

```rust
use hippmem_engine::{Engine, EngineConfig, RetrieveInput, RetrieveContext};
use hippmem_core::model::links::RetrievalMode;

let results = engine.retrieve(RetrieveInput {
    query: "How was the data import pipeline optimized?".into(),
    context: RetrieveContext::default(),
    top_k: 5,
    max_hops: Some(2),
    retrieval_mode: RetrievalMode::Balanced,
})?;

for r in &results.results {
    println!("[{:.3}] {} (dims: {:?})",
        r.final_score,
        r.memory.content.raw.chars().take(60).collect::<String>(),
        r.matched_dimensions
    );
}

// Diagnostics
println!("latency: {}ms", results.diagnostics.latency_ms);
println!("channel_contributions: {:?}", results.diagnostics.channel_contributions);
```

---

## Explain

### `engine.explain(memory_id: MemoryId, context: Option<RetrieveContext>) -> EngineResult<Explanation>`

Answers five questions: source / importance / connections / correction conflicts / recent activations.

**Explanation fields:**

| Field | Type | Description |
|------|------|------|
| `memory_id` | `MemoryId` | Memory ID |
| `content_summary` | `String` | Content summary |
| `current_importance` | `f32` | Current importance [0, 1] |
| `linked` | `Vec<LinkSummary>` | Other memories linked to this one |
| `corrections` | `Vec<MemoryId>` | Older memories corrected by this memory |
| `contradictions` | `Vec<MemoryId>` | Memories that contradict this one |
| `recent_activations` | `u32` | Number of recent activations |

**LinkSummary fields:**

| Field | Type | Description |
|------|------|------|
| `target` | `MemoryId` | Target memory ID |
| `link_type` | `LinkType` | Association type (`Causal`/`EntityOverlap`/`SemanticSimilar`/...) |
| `strength` | `f32` | Association strength [0, 1] |

```rust
let explanation = engine.explain(memory_id, None)?;

println!("importance: {:.3}", explanation.current_importance);
println!("linked_memories: {} ", explanation.linked.len());
for link in &explanation.linked {
    println!("  → {:?} {:?} (strength={:.3})", link.target, link.link_type, link.strength);
}
if !explanation.contradictions.is_empty() {
    println!("⚠ contradictions found: {:?}", explanation.contradictions);
}
```

---

## Consolidation

### `engine.consolidate(scope: ConsolidationScope) -> EngineResult<ConsolidationReport>`

Runs a consolidation cycle: Hebbian strengthening -> decay -> compaction -> summary compression.

**ConsolidationScope variants:**

| Variant | Description |
|------|------|
| `Full` | Consolidate all memories fully |
| `Incremental` | Incremental consolidation (only memories added/changed since the last consolidation) |
| `ByMemoryType(ContentType)` | Consolidate within a memory type scope |
| `ByTimeRange { from, to }` | Consolidate within a time range |
| `Reindex` | Rebuild all indexes |
| `EdgesOnly` | Only process edge decay and compaction |

**ConsolidationReport fields:**

| Field | Type | Description |
|------|------|------|
| `memories_processed` | `u64` | Number of memories processed |
| `edges_decayed` | `u64` | Number of edges decayed |
| `edges_archived` | `u64` | Number of weak edges archived |
| `edges_merged` | `u64` | Number of edges merged |
| `observation_promoted` | `u64` | Number of edges promoted from the observation zone |
| `summaries_created` | `u64` | Number of summary memories created |
| `contradictions_found` | `u64` | Number of contradiction pairs detected |
| `reindexed` | `bool` | Whether indexes were rebuilt |
| `elapsed_ms` | `u64` | Elapsed time (milliseconds) |

```rust
use hippmem_engine::ConsolidationScope;

// Incremental consolidation (regular scheduled task)
let report = engine.consolidate(ConsolidationScope::Incremental)?;
println!("decayed {} edges, archived {}", report.edges_decayed, report.edges_archived);

// Full consolidation (after deployment or after a long gap without consolidation)
let report = engine.consolidate(ConsolidationScope::Full)?;

// Edges only
let report = engine.consolidate(ConsolidationScope::EdgesOnly)?;
```

---

## Diagnostics

### `engine.inspect(query: InspectQuery) -> EngineResult<InspectReport>`

Queries the engine's internal state, for monitoring and debugging.

**InspectQuery variants:**

| Variant | Returns | Description |
|------|------|------|
| `Memory(MemoryId)` | `InspectReport::Memory` | View memory details and its in/out edges |
| `Edges(MemoryId)` | `InspectReport::Memory` | View edge details for a memory |
| `Channel(RecallChannel)` | `InspectReport::StoreStats` | Filter statistics by channel |
| `StoreStats` | `InspectReport::StoreStats` | Storage statistics |
| `QueueStatus` | `InspectReport::QueueStatus` | Background queue status |
| `StrongestEdges { limit }` | `InspectReport::Memory` | Ranking of strongest edges |
| `Contradictions { limit }` | `InspectReport::Memory` | Contradiction detection results |

**StoreStats fields:**

| Field | Type | Description |
|------|------|------|
| `memory_count` | `u64` | Total memory count |
| `edge_count` | `u64` | Total edge count |
| `observing_edge_count` | `u64` | Edges in the observation zone |
| `per_index_size` | `Vec<(RecallChannel, u64)>` | Index size per channel |
| `queue_backlog` | `u64` | Background queue backlog |
| `store_bytes` | `u64` | Storage file size (bytes) |

**QueueStatus fields:**

| Field | Type | Description |
|------|------|------|
| `pending_enrich` | `u64` | Memories waiting for enrich |
| `pending_consolidate` | `u64` | Memories waiting for consolidation |
| `in_flight` | `u64` | Tasks currently being processed |
| `oldest_pending_age_ms` | `u64` | Wait time of the oldest pending task |

```rust
use hippmem_engine::InspectQuery;

// Storage statistics
let report = engine.inspect(InspectQuery::StoreStats)?;
if let hippmem_engine::InspectReport::StoreStats(s) = report {
    println!("memories: {} edges: {} backlog: {}",
        s.memory_count, s.edge_count, s.queue_backlog);
}

// Queue status
let report = engine.inspect(InspectQuery::QueueStatus)?;
if let hippmem_engine::InspectReport::QueueStatus(q) = report {
    println!("pending enrich: {} pending consolidate: {} in flight: {}",
        q.pending_enrich, q.pending_consolidate, q.in_flight);
}
```

---

## Feedback

### `engine.feedback(input: FeedbackInput) -> EngineResult<()>`

Records usage feedback signals that drive Hebbian learning — association links between memories frequently used together are automatically strengthened.

**FeedbackInput fields:**

| Field | Type | Description |
|------|------|------|
| `retrieval_id` | `u64` | Retrieval request ID |
| `used_memory_ids` | `Vec<MemoryId>` | Memory IDs the user actually used |
| `signal` | `UsageSignal` | Usage signal |

**UsageSignal variants:**

| Variant | Meaning | Hebbian effect |
|------|------|-------------|
| `Referenced` | User referenced this memory | Small strengthening of association links |
| `UserConfirmedCorrect` | User confirmed the result is correct | Moderate strengthening of association links |
| `TaskSucceeded` | A task based on this memory succeeded | Large strengthening of association links |
| `UserRejected` | User rejected / flagged as wrong | Weakening of association links |

```rust
use hippmem_engine::FeedbackInput;
use hippmem_engine::UsageSignal;

engine.feedback(FeedbackInput {
    retrieval_id: 1,
    used_memory_ids: vec![memory_id],
    signal: UsageSignal::UserConfirmedCorrect,
})?;
```

---

## Error Code Quick Reference

| EngineError variant | Meaning | gRPC Status |
|-------------------|------|-------------|
| `Store(msg)` | Underlying storage error | `Internal` |
| `NotFound(id)` | Memory not found | `NotFound` |
| `InvalidInput(msg)` | Invalid input parameter | `InvalidArgument` |
| `SchemaTooNew(v)` | Incompatible schema version | `FailedPrecondition` |
| `Model(msg)` | Model call failed (non-fatal) | `Internal` |
| `BackendUnavailable(msg)` | Backend unavailable | `Unavailable` |
| `Internal(msg)` | Internal error | `Internal` |

---

## gRPC Mapping

| Rust API | gRPC RPC | Request Message | Response Message |
|----------|----------|----------------|------------------|
| `engine.write()` | `Write` | `WriteRequest` | `WriteResponse` |
| `engine.retrieve()` | `Retrieve` | `RetrieveRequest` | `RetrieveResponse` |
| `engine.explain()` | `Explain` | `ExplainRequest` | `ExplainResponse` |
| `engine.consolidate()` | `Consolidate` | `ConsolidateRequest` | `ConsolidateResponse` |
| `engine.inspect()` | `Inspect` | `InspectRequest` | `InspectResponse` |
| `engine.feedback()` | `Feedback` | `FeedbackRequest` | `FeedbackResponse` |

For the full proto definitions, see the [gRPC Guide](grpc-guide.md).
