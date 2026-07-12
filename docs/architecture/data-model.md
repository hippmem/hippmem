# Data Model

> This document describes HIPPMEM's core data model: the Rust types that represent memories, associations, and activation state. These types are the single source of truth for the engine's in-memory and on-disk representation.

A memory in HIPPMEM is not a flat document. It is a structured node that carries three layers — **content**, **understanding**, and **activation history** — and is connected to other memories through typed **association links**. This document walks through each layer, the link types, and the lifecycle of a memory unit.

---

## 1. Core newtypes and aliases

All domain types live in the `hippmem-core` crate. Persisted types derive `Debug, Clone, Serialize, Deserialize`. Identifiers are wrapped in newtypes to prevent mixing; floating scores are never `Eq`/`Hash`.

```rust
/// Memory unique ID. ULID-backed, u128. Lexicographic order ≈ creation order.
pub struct MemoryId(pub u128);

/// Unix millisecond timestamp (UTC).
pub struct Timestamp(pub i64);

/// Handle into the dense-vector index (not the vector itself).
pub struct VectorId(pub u64);

/// Bounded score in 0.0..=1.0 (strength / confidence / importance). Clamps on construction.
pub struct UnitScore(f32);
```

Hash keys for entities, topics, goals, events, and causal relations are `u64` values produced by `stable_hash64` (xxh3, fixed seed 0) over the normalized canonical text. The same input text always yields the same key across processes and versions. Emotion keys are `u8` (enum discriminants); temporal keys are `u32` (time-bucket codes).

---

## 2. MemoryUnit — the core object

`MemoryUnit` is the central data object. It bundles everything the engine knows about a single memory: what it says, what it means, how it connects, and how it has been used.

```rust
pub struct MemoryUnit {
    pub schema_version: u16,          // serialization version, currently 1
    pub id: MemoryId,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,        // invariant: >= created_at

    pub content: MemoryContent,
    pub context: WriteContext,
    pub understanding: MemoryUnderstanding,
    pub association_keys: AssociationKeys,
    pub links: Vec<AssociationLink>,  // out-edges; deduplicated by (target_id, link_type)

    pub activation: ActivationState,
    pub lifecycle: MemoryLifecycle,
    pub provenance: Provenance,
    pub stage: MemoryStage,
}
```

**Invariants:**

- `updated_at >= created_at`.
- `links` contains no duplicate `(target_id, link_type)` pairs (duplicates are merged, see [Edge deduplication](#5-associationlink-and-linktype)).
- `links` contains no self-loop (`target_id != self.id`).

**Persistence:** the whole unit is stored under `memory_id -> MemoryUnit`. The immutable parts (`content.raw`, `context`, `provenance`, `created_at`) are also appended to an append-only `memory_log` for auditability. Mutable parts (link overlays, activation state, lifecycle, stage, `updated_at`) are updated through an overlay layer.

---

## 3. MemoryContent

The content layer holds the raw text, optional summary and normalized forms, language, and content type.

```rust
pub struct MemoryContent {
    pub raw: String,                  // immutable original text
    pub summary: Option<String>,      // filled in enriched/consolidated stage
    pub normalized: Option<String>,   // filled in indexed stage
    pub language: Language,
    pub content_type: ContentType,
}
```

`Language` discriminates `Zh`, `En`, `Code`, `Mixed`, and a reserved `Other(u16)` for BCP-47 numeric codes. Language selection drives tokenization.

### ContentType

Each content type carries a default importance baseline and a decay-protection level. The mapping is a **parameter**, not a hard-coded rule.

| ContentType | Importance baseline | Decay protection |
|-------------|--------------------|------------------|
| `Correction` | 0.80 | High |
| `Decision` | 0.75 | High |
| `Preference` | 0.70 | High |
| `ProjectKnowledge` | 0.65 | Medium |
| `Event` | 0.50 | Medium |
| `UserStatement` | 0.50 | Low |
| `AssistantObservation` | 0.40 | Low |
| `Reflection` | 0.45 | Low (includes summaries) |
| `TaskState` | 0.35 | Low (easily stale) |
| `ToolResult` | 0.30 | Lowest |

---

## 4. WriteContext

`WriteContext` captures the environment in which a memory was written — session, project, task IDs, local time, preceding adjacent memories, and source references. All fields may be empty in the first version, but they are retained because they feed co-context recall and provenance.

```rust
pub struct WriteContext {
    pub conversation_id: Option<u64>,
    pub session_id: Option<u64>,
    pub project_id: Option<u64>,
    pub task_id: Option<u64>,
    pub user_id: Option<u64>,
    pub local_time: Timestamp,
    pub preceding_memory_ids: Vec<MemoryId>,
    pub source_refs: Vec<SourceRef>,
}
```

`SourceRef` points back to the origin: a conversation, file, tool result, external system, or another memory (`SourceKind::MemoryRef`).

---

## 5. MemoryUnderstanding and frames

The understanding layer is the structured extraction result. It is populated in two tiers:

- **Basic immediate dimensions** (produced at the `Indexed` stage): `entities`, `topics`, explicit `causal_claims`, and an initial `importance` value. These are always available — the deterministic fallback backend can produce them.
- **Strong semantic dimensions** (filled at the `Enriched` stage): `goals`, `preferences`, `emotions`, `decisions`, implicit `causal_claims`, and `contradictions`. When a strong dimension is not yet populated, the field is an empty `Vec`, not a missing capability.

```rust
pub struct MemoryUnderstanding {
    pub entities: Vec<EntityMention>,
    pub events: Vec<EventFrame>,
    pub goals: Vec<GoalFrame>,
    pub decisions: Vec<DecisionFrame>,
    pub preferences: Vec<PreferenceFrame>,
    pub emotions: Vec<EmotionFrame>,
    pub causal_claims: Vec<CausalClaim>,
    pub contradictions: Vec<ContradictionHint>,
    pub topics: Vec<TopicTag>,
    pub importance: UnitScore,
    pub confidence: UnitScore,
}
```

Each frame is a small struct. For example:

```rust
pub struct EntityMention {
    pub text: String,           // surface form in the source
    pub canonical: String,      // normalized name; hashed into entity_key
    pub entity_type: EntityType,  // Person, Project, Library, File, Org, Concept, Other
    pub span: Option<TextSpan>,
    pub confidence: UnitScore,
}

pub struct CausalClaim {
    pub cause: String,
    pub effect: String,
    pub kind: CausalKind,       // Explicit (connective hit) or Implicit (inferred)
    pub evidence_span: Option<TextSpan>,
    pub confidence: UnitScore,
}
```

`ContradictionHint` carries a statement, an optional conflicting memory ID (may be `None` at write time), a description note, and a confidence score.

---

## 6. AssociationKeys

`AssociationKeys` are the multi-dimensional recall keys derived from a memory's content and understanding. Each key family feeds a dedicated recall channel.

```rust
pub struct AssociationKeys {
    pub entity_keys: Vec<EntityKey>,
    pub temporal_keys: Vec<TemporalKey>,
    pub lexical_signature: LexicalSignature,
    pub semantic_signature: SemanticSignature,
    pub topic_keys: Vec<TopicKey>,
    pub emotion_keys: Vec<EmotionKey>,
    pub goal_keys: Vec<GoalKey>,
    pub event_keys: Vec<EventKey>,
    pub causal_keys: Vec<CausalKey>,
}
```

### Temporal keys

`temporal_keys` are multi-granularity bucket codes generated from `created_at` and `context.local_time`: a day bucket, an hour bucket, and (if a session exists) a session bucket. Two memories sharing any bucket key are temporally adjacent.

### Lexical and semantic signatures

```rust
pub struct LexicalSignature {
    pub simhash: [u64; 4],          // 256-bit SimHash for fast literal similarity
}

pub struct SemanticSignature {
    pub lexical_simhash: [u64; 4],          // same SimHash, stored for convenience
    pub dense_embedding_ref: Option<VectorId>, // handle into the vector index
    pub binary_code: [u64; 2],              // 128-bit LSH binary code
    pub topic_minhash: [u32; 16],           // 16-permutation MinHash for topic clustering
}
```

`simhash`, `binary_code`, and `topic_minhash` are always producible by deterministic algorithms, so the engine retains semantic recall capability even when no dense embedding model is available. `dense_embedding_ref` is `None` until the embedder has run; with the deterministic backend it points to a 256-dimensional hash vector, with an API backend to a 1024- or 1536-dimensional model vector.

> Changing the embedder backend changes the vector dimensionality. A `consolidate(Reindex)` operation must rebuild the semantic index after such a switch.

---

## 7. AssociationLink and LinkType

An `AssociationLink` is a typed, weighted edge between two memories. Edges are stored on the source's `links` vector and also registered in the target's in-edge index, so both directions are queryable.

```rust
pub struct AssociationLink {
    pub target_id: MemoryId,
    pub link_type: LinkType,
    pub direction: LinkDirection,
    pub strength: UnitScore,            // mutable; updated by Hebbian learning and decay
    pub confidence: UnitScore,
    pub evidence: LinkEvidence,
    pub formed_at: Timestamp,
    pub last_activated_at: Option<Timestamp>,
    pub activation_count: u32,
    pub observation: ObservationState,
}
```

### The 14 link types

| LinkType | Semantics | Spreading behavior |
|----------|-----------|--------------------|
| `EntityOverlap` | Shared entities | Bidirectional, baseline |
| `TemporalAdjacent` | Time-bucket overlap | Bidirectional, fast-decaying |
| `SemanticSimilar` | Semantic proximity | Bidirectional |
| `TopicRelated` | Shared topics | Bidirectional |
| `SameGoal` | Belongs to same goal | Bidirectional, boosted |
| `SameEvent` | Belongs to same event | Bidirectional, boosted |
| `Causal` | Cause/effect relation | Forward-preferring, boosted |
| `EmotionalResonance` | Shared emotion | Bidirectional, dampened |
| `Contradiction` | Conflicting claims | Suppressed (signal only) |
| `Correction` | Newer corrects older | Boosts corrector |
| `Elaboration` | Expands on another | Forward |
| `CoActivation` | Habitual co-activation | Bidirectional, slight boost |
| `Supersedes` | New replaces old | Boosts new |
| `Deprecated` | Old retained as history | Suppresses old |

`LinkDirection` is `Undirected`, `Forward`, or `Backward` and expresses the *semantic* direction; storage is bidirectional regardless.

### Strong vs. weak edges

Edge strength vs. weakness is decided by a `strength` threshold (a parameter), not by type. The engine caps the number of strong edges per node (default 8, min 3) and weak edges (default 24).

### LinkEvidence

Every edge records why it was built — which match dimensions contributed, per-dimension score breakdowns, and text spans. For `Causal`, `Contradiction`, and `Correction` edges, `evidence.text_spans` **must** be non-empty; otherwise the edge is downgraded or not built.

### Observation zone

```rust
pub enum ObservationState {
    Confirmed,                        // a real edge
    Observing { since: Timestamp },   // provisional, pending confirmation
}
```

A candidate whose association score falls in the observation band is stored as `Observing` rather than dropped. If it is later retrieved, referenced, or co-activated, it is promoted to `Confirmed`. If the observation window expires without activation, compaction removes it.

### Edge deduplication

When an edge with the same `(target_id, link_type)` already exists, no new edge is created. Instead, fields are merged: `strength = max(old, new)`, evidence is unioned, `confidence = max`.

---

## 8. ActivationState

`ActivationState` tracks how a memory has been used and with whom it co-activates. It drives Hebbian learning and decay.

```rust
pub struct ActivationState {
    pub last_retrieved_at: Option<Timestamp>,
    pub retrieval_count: u32,
    pub co_activations: Vec<CoActivationCount>,  // bounded by co_activation_keep (default 16)
    pub usage_score: UnitScore,
}

pub struct CoActivationCount {
    pub with: MemoryId,
    pub count: u32,
    pub last_at: Timestamp,
}
```

`co_activations` is the authoritative store of "how many times this memory co-activated with another, even when no edge exists between them." After each retrieval, every result pair increments its counter on both sides; the oldest entry is evicted when the bounded list fills. Hebbian consolidation reads this counter to decide whether to create a new `CoActivation` edge (threshold default 3).

### ActivationStep

Each retrieval records a trace of how energy reached every returned memory.

```rust
pub struct ActivationStep {
    pub from: Option<MemoryId>,           // None = seed (direct recall)
    pub to: MemoryId,
    pub via_link: Option<LinkType>,       // None for seeds
    pub channel: Option<RecallChannel>,   // which channel produced the seed
    pub hop: u8,
    pub energy_in: f32,
    pub energy_out: f32,
}
```

---

## 9. Lifecycle and stage

A memory has two independent state machines.

### MemoryLifecycle

```rust
pub enum MemoryLifecycle {
    Active,
    Compressed { into: MemoryId },   // folded into a summary memory
    Archived,
    Superseded { by: MemoryId },     // replaced by a newer memory
    Deprecated,                      // outdated but retained
    Negated { by: MemoryId },        // explicitly negated by the user
}
```

`Active` can transition to any state; the others are relatively stable and only changed by consolidation or correction. Memories do not return to `Active` without an explicit consolidation action that records evidence.

### MemoryStage

```rust
pub enum MemoryStage { Raw, Indexed, Enriched, Consolidated }
```

The stage progression is unidirectional: `Raw → Indexed → Enriched → Consolidated`. Re-understanding produces a new version or annotates the overlay rather than rolling the stage back.

---

## 10. Provenance

`Provenance` records where a memory came from, who produced its understanding, how reliable it is, and every revision it underwent.

```rust
pub struct Provenance {
    pub origin: SourceKind,
    pub generated_by: GeneratedBy,    // UserDirect / Extractor { backend } / Consolidation / Rule
    pub reliability: UnitScore,
    pub evidence_refs: Vec<SourceRef>,
    pub revision_history: Vec<RevisionMark>,
}
```

`GeneratedBy::Extractor { backend }` carries the backend identifier (e.g. `"deterministic"` or `"openai-text-embedding-3-small"`), so the origin of every extracted understanding is traceable. Low-confidence extractions are routed to the observation zone rather than polluting the fact layer.

---

## 11. Recall and retrieval types

The data model also defines the enums used by the retrieval pipeline.

### RecallChannel

Each channel is an independent scorer. The retrieval mode selects which channels run.

```rust
pub enum RecallChannel {
    Bm25, EntityInverted, SemanticDense, SemanticBinary,
    Temporal, TopicCluster, Goal, Event, Causal,
    RecentActivation, GraphSpreading,
}
```

### MatchDimension

Used in `LinkEvidence` and in each result's `matched_dimensions`:

```rust
pub enum MatchDimension {
    Entity, Semantic, Temporal, Topic, Goal, Event,
    Emotion, Causal, CoContext, Importance,
}
```

### RetrievalMode

| Mode | Channels | Max hops | Reranker |
|------|----------|----------|----------|
| `Fast` | BM25 + Entity + SemanticDense | 1 | No |
| `Balanced` | Multi-channel | 2 | Light |
| `Deep` | All channels | 3 | Yes |
| `Diagnostic` | All channels | 3 | Yes + full trace |

### RetrievalResult

```rust
pub struct RetrievalResult {
    pub memory: MemoryUnit,
    pub final_score: f32,
    pub activation_trace: Vec<ActivationStep>,
    pub matched_dimensions: Vec<MatchDimension>,
    pub warnings: Vec<MemoryWarning>,
}
```

`MemoryWarning` flags conditions such as `HasCorrection`, `HasContradiction`, `Superseded`, `Deprecated`, `LowConfidence`, and `StaleFreshness`. Memories flagged with `Contradiction` are never promoted as high-confidence facts.

---

## 12. Serialization and schema versioning

All values persisted to the embedded database use `bincode`; traces, diagnostics, and evaluation corpora use `serde_json`.

`MemoryUnit.schema_version` is currently `1`. On deserialization:

- version == current → normal load;
- version < current → run a migration function;
- version > current → return `Error::SchemaTooNew` (a structured error, never a panic).

This versioning is a data-integrity requirement, not a nice-to-have.

---

## 13. Type ownership

| Type group | Crate |
|-----------|-------|
| All domain types, newtypes, enums above | `hippmem-core` |
| `Clock` trait, RNG source trait | `hippmem-core` |
| Model traits (`Embedder`, `Extractor`, `Reranker`, `Summarizer`) | `hippmem-model` |
| `VectorIndex` / `Store` traits | `hippmem-store` |
| API container types (`WriteMemoryInput`, etc.) | `hippmem-engine` |

---

## Further reading

- [Algorithms](algorithms.md) — how the multi-channel recall, spreading activation, Hebbian consolidation, and decay use these types.
- [Model Backends](model-backends.md) — the `Embedder`, `Extractor`, `Reranker`, and `Summarizer` traits that populate the understanding and signatures.
- [Core Concepts](../concepts.md) — a plain-language introduction to the same model.
