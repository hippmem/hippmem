# Architecture Design

> This document describes HIPPMEM's system design: crate topology, responsibilities, data flow, and process model. It is the canonical reference for how the engine is assembled from its components.

HIPPMEM is a native associative memory engine written in Rust. It treats each memory as a structured node (`MemoryUnit`) connected to others by typed `association link`s, and answers queries through `spreading activation` over a `multi-channel recall` graph. Storage, retrieval, and consolidation are organized as a layered, single-direction dependency graph of crates.

---

## 1. Crate Topology

HIPPMEM is a cargo workspace of nine crates. Dependencies flow strictly downward with no cycles.

```text
                      hippmem-grpc        (optional, thin gRPC shell)
                           │
                           ▼
                      hippmem-engine       (public Rust API, orchestrator)
            ┌──────────────┼───────────────┬────────────────┐
            ▼              ▼               ▼                ▼
   hippmem-retrieval  hippmem-consolidation  hippmem-write   hippmem-eval
            │              │               │                (depends on engine)
            └──────┬───────┴───────┬───────┘
                   ▼               ▼
            hippmem-store     hippmem-model
                   │               │
                   └──────┬────────┘
                          ▼
                    hippmem-core         (types, traits, Clock, errors — no deps)
```

### Dependency rules (enforced)

- `hippmem-core` depends on no other HIPPMEM crate. It is the foundation: domain types, newtypes, `Clock`/`Rng` traits, `stable_hash64`, `UnitScore`, base errors, and the `config` module.
- `hippmem-store` and `hippmem-model` depend only on `core`.
- `hippmem-write`, `hippmem-retrieval`, and `hippmem-consolidation` depend on `core`/`store`/`model` and **do not depend on each other**. They are independent stages orchestrated by the engine.
- `hippmem-engine` depends on all of the above and is the public facade.
- `hippmem-eval` and `hippmem-grpc` depend on `engine` (and `core`).
- No reverse edges (e.g. `core` depending on `store`) are permitted; CI catches this naturally through cargo.

---

## 2. Crate Responsibilities

| Crate | Responsibility | Key contents |
|-------|----------------|--------------|
| `hippmem-core` | Domain types and infrastructure | All data-model types; `Clock`/`Rng` traits; `stable_hash64`; `UnitScore`; base error enum; `config` module (`AlgoParams`, `EngineConfig`) |
| `hippmem-store` | Storage and indexing | redb wrapper (`memory_log`, `memory_kv`, inverted indexes); Tantivy fulltext; `VectorIndex` trait + HNSW; binary-code recall; `association_graph`; `activation_log`; `consolidation_queue`; `Store` trait |
| `hippmem-model` | Model backends | Four traits (`Embedder`/`Extractor`/`Reranker`/`Summarizer`); API backends (reqwest); deterministic fallback; `ModelRegistry` |
| `hippmem-write` | Write-time association discovery | Staged write (raw → indexed → enriched); `AssociationKeys` generation; multi-dimensional candidate discovery; association scoring; edge creation |
| `hippmem-retrieval` | Spreading-activation retrieval | Multi-channel seed recall → RRF rank fusion → `initial_energy` assignment → spread traversal; merge / prune / cycle-break; rerank; warnings; explanation paths |
| `hippmem-consolidation` | Consolidation and evolution | Hebbian reinforcement; decay; compaction; summary merge; background worker; contradiction detection |
| `hippmem-engine` | Public API orchestration | The seven public `Engine` methods; holds `Store` + `ModelRegistry` + background runtime |
| `hippmem-eval` | Evaluation framework | Corpus loading; baseline comparison systems; metric computation |
| `hippmem-grpc` | gRPC service (optional) | proto + tonic service, thin wrapper over `Engine` |

> **Configuration.** There is no separate `hippmem-config` crate. Configuration lives in the `hippmem-core::config` module: `figment`-assembled `AlgoParams` (the algorithm parameter table), backend selection, storage paths, and background worker settings. `AlgoParams` defaults are compile-time constants that can be overridden by config.

---

## 3. Seven-Layer Logical View

The design whitepaper describes HIPPMEM as seven logical layers. They map onto crates and modules as follows.

| Logical layer | Implementation |
|---------------|----------------|
| Agent API | `hippmem-engine` (Rust API), `hippmem-grpc` (gRPC) |
| Context Ingestion | `hippmem-write::ingest` (normalizes input, assembles `WriteContext`) |
| Memory Understanding | `hippmem-write::understanding` calls `hippmem-model` (`Extractor`/`Embedder`) |
| Associative Write Engine | `hippmem-write::association` (scoring, edge creation) |
| Native Memory Store | `hippmem-store` (all indexes and the graph) |
| Spreading Activation Retrieval | `hippmem-retrieval` |
| Consolidation & Evolution | `hippmem-consolidation` |

---

## 4. Directory Layout

```text
hippmem/
├── Cargo.toml                      # [workspace], members
├── rust-toolchain.toml             # pinned stable toolchain
├── crates/
│   ├── hippmem-core/
│   │   └── src/{lib.rs, ids.rs, time.rs, score.rs, model_types/*, hash.rs,
│   │            error.rs, config.rs, rng.rs}
│   ├── hippmem-store/
│   │   └── src/{lib.rs, store.rs, memory_log.rs, kv.rs, fulltext.rs,
│   │            semantic/{mod.rs,vector_index.rs,hnsw.rs,binary.rs},
│   │            graph.rs, activation_log.rs, queue.rs}
│   ├── hippmem-model/
│   │   └── src/{lib.rs, traits.rs, registry.rs,
│   │            api/{openai.rs,anthropic.rs,http.rs},
│   │            deterministic/{embed.rs,extract.rs,rerank.rs,summarize.rs}, error.rs}
│   ├── hippmem-write/
│   │   └── src/{lib.rs, ingest.rs, understanding.rs, keys.rs,
│   │            candidates.rs, scoring.rs, edges.rs, staged.rs}
│   ├── hippmem-retrieval/
│   │   └── src/{lib.rs, seeds.rs, energy.rs, spreading.rs,
│   │            rerank.rs, warnings.rs, explain.rs}
│   ├── hippmem-consolidation/
│   │   └── src/{lib.rs, hebbian.rs, decay.rs, compaction.rs,
│   │            summarize.rs, worker.rs}
│   ├── hippmem-engine/
│   │   ├── src/{lib.rs, api.rs, write_api.rs, retrieve_api.rs,
│   │   │        explain_api.rs, consolidate_api.rs, inspect_api.rs,
│   │   │        feedback_api.rs, runtime.rs}
│   │   ├── examples/basic_usage.rs
│   │   └── src/bin/{hippmem-cli.rs, hippmem-server.rs}
│   ├── hippmem-eval/
│   │   ├── src/{lib.rs, corpus.rs, baselines.rs, metrics.rs, runner.rs}
│   │   └── fixtures/               # versioned evaluation corpus
│   └── hippmem-grpc/
│       ├── build.rs
│       ├── proto/hippmem.proto
│       └── src/{lib.rs, service.rs}
└── docs/                           # user-facing documentation
```

---

## 5. Physical Storage Layout

All persistent state lives under a configurable store directory. A single redb file holds the append-only log, mutable key/value tables, inverted indexes, and overlays; Tantivy and HNSW indexes live in sibling directories.

```text
<store_dir>/
├── hippmem.redb                # redb main database, tables:
│   ├── memory_log              # append-only: MemoryId -> RawRecord  (immutable)
│   ├── memory_kv               # MemoryId -> MemoryUnit (bincode)
│   ├── entity_index            # EntityKey -> Vec<MemoryId>
│   ├── topic_index             # TopicKey  -> Vec<MemoryId>
│   ├── goal_index              # GoalKey   -> Vec<MemoryId>
│   ├── event_index             # EventKey  -> Vec<MemoryId>
│   ├── temporal_index          # TemporalKey -> Vec<MemoryId>
│   ├── link_overlay            # MemoryId -> OutLinks / InLinks (mutable: strength/activation/decay)
│   ├── summary_overlay         # MemoryId -> summary relationships
│   ├── correction_overlay      # MemoryId -> correction/conflict/deprecation relationships
│   ├── activation_log          # append: retrieval traces and co-activation events
│   └── consolidation_queue     # pending background tasks
├── fulltext/                   # Tantivy index directory
└── semantic/                   # HNSW index + binary code tables
```

Key invariants:

- **Append-only `memory_log`.** Raw records are inserted once and never modified or deleted. All mutable state lives in the `*_overlay` tables.
- **Crash recovery.** redb transactions guarantee durability; Tantivy/HNSW indexes can be rebuilt from `memory_log` via `consolidate(Reindex)`.
- **Association graph.** The graph used by spreading activation is an in-memory projection of `link_overlay` (both out-edges and in-edges), loaded on demand and backed by a `dashmap` hot-tier cache.

---

## 6. Feature Flags

Feature flags are deliberately minimal.

| Feature | Crate | Effect | Default |
|---------|-------|--------|---------|
| `api-backends` | `hippmem-model` | Compile OpenAI/Anthropic clients | **off** (CI and defaults build with the deterministic fallback only) |
| `grpc` | workspace | Compile `hippmem-grpc` | off |

Runtime backend selection (Api / Deterministic / Auto) is **configuration**, orthogonal to features. With `api-backends` off, only the deterministic fallback is available, so CI builds and tests require no network and no API keys.

---

## 7. Data Flow

HIPPMEM has two main paths: the **write path** (ingest → store → schedule enrichment) and the **retrieval path** (query → multi-channel recall → spread → rerank → return).

### 7.1 Write Path

```text
Engine::write(WriteMemoryInput)
   │
   1. Generate MemoryId (ULID)
   2. extractor.extract_immediate(content)  →  ImmediateExtraction
   3. Build MemoryUnderstanding (entities / topics / explicit causals)
   4. write::keys::generate_keys()          →  AssociationKeys
   5. embedder.embed() (optional; deferred if no API backend)
   6. Build SemanticSignature (simhash / binary_code / topic_minhash, computed locally)
   7. Recall candidate old memories from store indexes:
        a. entity_index  by entity_keys
        b. temporal_index by temporal_keys
        c. topic_index   by topic_keys
        d. Tantivy BM25  search content.raw, top-N
        e. HNSW          search by semantic_signature, top-N
        f. merge + dedup → batch-read MemoryUnits
   8. write::staged::raw_to_indexed(candidates)  →  StagedWriteOutput
   9. Persist:
        a. memory_log.append(memory_id, bincode(unit))
        b. memory_kv.put(memory_id, bincode(unit))
        c. update inverted indexes (entity/topic/goal/event/temporal)
        d. fulltext.add_document(memory_id, content.raw)
        e. link_overlay.put(memory_id, bincode(out_links))
        f. if embedding ready → HNSW.add(memory_id, vector)
  10. background_tx.send(Enrich { memory_id, importance })
  11. Return WriteMemoryOutput { memory_id, stage=Indexed, created_links,
                                 understanding, warnings }
```

The synchronous portion of `write` completes through the `Indexed` stage (basic dimensions + initial edges + index writes) before returning. Strong semantic dimensions (goals, preferences, emotions, decisions, implicit causals, contradictions) and any deferred dense vector are computed by background enrich workers.

If a model call fails, `write` does **not** fail: the engine falls back to the deterministic backend or leaves the field empty and adds a `WriteWarning`. The raw content is always persisted first (append-only log + transaction), so a crash never loses the original memory.

### 7.2 Retrieval Path

```text
Engine::retrieve(RetrieveInput)
   │
   1. extractor.extract_immediate(query + context)
   2. Multi-channel seed recall (retrieval::seeds::multi_channel_seeds):
        a. BM25 channel        → fulltext_index.search(query, top_k)
        b. Entity inverted     → entity_index by entity_keys
        c. Temporal proximity  → temporal_index + context.recent_memory_ids
        d. Semantic channel    → embedder.embed(query) → HNSW search
        e. each channel yields (MemoryId, score); merge + dedup
   3. retrieval::energy::assign_initial_energy(seeds, query, context, params)
   4. retrieval::spreading::spread(seeds, store, max_hops, fan_out,
                                   decay_factor, type_modifiers)
        → Vec<(MemoryId, f32, ActivationTrace)>
   5. models.reranker.rerank(query, candidates)
   6. retrieval::warnings::check_warnings(results, store)
        → contradictions / staleness / supersession
   7. retrieval::explain::build_explanation(activation_trace)
   8. Build RetrieveOutput { results, trace, diagnostics }
   9. background_tx.send(HebbianCandidate { retrieval_id, used_ids })
```

The entire retrieval path is synchronous; only the post-retrieval Hebbian candidate recording is sent to the background. `RetrievalDiagnostics` records per-channel contribution (`seed_count`, `max_score`, `contributed_results`), the backend actually used, and latency — enabling transparent introspection of how results were assembled.

---

## 8. Process Model and Runtime Topology

HIPPMEM runs as a single process — no distribution, no clustering. The `Engine` owns a tokio multi-thread runtime, the shared store, the model registry, swappable algorithm parameters, and a bounded background task queue.

```text
Engine holds:
  Store          (Arc<dyn Store + Send + Sync>, thread-safe via redb Arc<Database>)
  ModelRegistry  (Arc, four Arc<dyn Trait> handles)
  AlgoParams     (Arc<RwLock<...>>, hot-swappable for tuning)
  tokio runtime:
    - Foreground:  write (sync to Indexed) / retrieve / explain / inspect
    - Background workers (consume consolidation_queue):
        * enrich worker(s):     complete strong semantic dimensions
                                (raw → enriched), may add causal/contradiction edges
        * consolidate worker:   Hebbian / decay / compaction / summary
                                (periodic + triggered)
    - Concurrency governance:
        * bounded background queue (default 4096)
        * in-flight dedup (dashmap)
        * importance-based priority
        * low-value deferral under pressure
        * strong-semantic tasks MUST NOT be dropped
```

### Foreground / background boundary

| API | Synchronous (before return) | Background |
|-----|------------------------------|------------|
| `write` | raw persistence + Indexed (basic dims + initial edges + indexes) | enrich (strong dims), deferred dense vector, consolidation |
| `retrieve` | entire path (seeds → spread → rerank → warnings) | post-retrieval Hebbian candidate |
| `explain` | entire path (read store) | — |
| `consolidate` | dispatch / trigger; returns acknowledgement or waits (scope-dependent) | bulk consolidation execution |
| `feedback` | record usage signal | Hebbian reinforcement applied on next cycle |
| `inspect` | entire path (read store + queue state) | — |

### Backpressure

The background queue is bounded. When full, the policy is either `BlockProducer` (briefly back-pressure the writer, the default) or `DropLowValue` (drop low-importance non-strong-semantic tasks). Strong-semantic enrich tasks are never dropped — they may only be delayed.

### Shutdown

`Engine::close` drops the queue sender, which causes workers to flush their current task and exit; their join handles are awaited (5s timeout, then abort); finally `store.close()` flushes and closes redb and Tantivy.

---

## 9. Binary Products

Two binaries live in `hippmem-engine`:

```text
hippmem
  ├── src/bin/hippmem-cli.rs     # CLI: opens a local store, runs one command, exits
  └── src/bin/hippmem-server.rs  # gRPC server: long-lived, serves remote clients
```

### CLI mode

```text
hippmem write -c "..." -t Decision
hippmem retrieve -q "..." -k 5
hippmem inspect store-stats
```

Each invocation opens the store, runs a single command, prints output (human-readable text or JSON), and exits. CLI output maps Engine API types to `serde_json` or formatted text.

### Server mode

```text
hippmem serve    # or: HIPPMEM_LISTEN=0.0.0.0:50051 hippmem serve
```

The process opens the store, starts a tonic gRPC server, and serves concurrent clients. Each gRPC request maps to an `Engine` method; `Arc<Engine>` is shared across requests, so concurrency safety is the engine's responsibility.

### Locking and safety

- A store directory cannot be opened by two `Engine` processes simultaneously — redb's file lock enforces this. A second open fails with `StoreError::Locked`.
- The gRPC server allows concurrent reads and writes from multiple clients; tokio's multi-thread runtime serializes redb transactions internally.

---

## 10. Configuration Loading

Configuration is assembled with `figment` in a strict precedence order:

```text
built-in defaults  <  TOML file  <  HIPPMEM_* environment variables  <  CLI flags
```

Supported environment variables:

| Variable | Field | Default |
|----------|-------|---------|
| `HIPPMEM_STORE_DIR` | `store_dir` | `./hippmem_data` |
| `HIPPMEM_CONFIG` | config file path | `hippmem.toml` |
| `HIPPMEM_LISTEN` | gRPC listen address | `0.0.0.0:50051` |
| `HIPPMEM_LOG_LEVEL` | log level | `info` |
| `HIPPMEM_ENRICH_ENABLED` | `background.enrich_enabled` | `true` |

---

## 11. Error Propagation

Each library crate defines its own `Error` enum (`thiserror`) with a `Result<T, Error>` alias. `hippmem-engine` aggregates lower-layer errors into the public `EngineError` and maps them to error codes. Lower-layer library types (`redb::*`, `tantivy::*`, `reqwest::*`) never appear in public signatures — they are converted at each crate boundary.

Error conversion rules:

- **Store errors** always propagate as `Err(EngineError::Store(...))` — data integrity takes priority.
- **Non-fatal model errors** (extractor timeout, embedding failure) do **not** become `Err`. They are recorded as `WriteWarning` and the call returns `Ok`, because the raw content is already persisted.
- **Background worker errors** are logged via `tracing` and may be retried; they never panic the engine.

---

## 12. Observability

Every stage emits `tracing` spans. The span hierarchy mirrors the retrieval pipeline (`seed → spread → merge → rerank`), which is exactly what `explain` and `inspect` surface to callers — they read structured records rather than ad-hoc state.

Key span names:

- `engine.write` — `memory_id`, `latency_ms`, `stage_reached`
- `engine.retrieve` — `query` (truncated), `top_k`, `hops_used`, `latency_ms`, `reranked`
- `engine.consolidate` — `scope`, `memories_processed`, `elapsed_ms`
- `enrich.{worker_id}`, `consolidate.{worker_id}` — background worker activity

In production, logs go to stdout (optionally JSON) and are collected by an external agent (journald, promtail, etc.).

---

## 13. Model Backends and Degradation

HIPPMEM defines four model capabilities as traits in `hippmem-model`: `Embedder`, `Extractor`, `Reranker`, and `Summarizer`. Each has two implementations:

- **API backends** — thin `reqwest` clients for OpenAI (embedding) and Anthropic (extraction/summarization). Compiled in only with the `api-backends` feature; require API keys from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- **Deterministic fallback** — local, dependency-free implementations that produce stable, reproducible output. Hash-based embeddings, rule-based extraction, identity reranker, extractive summarizer.

Backend selection is per-capability and configuration-driven (`Api` / `Deterministic` / `Auto`):

- `Auto` (default) — use the API backend when a key is present and the feature is compiled in; otherwise fall back to deterministic.
- `Api` — require the API backend; fail with `BackendUnavailable` if the key is missing.
- `Deterministic` — always use the fallback, regardless of keys.

This design has two consequences central to the architecture. First, CI builds and runs with no network and no keys: the fallback backend makes the entire engine — including evaluation — deterministic and reproducible. Second, model failures are non-fatal: when an API call fails at runtime, the engine records a `WriteWarning` and continues with the fallback or a deferred field, never losing the raw memory. This is what makes the write path robust under partial model outages.

---

## 14. Architectural Properties

- **Acyclic dependencies.** The crate graph has no cycles; cargo naturally enforces this.
- **No lower-layer types leak.** Public API signatures contain only HIPPMEM-owned types.
- **Indexes are rebuildable.** `consolidate(Reindex)` rebuilds Tantivy + HNSW + inverted indexes from `memory_log`; retrieval results are identical before and after.
- **Crash-safe.** Append-only `memory_log` + redb transactions mean the original memory is never lost on crash; indexes may lag but can be rebuilt.
- **Deterministic by default.** With the fallback backend, fixed seeds, and an injected `Clock`, all writes, noise generation, and retrieval are bit-for-bit reproducible — which is what makes CI gateable on exact metric thresholds.
