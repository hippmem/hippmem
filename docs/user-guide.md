# HIPPMEM User Guide

> This guide walks you through HIPPMEM from zero to working knowledge. We recommend spending 5 minutes on the [Quick Start](quickstart.md) first, then coming back here.

---

## 1. Core Concepts at a Glance

Three points are enough (detailed explanations in [Core Concepts](concepts.md)):

1. **A memory is not plain text; it is a structured information node**: each memory automatically carries entities, topics, causal assertions, and association links to other memories.
2. **Associations are discovered at write time**: writing a new memory → the engine automatically searches existing memories for matches → scores them → builds links. Associations are created wherever you write.
3. **Retrieval uses spreading activation**: it is not "match keywords → rank", but "find seeds → spread along association links → merge → explain". It works like human recollection.

---

## 2. The Lifecycle of a Memory

```
You say something
    │
    ▼
┌─────────────┐
│  raw → indexed │  synchronous (completed when write() returns)
│              │  extract entities/topics/causality → build inverted + full-text index → discover associations → build links
├─────────────┤
│  enriched    │  asynchronous (background worker, seconds to minutes)
│              │  fill in goal/preference/emotion/decision inferences
├─────────────┤
│ consolidated │  asynchronous (scheduled, default every hour)
│              │  Hebbian reinforcement / decay / compaction / summary compression
└─────────────┘
```

For you, it is just calling `write()` and `retrieve()`. Everything in between is done automatically by the engine.

---

## 3. CLI Operations

### Write

```bash
hippmem write -c "Decided to switch from RocksDB to redb because redb compiles faster as a pure-Rust crate." -t Decision
```

Output:
```
✓ memory_id: 2152674446544667315913634290010169280 stage: Indexed links: 1
```

`content_type` allowed values: `UserStatement` | `Decision` | `Preference` | `Event` | `TaskState` | `ProjectKnowledge` | `Reflection` | `Correction`

### Retrieve

```bash
hippmem retrieve -q "Why move away from RocksDB?" -k 5
```

Output:
```
1. [0.782] Decided to switch from RocksDB to redb... (dims: [Causal, EntityOverlap])
2. [0.543] The user prefers Rust for systems programming... (dims: [EntityOverlap, SemanticSimilar])
```

### Explain

```bash
hippmem explain -m 2152674446544667315913634290010169280
```

Output:
```
memory: Decided to switch from RocksDB... importance: 0.800 links: 2 corrections: 0
```

### Inspect

```bash
hippmem inspect store-stats   # storage statistics
hippmem inspect queue         # queue status
```

### Consolidate

```bash
hippmem consolidate           # incremental consolidation
```

---

## 4. Core Rust Library API

```rust
use hippmem_engine::{Engine, EngineConfig};

let engine = Engine::open(EngineConfig::default())?;

// Write
let out = engine.write(WriteMemoryInput {
    content: "The team adopted Rust for the data pipeline after evaluating Go and Python.".into(),
    content_type: Some(ContentType::Decision),
    ..Default::default()
})?;

// Retrieve
let results = engine.retrieve(RetrieveInput {
    query: "What language is the data pipeline written in?".into(),
    top_k: 5,
    ..Default::default()
})?;

// Explain
let explanation = engine.explain(out.memory_id, None)?;

// Feedback
engine.feedback(FeedbackInput { /* ... */ })?;

// Consolidate
engine.consolidate(ConsolidationScope::Incremental)?;

// Inspect
engine.inspect(InspectQuery::StoreStats)?;

engine.close()?;
```

For full signatures see the [API Reference](api-reference.md); for real-world scenario code see the [Cookbook](cookbook.md).

---

## 5. Choosing an Embedder Backend

HIPPMEM supports three embedding backends, configured via `EmbedderConfig`:

### 5.1 Backend Types

| Provider | Vector Dimensions | Description | Use Case |
|----------|---------|------|---------|
| `deterministic` (default) | 256d SimHash | deterministic degraded mode, zero dependencies, no network, pure compute | CI, offline, privacy, testing |
| `openai-compatible` | depends on model | online API, high semantic precision | production, high-quality retrieval |
| `onnx` (reserved) | depends on model | offline local inference | future: privacy + high precision |

### 5.2 Configuration Methods

**Method 1: Environment Variables (recommended)**

```bash
# Embedder backend
export HIPPMEM__EMBEDDER__PROVIDER="openai-compatible"
export HIPPMEM__EMBEDDER__BASE_URL="https://api.openai.com/v1"
export HIPPMEM__EMBEDDER__MODEL="text-embedding-3-small"
export HIPPMEM__EMBEDDER__DIMENSIONS=1536

# API Key (independent of the Embedder config)
export OPENAI_API_KEY="sk-xxxxxxxx"
```

**Method 2: TOML Configuration File**

```toml
# hippmem.toml
[embedder]
provider = "openai-compatible"
base_url = "https://api.openai.com/v1"
model = "text-embedding-3-small"
api_key = "sk-xxxxxxxx"   # optional; if not provided, read from the OPENAI_API_KEY env var
dimensions = 1536
```

**Method 3: In Code**

```rust
use hippmem_core::config::EmbedderConfig;
use hippmem_engine::EngineConfig;

// Deterministic degraded mode (default, no configuration needed)
let config = EngineConfig::default();

// OpenAI API
let config = EngineConfig {
    embedder: EmbedderConfig::OpenAiCompatible {
        base_url: "https://api.openai.com/v1".into(),
        model: "text-embedding-3-small".into(),
        api_key: None,  // read from the OPENAI_API_KEY env var
        dimensions: 1536,
    },
    ..EngineConfig::default()
};
```

**Method 4: CLI Arguments**

```bash
hippmem --embedding-provider openai-compatible \
        --embedding-base-url "https://api.openai.com/v1" \
        --embedding-model "text-embedding-3-small" \
        write -c "A decision worth remembering."
```

### 5.3 Where to Configure the API Key

| Method | Configuration Path | Security | Recommendation |
|------|---------|--------|------|
| **Environment variable** | `OPENAI_API_KEY=sk-...` | ⭐⭐⭐ not written to files | **Recommended for production** |
| TOML file | `[embedder] api_key = "sk-..."` | ⭐⭐ file permission control | convenient for development |
| In code | `api_key: Some("sk-...".into())` | ⭐ hardcoding risk | testing only |

> **Priority**: CLI arguments > environment variables > TOML configuration file > code defaults.

### 5.4 Channel Weight Tuning

After switching to an online API backend, adjust the `SemanticDense` channel weight so the higher-quality embeddings are not overwhelmed by BM25 keyword matching.

**Environment variable method (recommended)**:

```bash
# API backend users should set this — let real semantic vectors take effect
export HIPPMEM__CHANNEL_COEFF_SEMANTIC_DENSE=1.5

# Optional: fine-tune the BM25 weight
export HIPPMEM__CHANNEL_COEFF_BM25=0.8
```

**TOML configuration file method**:

```toml
[algo]
channel_coeff_semantic_dense = 1.5
channel_coeff_bm25 = 0.8
```

**Code method**:

```rust
use hippmem_core::config::AlgoParams;

let config = EngineConfig {
    algo: AlgoParams {
        channel_coeff_semantic_dense: 1.5,
        channel_coeff_bm25: 0.8,
        ..AlgoParams::default()
    },
    embedder: EmbedderConfig::OpenAiCompatible { /* ... */ },
    ..EngineConfig::default()
};
```

> See the "Channel Calibration Parameters" section of `docs/configuration.md` for details.
> API Key lookup order: `EmbedderConfig.api_key` → environment variable `OPENAI_API_KEY`.
> If neither is present, `Engine::open()` returns `EngineError::Model("auth/missing key")`.

### 5.5 Build Features

Using the `openai-compatible` backend requires enabling a feature at build time:

```bash
cargo build --features api-backends
```

If the feature is not enabled, specifying `openai-compatible` returns `ModelError::Unavailable`.

> The default configuration (`deterministic`) requires no features and compiles with zero dependencies — fully usable in offline/CI environments.

---

## 6. Evaluation Framework

HIPPMEM ships with an evaluation system (`hippmem-eval`), including:

- **5 baseline comparisons**: BM25 Only / Embedding Only / Hybrid / RAG Summary / HIPPMEM Full
- **10 evaluation task types**: FactRecall / PreferenceRecall / ProjectContinuity / CausalTrace / ContradictionDetection / StateChange / ImplicitAssociation / NoiseResistance / LongTailRecall / ExplanationQuality
- **3 core metrics**: Recall@K / Precision@K / Explanation Accuracy

```bash
cargo test -p hippmem-eval thresholds_m6
```

---

## 7. Frequently Asked Questions

**Q: How is this different from a vector database?**

A vector database only does semantic similarity search. HIPPMEM adds: write-time association discovery, spreading-activation retrieval, Hebbian evolution, and explainable output. See [Comparison](comparison.md) for details.

**Q: Can it work without a network?**

Yes. The default deterministic backend uses rule-based extraction + SimHash semantics; all core features work offline.

**Q: Where is the data stored?**

By default in the `./hippmem_data/` directory: hippmem.redb (main storage) + fulltext/ (Tantivy BM25 index).

**Q: What scale does it support?**

The design target is 100K to 1M memories — for typical daily use this lasts years, and 1M is effectively a lifetime for an individual user. See [Capacity Planning](capacity-planning.md) for detailed estimates, hardware scaling, and worst-case analysis.

**Q: How do I contribute?**

Read [CONTRIBUTING.md](../CONTRIBUTING.md) for the development setup, commit conventions, and DCO requirements.
