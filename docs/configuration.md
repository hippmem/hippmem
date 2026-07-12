# Configuration Reference

> A complete listing of all fields, default values, ranges, and effects for the Engine and algorithm parameters.

---

## EngineConfig

Passed in when constructing `Engine::open()`.

| Field | Type | Default | Description |
|------|------|--------|------|
| `store_dir` | `PathBuf` | `"./hippmem_data"` | Storage directory. The redb file is created here, the Tantivy index under the `fulltext/` subdirectory |
| `algo` | `AlgoParams` | `AlgoParams::default()` | All algorithm parameters |
| `embedder` | `EmbedderConfig` | `EmbedderConfig::default()` | **Embedder backend configuration**. Defaults to deterministic 256d SimHash |
| `backend` | `BackendSelection` | `BackendSelection::default()` | Model backend selection (extractor/reranker/summarizer; embedder has moved to `EmbedderConfig`) |
| `background` | `BackgroundConfig` | `BackgroundConfig::default()` | Background worker configuration |

### EmbedderConfig

> Selects the embedding backend via the `provider` field, supporting three options. See [Embedder backend configuration](#embedder-backend-configuration) below.

### BackendSelection

| Value | Description |
|----|------|
| `Auto` | Automatic: use the API if an API key is present, otherwise fall back to the deterministic degraded backend |
| `Api` | Force the API backend |
| `Deterministic` | Force the deterministic degraded backend |

> **Note**: `BackendSelection` only controls extractor/reranker/summarizer. **The embedder is controlled independently by `EmbedderConfig`**.

### BackgroundConfig

| Field | Type | Default | Description |
|------|------|--------|------|
| `enrich_workers` | `usize` | `2` | Concurrency for strong-semantic enrich |
| `consolidate_workers` | `usize` | `1` | Concurrency for consolidation |
| `queue_capacity` | `usize` | `4096` | Background queue capacity (bounded; writes block when full) |
| `consolidate_interval_ms` | `u64` | `3_600_000` | Periodic consolidation trigger interval, default 1 hour |
| `enrich_enabled` | `bool` | `true` | Whether background enrich is enabled |

---

## Embedder Backend Configuration

> The embedder (text → dense vector) backend is controlled by `EmbedderConfig`, selected via the `provider` field in TOML.

### Three Providers

#### 1. Deterministic (default)

Deterministic 256d SimHash degraded backend. **Zero dependencies, no network, pure computation** — the same text always yields the same vector. CI and default configurations use this option.

```toml
[embedder]
provider = "deterministic"
dimensions = 256  # optional, default 256
```

#### 2. OpenAI-Compatible (online API)

Services compatible with the OpenAI Embeddings API format. Supported:

| Service | base_url | model | dimensions |
|------|----------|-------|------------|
| **OpenAI** | `https://api.openai.com/v1` | `text-embedding-3-small` | 1536 |
| **OpenAI** | `https://api.openai.com/v1` | `text-embedding-3-large` | 3072 |
| **Ollama** (local) | `http://localhost:11434/v1` | depends on loaded model | varies |
| **vLLM** (self-hosted) | `http://localhost:8000/v1` | per deployment | per model |

```toml
# OpenAI
[embedder]
provider = "openai-compatible"
model = "text-embedding-3-small"
dimensions = 1536
# base_url defaults to "https://api.openai.com/v1"; no need to specify explicitly

# Ollama (local)
[embedder]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "nomic-embed-text"
dimensions = 768
```

- `api_key` is optional: when not provided, it is read automatically from the `OPENAI_API_KEY` environment variable
- If neither is present -> `Engine::open()` returns `EngineError::Model(auth/missing key)`
- Requires the `api-backends` feature at compile time: `cargo build --features api-backends`

> **Online API backend**: when using an online embedding API, true semantic vectors (e.g., 1536d from OpenAI) provide richer representations than the deterministic 256d SimHash fallback.
> It is recommended to also raise the weight of the SemanticDense channel so the semantic advantage shows through:
> ```toml
> [embedder]
> provider = "openai-compatible"
> model = "text-embedding-3-small"
> dimensions = 1536
>
> [algo]
> channel_coeff_semantic_dense = 1.5
> ```
> The program also runs fine without this setting — raising the coefficient simply ensures the semantic channel is not drowned out by other channels (especially BM25 keyword matching).

#### 3. ONNX (planned)

Offline local inference via ONNX Runtime. Configuration schema is defined, implementation is on the roadmap:

```toml
[embedder]
provider = "onnx"
model_name = "bge-small-zh-v1.5"
model_cache_dir = "/home/user/.cache/hippmem/models"
dimensions = 512
```

Selecting `onnx` without a compiled runtime returns `ModelError::Unavailable`.

### Environment Variable Overrides

All `EmbedderConfig` fields can be overridden via environment variables (figment layering):

```bash
export HIPPMEM__EMBEDDER__PROVIDER="openai-compatible"
export HIPPMEM__EMBEDDER__BASE_URL="https://api.openai.com/v1"
export HIPPMEM__EMBEDDER__MODEL="text-embedding-3-small"
export HIPPMEM__EMBEDDER__DIMENSIONS=1536
export OPENAI_API_KEY="sk-xxxxxxxx"
```

> The `__` double underscore in environment variables separates nesting levels. For example, the `model` field under the `[embedder]` table maps to `HIPPMEM__EMBEDDER__MODEL`.

### Dimension Compatibility Warning

**Switching the embedder backend changes the vector dimension** (256 -> 1024 -> 1536 -> ...). Stored vector indexes are not compatible with a new embedder dimension.

| Switch direction | Impact | Handling |
|----------|------|------|
| Deterministic -> API | 256d -> 1024d+ | Need to rebuild the semantic index (`consolidate(Reindex)`) |
| API -> Deterministic | 1024d+ -> 256d | Same as above |
| OpenAI small → large | 1536d → 3072d | Same as above |

Index rebuild currently requires a manual step. Automatic migration is on the roadmap.

---

## AlgoParams

All algorithm parameters are centralized in one place (`hippmem_core::config::AlgoParams`) and can be overridden via figment layering: defaults < TOML file < environment variables.

### Multi-dimension Weights (affect association scoring)

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `w_entity` | `0.20` | [0, 1] | Shared-entity dimension weight |
| `w_topic` | `0.18` | [0, 1] | Shared-topic dimension weight |
| `w_semantic` | `0.18` | [0, 1] | Semantic-similarity dimension weight |
| `w_causal` | `0.10` | [0, 1] | Causal-relationship dimension weight |
| `w_temporal` | `0.08` | [0, 1] | Temporal-proximity dimension weight |
| `w_goal` | `0.10` | [0, 1] | Shared-goal dimension weight |
| `w_emotion` | `0.08` | [0, 1] | Shared-emotion dimension weight |
| `w_preference` | `0.08` | [0, 1] | Preference-alignment dimension weight |

### Edge-build Thresholds

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `strong_edge_threshold` | `0.55` | [0, 1] | Above this score -> strong edge |
| `weak_edge_threshold` | `0.25` | [0, 1] | Below this score -> observation zone |
| `edge_build_min_score` | `0.25` | [0, 1] | Below this score -> no edge created |

### Spreading Parameters

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `decay_factor` | `0.55` | [0, 1] | Per-hop energy decay coefficient. 0.55 means energy becomes 55% of the previous value after each hop |
| `max_hops_default` | `2` | [1, 5] | Default maximum spreading hops |
| `min_propagation_energy` | `0.05` | [0, 1] | Spreading stops below this energy |
| `fan_out_default` | `10` | [1, 50] | Maximum number of edges to expand per node |
| `max_seeds_per_channel` | `20` | [1, 100] | Maximum number of seeds per channel |

### Channel Calibration Parameters

> Controls the relative weight of each recall channel in initial energy computation. These coefficients prevent any single channel (e.g., BM25's unbounded scores) from dominating the result.

#### BM25 Normalization

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `bm25_norm_factor` | `2.0` | (0, infinity) | BM25 score normalization divisor: `tanh(bm25_raw / factor)` maps to [0,1] |

#### Per-channel Energy Coefficients

Each recall channel has an independent energy coefficient, all defaulting to `1.0`. Adjusting a coefficient changes the channel's relative influence in retrieval.

| Parameter | Default | Description |
|------|--------|------|
| `channel_coeff_bm25` | `1.0` | BM25 full-text retrieval channel |
| `channel_coeff_semantic_dense` | `1.0` | SemanticDense dense-vector channel |
| `channel_coeff_semantic_binary` | `1.0` | SemanticBinary binary-code channel |
| `channel_coeff_entity` | `1.0` | EntityInverted entity-inverted channel |
| `channel_coeff_topic` | `1.0` | TopicCluster topic channel |
| `channel_coeff_temporal` | `1.0` | Temporal temporal-proximity channel |
| `channel_coeff_goal` | `1.0` | Goal goal channel |
| `channel_coeff_event` | `1.0` | Event event channel |
| `channel_coeff_causal` | `1.0` | Causal causal channel |
| `channel_coeff_recent` | `1.0` | RecentActivation recent-activation channel |

> **Semantic channel coefficient formula**: `initial_energy = channel_coeff x query_match x 0.40 + ...`
> A coefficient of `0.0` fully disables a channel; a coefficient of `2.0` doubles the channel's energy.

### Hebbian Parameters

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `hebbian_learning_rate` | `0.08` | [0, 1] | Strengthening magnitude per co-activation |
| `hebbian_strength_cap` | `1.0` | [0, 1] | Edge strength cap |
| `coactivation_min_count` | `2` | [1, 100] | Minimum co-activation count to trigger Hebbian |
| `coactivation_window_secs` | `3600` | [60, 86400] | Co-activation time window |

### Decay Parameters

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `decay_per_cycle` | `0.97` | [0.5, 1.0] | Per-cycle strength multiplier. 0.97 means strength retains 97% each cycle |
| `min_retained_strength` | `0.10` | [0, 1] | Below this strength -> archived |
| `stale_observation_days` | `30` | [7, 365] | Memories in the observation zone that have not been activated for this many days -> evicted |

### Compaction Parameters

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `node_degree_limit` | `50` | [10, 500] | Maximum out-degree per node |
| `compaction_min_edge_strength` | `0.05` | [0, 1] | Edges below this strength are archived by compaction |
| `summary_similarity_threshold` | `0.7` | [0, 1] | Similarity above this -> triggers summary compression |

### Cold-start Parameters

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `cold_start_count` | `500` | [0, 10000] | The first N memories are in the cold-start period, with reduced per-dimension weights (to prevent overfitting) |

### Initial Strength

| Parameter | Default | Range | Description |
|------|--------|------|------|
| `initial_strength_default` | `0.50` | [0, 1] | Initial strength of newly created edges |
| `importance_default` | `0.50` | [0, 1] | Default importance |

---

## Configuration Examples

### TOML File

```toml
# hippmem.toml
[algo]
w_entity = 0.25
w_causal = 0.15
strong_edge_threshold = 0.60
decay_per_cycle = 0.95
cold_start_count = 100

[background]
consolidate_interval_ms = 1800000  # 30 minutes
```

```rust
use hippmem_engine::EngineConfig;
use figment::Figment;
// Load config: defaults < hippmem.toml < environment variables
let config: EngineConfig = Figment::new()
    .merge(figment::providers::Toml::file("hippmem.toml"))
    .extract()?;
```

### In Code

```rust
use hippmem_engine::{EngineConfig, BackgroundConfig};
use hippmem_core::config::AlgoParams;

let config = EngineConfig {
    algo: AlgoParams {
        w_entity: 0.25,
        w_causal: 0.15,
        ..AlgoParams::default()
    },
    background: BackgroundConfig {
        consolidate_interval_ms: 1_800_000, // 30 min
        ..BackgroundConfig::default()
    },
    ..EngineConfig::default()
};
```

---

## Tuning Recommendations

| Scenario | Recommendation |
|------|------|
| **Emphasize causal reasoning** | Raise `w_causal` to 0.15-0.20 |
| **Chinese-dominant content** | Default parameters already account for jieba tokenization; no adjustment needed |
| **High write rate (>100/min)** | Raise `consolidate_interval_ms` to 2h |
| **Low-latency retrieval (<50ms)** | `max_hops=1`, `fan_out=5`, `max_seeds_per_channel=10` |
| **Deep recall (>500ms acceptable)** | `max_hops=3`, `fan_out=20`, `consolidate_interval_ms=4h` |
| **Privacy / offline first** | `BackendSelection::DeterministicOnly` |
| **Benchmark comparisons** | Fix `AlgoParams::default()` and only change the backend selection |
| **Online API backend** | `channel_coeff_semantic_dense = 1.5` (recommended), optionally `channel_coeff_bm25 = 0.8` |
| **Semantic first (few keywords)** | `channel_coeff_semantic_dense = 2.0`, `channel_coeff_bm25 = 0.5` |
| **Keyword matching first** | `channel_coeff_bm25 = 1.5`, `bm25_norm_factor = 1.5` (steeper normalization curve) |
