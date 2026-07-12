# Model Backends

> This document describes HIPPMEM's model-backend trait system: the four capability traits (`Embedder`, `Extractor`, `Reranker`, `Summarizer`), the registry that assembles them, the deterministic fallback that runs without network access, and the API backends that enhance quality when keys are available.

HIPPMEM is designed to run end-to-end with **no network and no API keys**. Every model capability has a deterministic fallback implementation that produces real, non-trivial results — not stubs. Remote model APIs (OpenAI, Anthropic, ONNX) are optional enhancements layered on top. This separation is what lets the engine develop, test, and ship in CI without secrets, while still benefiting from real models in production.

---

## 1. Design principles

1. **Traits live at the core semantic layer; implementations are pluggable.** Trait signatures use only `hippmem-core` domain types and primitive `String`/vector inputs. No vendor-specific type ever appears in a trait signature.
2. **Async contract, sync fallback.** Trait methods are `async` because model calls are I/O-bound. The deterministic implementations are pure computation and resolve immediately.
3. **Degradation is a first-class requirement.** Every trait has a `Deterministic*` implementation that is a real, reproducible algorithm — hash embeddings do approximate recall, rule-based extraction produces real structures, BM25 ranks documents. The end-to-end pipeline produces non-trivial output with no model.
4. **Backend selection is configuration-driven.** At runtime, configuration chooses the backend per capability. A missing key silently falls back to deterministic and emits one `tracing::warn`.
5. **Output carries provenance.** Any model-generated understanding is tagged with `GeneratedBy::Extractor { backend }`, so the origin of every extraction is traceable. Low-confidence extractions are routed to the observation zone rather than polluting the fact layer.

All traits and implementations live in the `hippmem-model` crate.

---

## 2. The four traits

```rust
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> ModelResult<Vec<Vec<f32>>>;
    fn backend_id(&self) -> &str;
}

#[async_trait::async_trait]
pub trait Extractor: Send + Sync {
    async fn extract_immediate(&self, content: &MemoryContent) -> ModelResult<ImmediateExtraction>;
    async fn extract_strong(&self, content: &MemoryContent) -> ModelResult<StrongExtraction>;
    fn backend_id(&self) -> &str;
}

#[async_trait::async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, candidates: &[String]) -> ModelResult<Vec<f32>>;
    fn backend_id(&self) -> &str;
}

#[async_trait::async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, sources: &[SummarizeInput]) -> ModelResult<SummaryOutput>;
    fn backend_id(&self) -> &str;
}
```

The `Extractor` trait is split into two stages that mirror the memory's stage progression:

- `extract_immediate` — basic immediate dimensions (entities, topics, explicit causal claims, language, content type, importance). Must be available for every backend; the deterministic backend uses rules.
- `extract_strong` — strong semantic dimensions (goals, preferences, emotions, decisions, implicit causals, contradiction hints). The deterministic backend produces low-confidence rule-based output; the API backend uses a language model.

Auxiliary types:

```rust
pub struct ImmediateExtraction {
    pub entities: Vec<EntityMention>,
    pub topics: Vec<TopicTag>,
    pub explicit_causals: Vec<CausalClaim>,   // kind = Explicit
    pub language: Language,
    pub content_type: Option<ContentType>,
    pub importance: UnitScore,
}

pub struct StrongExtraction {
    pub goals: Vec<GoalFrame>,
    pub preferences: Vec<PreferenceFrame>,
    pub emotions: Vec<EmotionFrame>,
    pub decisions: Vec<DecisionFrame>,
    pub implicit_causals: Vec<CausalClaim>,   // kind = Implicit
    pub contradictions: Vec<ContradictionHint>,
    pub confidence: UnitScore,                // deterministic backend returns a low value
}
```

The engine holds `Arc<dyn Embedder>`, `Arc<dyn Extractor>`, `Arc<dyn Reranker>`, and `Arc<dyn Summarizer>`, assembled by the `ModelRegistry`. Upper-layer algorithms depend only on the traits.

---

## 3. API backends (enhancement)

API backends are thin `reqwest` clients. Each implements the corresponding trait. Keys are read from environment variables; a missing key makes that backend unavailable, and the registry falls back to deterministic.

| Trait | Default API backend | Endpoint / model | Environment variable |
|-------|---------------------|------------------|----------------------|
| `Embedder` | OpenAI-compatible (OpenAI / Ollama / vLLM) | `text-embedding-3-small` (OpenAI, 1536d) | `OPENAI_API_KEY` |
| `Extractor` | Anthropic Claude | Messages API, structured JSON output | `ANTHROPIC_API_KEY` |
| `Reranker` | (identity — deterministic fallback only) | passes candidates through without reordering | — |
| `Summarizer` | Anthropic Claude | Messages API | `ANTHROPIC_API_KEY` |

Key points:

- **Structured extraction.** The `Extractor` API implementation asks Claude for JSON output and parses it with `serde` into `ImmediateExtraction` / `StrongExtraction`. A parse failure returns `ModelError::Parse` and records evidence — it never panics and never pollutes the fact layer.
- **Timeout and retry.** Every call has a configurable timeout (default 30s) and bounded retries with exponential backoff. Exhausted retries return an error; the caller decides whether to fall back to deterministic or queue a background retry.
- **Multilingual.** Embedder and Reranker select model variants that support Chinese.
- **Provenance.** `backend_id()` returns the concrete model name (e.g. `"openai-text-embedding-3-small"`) and is written into `GeneratedBy::Extractor { backend }`.
- **Cost and observability.** Every API call is wrapped in a `tracing` span that records token usage and latency for the evaluation cost metrics.

API backends are **not** exercised in CI (no keys). Their correctness is verified by local real-backend runs and by contract tests with mock HTTP servers that validate client parsing logic.

> The Embedder backend has evolved beyond a single OpenAI client. The current `EmbedderConfig` supports three variants: `Deterministic` (256d hash), `OpenAiCompatible` (OpenAI / Ollama / vLLM, configurable model and dimension), and `Onnx` (local ONNX runtime). All variants implement the same `Embedder` trait.

---

## 4. Deterministic fallback backends

These are not stubs. They are deterministic, reproducible algorithms that produce non-trivial output, ensuring the pipeline closes end-to-end with no model.

### DeterministicEmbedder

- `dim()` returns the configured dimension (default **256**, differs from the API backend's 1024/1536 — the system builds its vector index to match the active embedder's `dim()`, so switching backends requires a `consolidate(Reindex)` rebuild).
- `embed()` performs **feature-hash embedding**: tokenize the text, project each token through several independent hash functions into the `dim`-dimensional space, accumulate, and L2-normalize. The same text always yields the same vector; texts sharing tokens have closer vectors, enabling real approximate recall.
- Pure computation, no network, no randomness.

### DeterministicExtractor

`extract_immediate()`:

- **Entities** — dictionary + regex (capitalized proper nouns, `@mentions`, file/library name patterns, code identifiers) → `EntityMention` with medium confidence.
- **Topics** — high-frequency keywords → `TopicTag`.
- **Explicit causals** — connective rules (Chinese causal conjunctions and English "because/therefore/so/thus/hence/leads to") split cause/effect → `CausalClaim { kind: Explicit }`.
- **Importance** — heuristic score from length, type, and decision-word presence.

`extract_strong()` — low-confidence rule-based extraction:

- Preferences: Chinese preference verbs and English "prefer/avoid/like/dislike" → `PreferenceFrame`.
- Emotions: emotion lexicon → `EmotionFrame`.
- Goals: Chinese goal/intent verbs and English "goal/plan to/aim to" → `GoalFrame`.
- Decisions: Chinese decision verbs and English "decide/choose/adopt/drop" → `DecisionFrame`.
- Implicit causals / contradictions: may return empty `Vec`s when confidence is insufficient.
- Overall `confidence` is notably lower than the API backend (≤ 0.5), which routes these extractions to the observation zone by default.

Fully deterministic — fixed dictionaries, no randomness.

### DeterministicReranker

`rerank()` scores query-candidate term overlap with a BM25-style weighting (reusing the full-text scorer or a lightweight standalone implementation). Deterministic, network-free — effectively "BM25 as reranker."

### DeterministicSummarizer

`summarize()` is **extractive**: sentences are scored by keyword coverage and position, the top sentences are concatenated, `covers` is set to all input IDs, and `confidence` is low. No generation, no hallucination risk, fully deterministic.

### Dictionaries

The rule-based extractor depends on small static dictionaries shipped in the repository at `crates/hippmem-model/resources/`. They are compiled into the binary with `include_str!` (no runtime file IO) so CI stays offline:

| File | Contents |
|------|----------|
| `causal_connectives.txt` | Causal connectives (Chinese and English) |
| `emotion_lexicon.txt` | Emotion word → `EmotionKind` mapping (`word\tcategory`) |
| `preference_markers.txt` | Preference markers with polarity |
| `goal_markers.txt` | Goal markers |
| `decision_markers.txt` | Decision markers |
| `stopwords_zh.txt` / `stopwords_en.txt` | Stopwords for topic keywords and BM25 |

Each file is tens to hundreds of lines — enough to produce non-trivial structure, not full coverage. Adding entries requires only editing the resource files, no code change. Dictionary content is not an algorithm parameter; the file list and format are fixed by this contract.

---

## 5. Backend selection and the registry

```rust
pub struct ModelRegistry {
    pub embedder: Arc<dyn Embedder>,
    pub extractor: Arc<dyn Extractor>,
    pub reranker: Arc<dyn Reranker>,
    pub summarizer: Arc<dyn Summarizer>,
}

pub enum BackendChoice { Api, Deterministic, Auto }

pub struct BackendSelection {
    pub extractor: BackendChoice,
    pub reranker: BackendChoice,
    pub summarizer: BackendChoice,
}
// Default: all Auto

pub struct BackendUsage {
    pub embedder: String,            // backend_id(), e.g. "deterministic-hash"
    pub reranker: Option<String>,    // None when rerank is not enabled
}
```

Assembly is configuration-driven:

- Each non-embedder capability can independently select `Api`, `Deterministic`, or `Auto`.
- `Auto` (the default) checks for the relevant environment variable: if the key is present, it uses the API backend; otherwise it falls back to deterministic and logs one `warn`.
- The Embedder has its own `EmbedderConfig` with the three variants described above.
- **CI environment** sets no keys → everything falls back to deterministic → `cargo test` runs offline.
- **Local real evaluation** sets keys and configures `Api` → real quality metrics.

`BackendUsage` records which backend was actually used by a retrieval and is written into `RetrievalDiagnostics.backend_used`, so every result is traceable to its backend.

---

## 6. Errors

```rust
#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("network/timeout: {0}")] Network(String),
    #[error("auth/missing key for backend {0}")] Auth(String),
    #[error("rate limited")] RateLimited,
    #[error("parse model output: {0}")] Parse(String),
    #[error("backend unavailable: {0}")] Unavailable(String),
}
pub type ModelResult<T> = Result<T, ModelError>;
```

The library **never** panics on a model error. Callers (the write and retrieve paths) decide how to handle it: fall back to deterministic, queue a background retry, or leave the dimension empty and record a warning.

---

## 7. Contract tests

Both backend families share one set of contract tests, located in `hippmem-model/tests/`. The deterministic backend runs them in CI; the API backends run them against mock HTTP servers that return fixed JSON.

- **Embedder** — `embed` output length equals input length; each vector has length `dim()`; deterministic (same input → same output, for the fallback); L2-normalized where the contract requires.
- **Extractor** — `extract_immediate` on a text containing connectives produces at least one `Explicit` causal; entity extraction hits known samples; output structure is valid (spans in bounds, confidence in `[0, 1]`).
- **Reranker** — output length equals candidate count; scores are finite (no NaN).
- **Summarizer** — `covers` equals the input ID set; summary is non-empty.
- **Provenance** — every backend's `backend_id()` is stable and non-empty.

A pipeline integration test verifies the end-to-end path: write a text containing a Chinese causal-conjunction pattern → deterministic extraction → generate `causal_key` → build a `Causal` edge, all offline. This test is a key acceptance gate — it proves the pipeline is correct (not merely that the model is smart).

---

## 8. Design rationale

- The core read/write path (write through `indexed`, basic retrieval) is **fully usable** under the deterministic backend, with no remote dependency.
- Remote model APIs are an **enhancement path** for understanding, recall, and reranking quality. They may be absent.
- "Use vendor APIs" and "do not depend on remote SaaS" are therefore not in conflict: the design is **default enhancement plus a mandatory deterministic floor**.

This is what makes HIPPMEM reliable in restricted environments (air-gapped, CI, on-prem) while still benefiting from frontier models when they are available.

---

## Further reading

- [Data Model](data-model.md) — the types the traits produce and consume.
- [Algorithms](algorithms.md) — how `Embedder`, `Extractor`, `Reranker`, and `Summarizer` feed the retrieval and consolidation pipeline.
- [Configuration Reference](../configuration.md) — `EmbedderConfig` and `BackendSelection` fields with defaults.
