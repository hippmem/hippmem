# Evaluation Framework

> This document describes HIPPMEM's evaluation methodology: corpus format, baseline comparison systems, metric tiers, automatic threshold gating, and determinism guarantees. It is the canonical reference for how retrieval quality is measured and gated.

Evaluation lives in the `hippmem-eval` crate, which depends on `hippmem-engine`. The versioned corpus resides in `crates/hippmem-eval/fixtures/`.

---

## 1. Why Ground Truth Is ID-Level

The premise of automatic evaluation is that results must be machine-checkable. Therefore the corpus ground truth is **structured**: a set of expected `MemoryId`s, expected matched dimensions, and expected warnings — not "is the answer text good." This lets `Recall@K`, `Precision@K`, explanation accuracy, and contradiction awareness be computed by direct ID comparison, with no human or LLM in the loop.

The evaluation is split into two tiers:

- **Structural metrics** — based on ID-level ground truth, fully deterministic, network-free, and used as CI gating.
- **Quality metrics** — rely on an LLM judge and real backends, run locally only, never used as task acceptance.

This split ensures automatic acceptance is never blocked by model availability or network state.

---

## 2. Corpus Format

Each evaluation case (`EvalCase`) is a JSON file under `fixtures/<suite>/`. A case declares an ordered write sequence, a query, and structured ground truth.

```jsonc
{
  "case_id": "fact-recall-001",
  "task_type": "FactRecall",
  "writes": [
    { "local_id": "m1", "content": "I'm using redb for storage",
      "content_type": "ProjectKnowledge", "context": { "session_id": 1 } },
    { "local_id": "m2", "content": "Because RocksDB is too heavy to compile, I switched away",
      "content_type": "Decision", "context": { "session_id": 1 } },
    { "local_id": "noise_*", "content": "...", "n": 200 }
  ],
  "query": { "text": "why not RocksDB for storage", "mode": "Balanced",
             "top_k": 10, "context": { "session_id": 2 } },
  "ground_truth": {
    "relevant": ["m2"],
    "also_acceptable": ["m1"],
    "expected_dimensions": ["Causal", "Entity"],
    "expected_warnings": [],
    "expected_edges": [
      { "from": "m2", "to": "m1", "link_type": "Causal" }
    ]
  }
}
```

Key points:

- `local_id` is unique within a case. The runner writes each entry, builds a `local_id → MemoryId` mapping, and evaluates against that mapping.
- Bulk noise (`noise_*` with `n`) is produced by a deterministic generator with a fixed seed, used for noise-resistance tasks.
- The corpus is **versioned** (`fixtures/v1/`). New hard cases are added to `v1` or a new `v2` is opened; old versions are retained to guard against overfitting regressions.

---

## 3. Baseline Comparison Systems

All baselines implement a single trait and run against the same corpus, ensuring comparability.

```rust
trait RetrievalSystem {
    async fn write(...);
    async fn query(...) -> Vec<MemoryId>;
}
```

| Baseline | Implementation | Notes |
|----------|----------------|-------|
| `Bm25Only` | `hippmem-store` fulltext index only | Pure keyword |
| `EmbeddingOnly` | Dense-vector KNN only | Pure semantic (fallback backend = hash vectors) |
| `HybridBm25Embedding` | BM25 + vector score fusion | Common hybrid |
| `RagSummaryMemory` | Summarize on write + vector recall | Standard RAG + summary |
| `HippmemFull` | Full `Engine` with spreading activation | The system under test |

The baselines are thin wrappers inside `hippmem-eval` that reuse `hippmem-store` and `hippmem-model`. CI runs all baselines with the deterministic fallback backend; relative comparisons (`HippmemFull` vs others) hold and are deterministic under the fallback.

---

## 4. Task Types

Ten task types exercise different retrieval capabilities. Each type emphasizes different ground-truth dimensions (e.g. `CausalTrace` stresses `expected_edges` containing `Causal`; `ContradictionDetection` stresses `expected_warnings`).

1. `FactRecall` — factual recall
2. `PreferenceRecall` — preference recall
3. `ProjectContinuity` — project continuity
4. `CausalTrace` — causal tracing
5. `ContradictionDetection` — contradiction detection
6. `StateChange` — state change
7. `ImplicitAssociation` — implicit association (no keyword overlap)
8. `NoiseResistance` — noise resistance
9. `LongTailRecall` — long-tail / long-span recall
10. `ExplanationQuality` — explanation quality

---

## 5. Metric Tiers

### 5.1 Structural metrics (deterministic, CI-gated)

For each case, the runner takes the system's returned `Vec<MemoryId>` (and, for `HippmemFull`, the retrieval trace) and computes:

- **Recall@K** = `|relevant ∩ topK| / |relevant|`
- **Precision@K** = `|(relevant ∪ also_acceptable) ∩ topK| / K`
- **Explanation Accuracy** (HippmemFull only) = fraction of `expected_edges` / `expected_dimensions` hit, compared against `activation_trace` and `matched_dimensions`
- **Contradiction Awareness** = fraction of `expected_warnings` actually produced
- **Long-tail Recall** = Recall@K computed only on `LongTailRecall` cases
- **Latency** = measured `diagnostics.latency_ms` (recorded, not hard-gated, as it is environment-dependent)

Aggregation: per `task_type` average plus a global average. Output is an `EvalReport` (JSON + table).

### 5.2 Quality metrics (LLM judge, local real-backend only, not CI-gated)

- **User Outcome** — a real LLM judges whether the answer with HIPPMEM is more helpful. Runs only with `--features api-backends` and keys present.
- **Explanation naturalness** — LLM scores explanation path readability.
- **Cost** — aggregate API token / call counts (from trace backend stats), reported locally.

These are never acceptance criteria; they serve as manual reference and tuning input.

---

## 6. Determinism and Reproducibility

- All write sequences and noise generation use a **fixed random seed** (injected `Rng`), so multiple runs produce identical results.
- Time comes from an injected `Clock`. Cases may declare a virtual timeline (each write carries a relative time), which drives freshness, temporal proximity, and long-tail testing.
- CI runs structural metrics with no network, the fallback backend, and fixed seeds, so metric values are **bit-for-bit reproducible** and can be asserted exactly.

---

## 7. Automatic Threshold Gating

Thresholds are encoded as executable assertions in `hippmem-eval`. Each threshold maps one-to-one to a row in the threshold table and is materialized as a `Threshold` literal in `builtin_thresholds(gate_set)` — the single executable mapping. Changing a threshold means changing the table and that function, never scattering special cases in code.

Representative thresholds (initial values, adjustable as the corpus evolves):

| ID | Condition |
|----|-----------|
| Global `Recall@10 ≥ 0.85` | Overall recall under the fallback backend, v1 corpus |
| `HippmemFull.Recall@10 > Bm25Only.Recall@10` and `> EmbeddingOnly.Recall@10` | Directional advantage proven |
| `ExplanationAccuracy ≥ 0.80` | Explanation paths must be accurate |
| `HippmemFull.Recall@10 > Hybrid.Recall@10` on `ImplicitAssociation` | Associative advantage proven |
| `ContradictionAwareness ≥ 0.80` | Contradiction detection quality |
| `NoiseResistance` Recall after consolidation ≥ before | Evolution must not regress |
| All ten task types have cases and run through; report generated | Coverage completeness |
| `HippmemFull` not worse than any baseline on ≥ 7/10 task types | Comprehensive advantage |

If a threshold becomes unreasonable due to corpus bias, the fix is to adjust the corpus or the threshold table and record the reason — never to special-case the code to bypass it.

---

## 8. Threshold Types

```rust
pub struct Threshold {
    pub id: String,
    pub metric: MetricKind,
    pub scope: MetricScope,
    pub op: CmpOp,
    pub bound: ThresholdBound,
    pub gate_set: GateSet,   // which release gate this threshold belongs to
    pub note: String,
}

pub enum MetricKind {
    RecallAtK { k: usize },
    PrecisionAtK { k: usize },
    ExplanationAccuracy,
    ContradictionAwareness,
    LongTailRecall,
}

pub enum MetricScope {
    Global,
    TaskType(TaskType),
    System(SystemKind),
}

pub enum CmpOp { Ge, Gt, Le, Lt }

pub enum ThresholdBound {
    Absolute(f32),
    RelativeTo { system: SystemKind, baseline: SystemKind },
}

pub struct Violation {
    pub threshold_id: String,
    pub expected: String,
    pub actual: f32,
    pub scope_desc: String,
}
```

`ThresholdBound::RelativeTo` enables system-versus-system comparisons (e.g. `HippmemFull.Recall@10 > Bm25Only.Recall@10`).

---

## 9. Run Entry Points and Output

```rust
pub async fn run_suite(
    suite_dir: &Path,
    systems: &[SystemKind],
    params: &AlgoParams,
) -> EngineResult<EvalReport>;

pub fn assert_thresholds(
    report: &EvalReport,
    thresholds: &[Threshold],
) -> Result<(), Vec<Violation>>;

pub fn builtin_thresholds(gate_set: GateSet) -> Vec<Threshold>;
```

- **CLI (local):** `cargo run -p hippmem-eval -- --suite fixtures/v1 --systems all`
- **CI:** a `#[test]` calls `run_suite` + `assert_thresholds(report, &builtin_thresholds(...))`, using only structural metrics and the current release gate's thresholds.

### `EvalReport`

```rust
pub struct EvalReport {
    pub suite: String,                  // e.g. "v1"
    pub params_digest: String,          // stable summary of AlgoParams (tuning replay)
    pub per_system: Vec<SystemReport>,
    pub generated_at: Timestamp,        // injected Clock (reproducible)
}

pub struct SystemReport {
    pub system: SystemKind,
    pub global: MetricSet,
    pub per_task_type: Vec<(TaskType, MetricSet)>,
    pub per_case: Vec<CaseResult>,      // per-case detail for diagnosis
}
```

`EvalReport` derives `Serialize` and can be exported as JSON. The `per_case` detail guarantees any threshold violation can be drilled down to the specific case's returned IDs and expected ground truth.

---

## 10. Methodology Summary

- **Ground truth is structured (ID-level), not natural language** — enabling automatic, deterministic evaluation.
- **Two-tier metrics** — structural metrics gate CI; quality metrics are local-only reference.
- **Same trait, same corpus** for all baselines — fair comparison.
- **Determinism via injected `Clock`/`Rng` and fixed seeds** — bit-for-bit reproducible runs.
- **Thresholds are executable literals**, mapped one-to-one from a single table — no scattered special cases.
- **Tuning loop supported** — `run_suite` accepts `AlgoParams`, so different parameter sets can be replayed and compared against the same corpus.
