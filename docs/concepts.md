# Core Concepts

> This document explains HIPPMEM's core concepts in plain language. After reading it, you'll understand how the engine "thinks" without looking at the code.
> Want to jump straight to code? Head to the [Cookbook](cookbook.md).

---

## 1. MemoryUnit: a memory is not a document, it's a structured information node

Traditional RAG treats memories as documents — store a piece of text, retrieve it via embedding. HIPPMEM treats each memory as a **structured node** (MemoryUnit):

```
┌─────────── MemoryUnit ───────────┐
│  content: "Switched to redb      │
│            because RocksDB was    │
│            slow to compile"       │
│  content_type: Decision           │
│  created_at: 2026-06-05T10:30:00  │
│                                   │
│  understanding:                   │
│    entities: [RocksDB, redb]     │
│    topics: [storage, compilation] │
│    causal_claims: [               │
│      {cause: "slow compile",      │
│       effect: "switched to redb"}]│
│    ]                              │
│                                   │
│  out_edges:                       │
│    → memory_id_42 (EntityOverlap) │
│    → memory_id_78 (Causal)        │
│                                   │
│  activation_state:                │
│    strength: 0.73                 │
│    last_activated: 2026-06-05     │
│    activation_count: 5            │
└──────────────────────────────────┘
```

A MemoryUnit has three layers:

| Layer | Content | Example |
|-------|---------|---------|
| **Content layer** | Raw text + type + time | "Because RocksDB is too slow..." / Decision |
| **Understanding layer** | Structured info auto-extracted by the engine | entities, topics, causal claims, emotions |
| **Activation layer** | How many times this memory has been used, how important it is | strength=0.73, activated 5 times |

---

## 2. Associations: the "edges" between memories

When two memories share entities, topics, or have causal/temporal relationships, the engine automatically builds **association links** (AssociationLink).

### Association types (partial)

```
      Causal              EntityOverlap        SemanticSimilar
    A ──→ B              A ──── B             A ····· B
   (A causes B)         (both mention Rust)   (similar meaning)

    Temporal             Contradiction         Correction
    A ═══ B              A ←→ B               A ←─ B
   (close in time)      (contradict each other) (B corrects A)
```

### Strong edges vs weak edges vs observation zone

- **Strong edge** (strength ≥ 0.55): high-confidence association hit on multiple dimensions. Directly participates in spreading activation.
- **Weak edge** (0.25 ≤ strength < 0.55): single-dimension hit or insufficient confidence. Participates in spreading but decays faster.
- **Observation zone** (strength < 0.25): the relationship is very weak; does not participate in spreading for now. If later co-activation strengthens it, it gets promoted; otherwise it gets pruned.

Edge strength is not fixed — every co-activation reinforces it via Hebbian learning, and prolonged disuse leads to decay.

---

## 3. The staged memory pipeline

Every memory goes through four stages from ingestion to maturity:

```
raw ──→ indexed ──→ enriched ──→ consolidated
 │         │           │              │
 │ sync    │ sync      │ async (bg)   │ async (scheduled)
 │         │           │              │
 raw text   indexed     fill strong    mature
 immutable  discover    semantics      Hebbian/
            assocs      goal/          decay/
            BM25+entity preference/    compaction
            +sem+time   emotion/decision
```

### Raw → Indexed (sync, completed immediately on write)

1. Tokenization (mixed Chinese/English)
2. Instant understanding extraction (entities/topics/explicit causality/events/time, via deterministic rules or LLM)
3. Generate AssociationKeys (entity keys / topic keys / time-bucket keys / SimHash signatures)
4. Candidate discovery: search existing memories for potential matches using keys
5. Multi-dimensional association scoring: EntityOverlap + TopicOverlap + Temporal + SemanticSimilar + ...
6. Edge creation: score ≥0.55 → strong edge; 0.25–0.55 → weak edge; <0.25 → observation zone
7. Persistence: write to memory_log + memory_kv + inverted index + BM25 index

By the time `engine.write()` returns, the memory is already in the Indexed stage.

### Indexed → Enriched (async, background worker)

- Fill in strong semantic dimensions: goals, preferences, emotions, decision rationale
- Rule-based (degraded backend) or LLM inference (API backend)
- Update the MemoryUnit's `understanding` field after enrichment
- Failures don't panic; they return with a WriteWarning

### Enriched → Consolidated (async, scheduled trigger)

- **Hebbian reinforcement**: read co-activation pairs from activation_log, reinforce the corresponding edge strengths
- **Decay**: every edge is multiplied by `decay_per_cycle` (default 0.97) per cycle
- **Compaction**: weak edges below `min_retained_strength` get archived; nodes exceeding out-degree limits keep only their strongest edges
- **Summary compression**: multiple similar low-level memories → Summarizer → one high-level summary memory + `covers` links

---

## 4. Spreading activation: retrieval works like human recall

HIPPMEM's retrieval is not "sort by match score"; it simulates the brain's spreading activation process:

```
  Input: "Why did we move away from RocksDB?"
          │
          ▼
  ┌─ Step 0: Seed recall ─────────────────┐
  │  BM25 channel   → "RocksDB" → memory_42│
  │  Entity channel → "RocksDB" → memory_42│
  │  Entity channel → "redb"    → memory_78│
  │  Semantic       → "storage choice" → m55│
  └────────────────────────────────────────┘
          │
          ▼
  ┌─ Hop 1: Spread ───────────────────────┐
  │  memory_42 → [EntityOverlap] → mem_99 │
  │  memory_42 → [Causal]        → mem_78 │
  │  memory_78 → [Temporal]      → mem_33 │
  │  (energy ×0.55 decay)                  │
  └────────────────────────────────────────┘
          │
          ▼
  ┌─ Hop 2: Continue spreading ───────────┐
  │  mem_78 → [EntityOverlap] → mem_12     │
  │  (energy ×0.55×0.55 = 30%)             │
  │  below min_propagation_energy (5%) → stop│
  └────────────────────────────────────────┘
          │
          ▼
  ┌─ Merge + Rerank ───────────────────────┐
  │  Same memory hit via multiple paths    │
  │    → keep the highest energy           │
  │  Sort by final_score                   │
  │  Attach matched_dimensions             │
  │  Attach activation_trace               │
  │  Attach warnings (contradiction/       │
  │    stale/low-confidence)               │
  └────────────────────────────────────────┘
```

### The essential difference from "vector similarity search"

| | Vector search | HIPPMEM spreading activation |
|----|---------------|------------------------------|
| Recall mechanism | One-shot similarity match | Multi-channel seeds + graph diffusion |
| Result source | Direct hits only | Up to 2–3 hop indirect associations |
| Explainability | "vector distance 0.87" | "spread from memory_42 via a Causal edge" |
| Cold start | Requires building a vector index first | BM25 + entity rules always available |

---

## 5. Deterministic degradation: runs with zero dependencies

HIPPMEM can run without any external API. This is achieved via the **deterministic degradation backend**:

| Capability | API backend | Deterministic degradation |
|------------|-------------|---------------------------|
| **Embedding** | OpenAI text-embedding-3 (1536d) | hash → deterministic 256d |
| **Entity/topic extraction** | LLM (Claude/GPT) | Rules: proper-noun detection + jieba POS tagging |
| **Causal extraction** | LLM | Rules: "because…so…", "leads to…", "then…" connector matching |
| **Reranking** | LLM | BM25 score normalization |
| **Summarization** | LLM | Extractive summary (first sentence + entity list) |

Degraded semantic accuracy is indeed lower (256d SimHash is no match for 1536d dense vectors), but all core loops (write → index → retrieve → explain → consolidate) still work. This ensures:

- **CI tests** need no API key
- **Offline environments** are deployable
- **Privacy-sensitive scenarios** keep data on local machine

---

## 6. Consolidation: "forgetting" and "reinforcement" of memories

### Hebbian learning — "neurons that fire together, wire together"

```
activation_log:
  retrieval_123: [memory_42, memory_78, memory_99]

→ memory_42 ↔ memory_78 co-activated → edge strength +hebbian_learning_rate(0.08)
→ memory_42 ↔ memory_99 co-activated → edge strength +0.08
→ memory_78 ↔ memory_99 co-activated → edge strength +0.08
```

Memories frequently retrieved together automatically strengthen their associations. This is "use it, remember it."

### Decay — natural forgetting

- Every edge is multiplied by `decay_per_cycle` (0.97) per cycle
- Below `min_retained_strength` (0.10) → archived
- The **protected set** does not decay: Causal edges / Correction edges / Supersedes edges / strong preference edges / decision rationale

### Compaction — pruning weak associations

- Node out-degree exceeds limit → keep only the top N strongest edges
- Long-unused observation-zone edges → pruned
- Archived edges are recorded in correction_overlay (traceable)

---

## 7. Capability overview

| # | Capability | Meaning |
|---|------------|---------|
| 1 | Native MemoryUnit model | Content + understanding + activation as one unit |
| 2 | Write-time structured understanding | Auto-extract entities/topics/events/causality |
| 3 | Write-time association discovery | Multi-dimensional candidate discovery + association scoring |
| 4 | Native association links | 14 edge types + evidence + confidence |
| 5 | Multi-channel recall | BM25 + entity + semantic + temporal + graph diffusion |
| 6 | Spreading activation retrieval | 1–2 hops + edge-weight modulation + cycle prevention + pruning |
| 7 | Explanation paths | activation_trace + matched_dimensions |
| 8 | Activation log | Full recording of retrieval/usage/feedback |
| 9 | Hebbian reinforcement | Co-activation → edge reinforcement |
| 10 | Decay + compaction | Natural forgetting + weak-edge cleanup |
| 11 | Contradiction detection | Auto-discovery of mutually contradictory memories |
| 12 | Observation-zone pruning | Auto-filtering of low-quality edges |
| 13 | Causal tracing | Explicit causal extraction + Causal edges |
| 14 | Deterministic degradation | Full end-to-end loop with zero external APIs |
