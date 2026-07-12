# HIPPMEM Glossary

This glossary is the terminology anchor for HIPPMEM. Each term is given a one-sentence definition; full specifications live in the architecture and data-model documents.

## Core Objects

### MemoryUnit
HIPPMEM's core data object — not a plain document, but a network node carrying the three layers of memory: content, associations, and activation history.

### MemoryContent
The part of a `MemoryUnit` that holds the raw text, summary, normalized text, language, and content type.

### WriteContext
The environment in which a write occurs — session, task, project, user, adjacent memories, and source references — and the most valuable information that is hard to reconstruct after the fact.

### MemoryUnderstanding
The structured understanding of a memory produced by an algorithm or model, including entities, events, goals, decisions, preferences, emotions, causal claims, contradiction hints, importance, and confidence.

### AssociationKeys
The multi-dimensional index keys (entity, temporal, lexical signature, semantic signature, topic, emotion, goal, event, causal) used to quickly discover candidate associations at write time.

### AssociationLink
A native association edge between two `MemoryUnit`s, carrying type, direction, strength, confidence, evidence, formation time, and activation history; a first-class citizen of the data model.

### ActivationState
The history of how a memory has been retrieved, co-activated, reinforced, and decayed over time.

### Provenance
The origin, evidence, generation method, and trustworthiness of a memory, supporting traceability and auditability.

### MemoryLifecycle
The state machine of a memory: active, compressed, archived, superseded, negated, and so on.

## Understanding Frames

These are the structured "frames" inside `MemoryUnderstanding`:

- **EntityMention** — a person, project, library, file, organization, or concept mentioned in the text, with its position and type.
- **EventFrame** — the time, participants, action, and result of an event.
- **GoalFrame** — a user or project goal, its constraints, and its state; a strong-semantic dimension.
- **DecisionFrame** — the content, rationale, time, and override status of a decision; a strong-semantic dimension.
- **PreferenceFrame** — the preference object, direction (like / dislike), strength, and validity; a strong-semantic dimension.
- **EmotionFrame** — the emotion type, intensity, and trigger object; a strong-semantic dimension.
- **CausalClaim** — a directed cause-to-effect assertion with confidence and evidence.
- **ContradictionHint** — a pointer to two pieces of information that may conflict; a strong-semantic dimension.

## Understanding Tiers

- **Immediate Dimensions** — entities, time, keyword and semantic signatures, preliminary topics, and explicit causal clues, produced in the synchronous `indexed` stage.
- **Strong-Semantic Dimensions** — goals, preferences, emotions, decisions, implicit causality, and contradiction hints; first-class capabilities that may be filled asynchronously with low initial confidence.
- **Consolidation Dimensions** — summary merging, long-term state evolution, cross-session contradictions, and correction relationships, produced by background consolidation in the `consolidated` stage.

## Staged Memory

A write is not a single insert but a staged state flow:

```
raw -> indexed -> enriched -> consolidated
```

- **raw** — original content persisted and traceable.
- **indexed** — immediate dimensions produced; lightweight entities, time, keywords, semantic signature, and initial candidate associations; the synchronous write returns here.
- **enriched** — strong-semantic dimensions filled in.
- **consolidated** — summaries, merges, weak-edge cleanup, cross-session corrections, and long-term state evolution completed.

## Retrieval and Activation

### Spreading Activation
HIPPMEM's retrieval paradigm: energy propagates from seed memories along association edges hop by hop, and memories hit by multiple paths are reinforced; distinct from single-stage nearest-neighbor search.

### Seed Discovery / Multi-Channel Recall
The first step of retrieval, in which each recall channel (BM25, entity, semantic, temporal, goal, event, recent activation) independently contributes candidates as spreading seeds; no single channel may monopolize the result.

### Recall Channel
An independent candidate-discovery pathway with its own scoring and observable contribution.

### Activation Energy
The scalar energy a node holds during spreading; seeds are energized by an initial-energy formula and propagation decays per hop according to a propagated-energy formula.

### Rank-based Fusion (RRF)
The seed-fusion strategy: each recall channel ranks independently and ranks are combined via the Reciprocal Rank Fusion formula `Σ 1/(k + rank)`; ranks are comparable across channels while raw scores are not.

### Activation Trace
The step-by-step record of how energy spread from seeds to results during a retrieval; the data basis of the explanation path.

### Explanation Path
The "why was this recalled" statement attached to a result — matched dimensions, edges traversed, contributing scores, and risk warnings; one of HIPPMEM's core differentiators.

### Hop
The number of edge steps taken in spreading activation; the default is 1–2 hops, with 3 hops available as an enhanced or diagnostic mode.

### Fan-out
The maximum number of neighbors a node may expand along a given edge type in a single hop, used for pruning and noise control.

### Observation Zone
A holding area for low-confidence but potentially valuable candidate associations: they are not promoted to strong edges immediately; if they are retrieved, referenced, or co-activated within an observation window they are promoted, otherwise they are cleaned up.

### RetrievalMode
The retrieval tier (such as Fast, Balanced, Deep, Diagnostic) that determines channel count, hop count, and whether a reranker is enabled.

## Link Types

- **EntityOverlap** — shares an entity; suited to lateral recall.
- **TemporalAdjacent** — adjacent in time; suited to narrative context, decays quickly.
- **SemanticSimilar** — semantic neighbor.
- **TopicRelated** — same topic cluster.
- **SameGoal** — serves the same goal.
- **SameEvent** — belongs to the same event chain.
- **Causal** — causal relation; directed, suited to directional tracing; must carry evidence.
- **EmotionalResonance** — similar emotional state or emotional turning point.
- **Contradiction** — two memories conflict; surfaced as a risk hint, not used directly as fact; must carry evidence.
- **Correction** — a new memory explicitly revises an old one; preferentially overrides the old memory; must carry evidence.
- **Elaboration** — one memory expands or supplements another.
- **CoActivation** — historically recalled together; suited to discovering habitual associations.
- **Supersedes** — a new memory overrides an old one on time or authority.
- **Deprecated** — an old memory is stale but retained for historical continuity.

## Evolution and Consolidation

### Hebbian Reinforcement
"Cells that fire together wire together": when memories are co-activated and then used or confirmed, the strength of their connecting edge is increased; repeatedly co-activated memories without an edge get a new `CoActivation` edge.

### Decay
The strength of long-unused, low-confidence, low-value edges slowly decreases; edges below a threshold with no retention reason are archived or deleted.

### Consolidation
Background maintenance of the memory network: Hebbian reinforcement, decay, compression and merging, contradiction discovery, weak-edge cleanup, and index rebuilds.

### Compaction
A sub-process of consolidation that governs the association graph's edges — weak-edge decay, archival, deletion, merging, and promotion of observation edges — triggered by observable thresholds.

### Summary Memory
A memory that compresses many similar low-level memories into a summary, retaining traceability to the originals through a `covers` field.

## Storage

### Memory Log
The append-only, immutable store of raw memory writes, favoring auditability, recovery, and index rebuilds.

### Overlay
The mutable state layer separated from the append-only log: `link_overlay` (edge strength, activation, decay), `summary_overlay` (summaries, supersedence), and `correction_overlay` (corrections, conflicts, deprecation).

### Memory Store
The storage layer that unifies the memory log, KV store, full-text index, semantic index, association graph, activation log, and consolidation queue, exposing only memory semantics rather than underlying library concepts.

### SemanticSignature
A multi-layer semantic fingerprint — lexical SimHash, dense embedding reference, binary code, and topic MinHash — whose components complement each other to improve recall stability.

## Model Backends

### Model Backend
A pluggable implementation of a model capability (embedding, extraction, reranking, summarization), available as either an API backend or a deterministic fallback backend.

- **Embedder** — text to dense vector.
- **Extractor** — text to structured understanding (entities, goals, emotions, decisions, causality, contradictions).
- **Reranker** — (query, candidates) to a reranking score.
- **Summarizer** — multiple memories to a summary.

### Deterministic Backend
A reproducible implementation that requires no network and no real model — hash-based pseudo-vectors, rule-based extraction, BM25 as a reranker, extractive summarization — used for offline development, CI, and contract testing.

## Evaluation

- **Recall@K / Precision@K** — standard IR metrics computed against structured id-level ground truth.
- **Explanation Accuracy** — whether the explanation path is truthful (matched edges and dimensions agree with ground truth).
- **Contradiction Awareness** — whether conflicts between new and old information are surfaced.
- **Baselines** — pure BM25, pure embedding, hybrid, RAG with summary, and HIPPMEM-full, all run through the same interface for comparison.
