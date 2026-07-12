# HIPPMEM: A Native Association Memory Engine

## What HIPPMEM Is

HIPPMEM is a long-term memory component for AI agents. Its goal is not to "store some text and search it back with embeddings." It is to build a memory system that lets an agent:

- Remember facts: what a user said, what happened, who is connected to whom.
- Remember relations: how events, entities, emotions, goals, preferences, and decisions connect to one another.
- Remember change: how preferences evolve, how state shifts, how contradictions appear and resolve.
- Remember causality: what led to what, which explanation supports which judgment.
- Associate actively: given a current context, surface memories that are relevant but not necessarily lexically similar.
- Grow stronger with use: frequently co-activated connections are reinforced; long-unused ones decay.
- Be explainable: not just return "relevant memories," but also explain why they are relevant and through which path.

The intended effect is that an agent using HIPPMEM maintains continuity across sessions, projects, and tasks in a way that feels closer to a long-term collaborator than to a stateless query interface.

## First Principles

### Memory Is Not Data — It Is Data + Associations + Activation History

An isolated piece of text is not a memory; it is a record. A real memory has at least three layers:

1. **Content**: what this memory actually says.
2. **Associations**: which other memories, entities, topics, times, emotions, and goals it relates to.
3. **Activation**: in which situations it has been recalled, with which other memories, and whether its importance has grown.

Therefore, HIPPMEM's core object is not a `Document` but a `MemoryUnit`.

### Association Discovery Must Happen at Write Time

Write time carries the richest context: the current conversation window, the user's expressed goal and tone, the active task, project, and file, adjacent events, the agent's reasoning and summary, tool outputs, and the user's confirmation or rejection. If we only persist raw text and try to guess relationships at retrieval time through embeddings or keywords, most of that context is already lost.

Architecture consequence: write-time association discovery is the default path, not a remedial step.

### Retrieval Is Activation, Not Lookup

Human recall is not a database query; it is spreading activation. A cue wakes a few seed memories, seeds propagate energy along association edges, memories hit by multiple paths are reinforced, low-energy or low-relevance memories naturally fade, and what finally surfaces is a context network rather than a set of isolated texts.

Architecture consequence: the retrieval model is `seed discovery + spreading activation + reranking + explanation`, not a single-stage nearest-neighbor search.

### Semantic Similarity Is One Kind of Association, Not the Whole of Memory

Conventional RAG over-relies on semantic similarity. Consider:

- "I like cats" and "yesterday I adopted a Ragdoll" are semantically related but lexically dissimilar.
- "I worked over the weekend because the project was delayed" and "I've been tired lately" are connected through cause and state.
- "Stop using that library" and "the last production incident" are connected through a decision rationale.
- "Today I'm discouraged" and "a similar mood three months ago" are connected through emotional resonance.

Architecture consequence: HIPPMEM must natively support multi-dimensional association — entity, temporal, semantic, topic, event, causal, emotional, goal, task, and co-activation.

### A Memory System Must Self-Organize, Self-Reinforce, and Self-Forget

A long-term memory system cannot only accumulate, or it becomes a noise graveyard. It must integrate new memories into the existing network, reinforce frequently used connections, decay obsolete ones, merge redundant memories, compress or archive low-value ones, let new evidence revise old conclusions, and hold contradictory memories with an explicit explanation.

Architecture consequence: HIPPMEM has a background consolidation layer, not only foreground read/write APIs.

## Core Capabilities

HIPPMEM's first version deliberately ships a complete memory loop rather than a minimal retrieval wrapper. The following capabilities are each first-class:

1. **Native `MemoryUnit` model** — content, write context, structured understanding, association keys, links, activation state, lifecycle, and provenance in a single object.
2. **Write-time structured understanding** — extraction of entities, topics, events, goals, preferences, emotions, and causal clues at write time.
3. **Write-time association discovery** — at minimum entity, temporal, semantic, topic, goal, and shared-context candidates discovered as the memory is written.
4. **Native association links** — edges with type, direction, strength, confidence, and evidence.
5. **Multi-channel recall** — BM25, entity, semantic, graph-spreading, and co-activation channels feeding the same candidate pool.
6. **Spreading-activation retrieval** — 1–2 hops by default, with per-type edge weights, cycle prevention, and energy pruning.
7. **Explanation paths** — every returned result explains why it was recalled.
8. **Activation log** — records of retrievals, usage signals, and co-activation events.
9. **Hebbian reinforcement** — connections that are co-activated and then used or confirmed grow stronger.
10. **Baseline decay** — long-unused weak edges slowly decay and may be archived.
11. **Contradiction and correction relationships** — conflicts are recorded and surfaced as risk hints at retrieval time.
12. **Background consolidation jobs** — summarization, merging, index updates, weak-edge cleanup.
13. **Evaluation framework** — fixed scenarios that measure memory effectiveness against baselines.
14. **Observability and diagnostics** — inspection of indexes, edges, activation history, and per-channel contribution.
15. **Full data export** — JSONL dump of all `MemoryUnit` records for backup, migration, and external tooling.

Strong-semantic dimensions — goals, preferences, emotions, decisions, implicit causality, contradiction hints — are first-class capabilities, not optional add-ons. They may be filled in asynchronously, may start with low confidence, and may initially land in an observation zone, but they are not cut from the design.

## How HIPPMEM Differs Fundamentally

### Versus a Vector Database

A vector database stores records and answers nearest-neighbor queries in embedding space. It is a retrieval primitive, not a memory system. It does not model relationships as first-class edges, it has no activation history, it does not consolidate, and it cannot explain why a memory was recalled beyond a similarity score. HIPPMEM uses embeddings as one recall channel among several; it does not let embeddings define what memory means.

### Versus a Graph Database

A graph database stores nodes and edges and answers graph-pattern queries. It is a generic substrate with no memory-specific semantics: no activation energy, no Hebbian reinforcement, no decay, no write-time association discovery, no contradiction and correction primitives. HIPPMEM may use graph traversal internally, but the notions of `MemoryUnit`, `AssociationLink`, activation, consolidation, and forgetting are HIPPMEM's own data model and algorithmic semantics. An external graph database can be an implementation detail; it cannot define the product boundary.

### Versus a RAG Pipeline

Conventional RAG chunks documents, embeds them, and retrieves similar chunks at query time. It is similarity search plus generation. It discards write-time context, treats "relevant" as a single similarity score, has no network structure, no reinforcement, no forgetting, and no explanation path. HIPPMEM is built on the opposite assumption: a memory is a node in an evolving association network, and retrieval is activation across that network, not lookup against a flat index.

## Architecture Overview

HIPPMEM is structured as seven layers, but the structure is not a traditional three-tier stack. It is a dynamic memory network that grows at write time, activates at retrieval time, and is continuously consolidated in the background.

```
Agent API
  write / retrieve / explain / consolidate / inspect / dump / eval
        |
Context Ingestion Layer
  raw input, session context, tool results, user feedback, task state
        |
Memory Understanding Layer
  summary, entities, events, goals, emotions, causality, time, semantic signature, embedding
        |
Associative Write Engine
  multi-dimensional candidate discovery, association scoring, bidirectional edge creation,
  confidence recording, explanation generation
        |
Native Memory Store
  memory log, attribute index, full-text index, semantic index, association graph, activation log
        |
Spreading-Activation Retrieval
  multi-channel seed recall, spreading activation, reranking, conflict checking, explanation path
        |
Consolidation and Evolution Layer
  Hebbian reinforcement, decay, compaction, merging, correction, archival, eval feedback
```

The core data object is `MemoryUnit`, which carries content, write context, structured understanding, association keys, links, activation state, lifecycle, and provenance. Two neighboring memories are connected by an `AssociationLink` with a typed `LinkType` (`EntityOverlap`, `TemporalAdjacent`, `SemanticSimilar`, `TopicRelated`, `SameGoal`, `SameEvent`, `Causal`, `EmotionalResonance`, `Contradiction`, `Correction`, `Elaboration`, `CoActivation`, `Supersedes`, `Deprecated`), a direction, a strength, a confidence, and recorded evidence.

## Write Time: Growing the Network

A write is not a single insert. It is a staged process:

```
raw -> indexed -> enriched -> consolidated
```

- **raw** — original content persisted and traceable.
- **indexed** — lightweight entities, time, keywords, semantic signature, and initial candidate associations are produced in the synchronous path so the write can return quickly.
- **enriched** — strong-semantic dimensions (goals, preferences, emotions, decisions, implicit causality, contradiction hints) are filled in, possibly asynchronously.
- **consolidated** — summaries, merges, weak-edge cleanup, cross-session corrections, and long-term state evolution are completed by background jobs.

At each stage, candidate old memories are discovered through multiple registers and indexes — entity inverted index, temporal adjacency, semantic neighbors, topic clusters, goal and event matches, causal clues, and recent activation. Candidates are scored across multiple dimensions, and the multi-dimensional hit rule is explicit: a candidate hit on entity plus time plus goal is generally more valuable than one hit only on a high embedding similarity.

Edges are created in strong/weak tiers (e.g. a handful of strong edges, up to a few dozen weak ones). Causal, contradiction, and correction edges must carry evidence. Low-confidence but potentially valuable candidates enter an **observation zone** rather than becoming strong edges immediately; if they are later retrieved, referenced, or co-activated, they are promoted, otherwise they are cleaned up.

## Retrieval: Spreading Activation, Not a Single Search

Retrieval follows this flow:

```
1. Understand the query: entities, goals, topics, time, emotions, task state.
2. Multi-channel seed recall: BM25, entity, semantic, temporal, goal, event, recent activation.
3. Assign initial activation energy to each seed.
4. Propagate energy along association edges for 1-3 hops.
5. Merge energy from multiple paths converging on the same memory.
6. Check for contradictions, supersedence, expiry, and corrections.
7. Rerank with a model or rules.
8. Return a memory bundle: content + score + explanation path + risk warnings.
```

Initial energy combines query match, context match, importance, freshness, and reliability. Propagated energy is the source energy scaled by link strength, link confidence, a per-hop decay factor, and a per-type modifier. Different link types have different propagation rules: `EntityOverlap` is good for lateral recall, `Causal` is good for directional tracing, `Correction` should preferentially override the older memory, `Contradiction` should be surfaced as a risk rather than used as fact, and `CoActivation` is good at discovering habitual associations.

Propagation has explicit stopping conditions to prevent the graph from degenerating into an unbounded full-graph scan:

- A `visited` set and path state prevent cycles.
- Energy below `min_propagation_energy` terminates a branch immediately.
- Per-node and per-edge-type fan-out caps prioritize high-strength, high-confidence, evidenced edges.
- Multiple paths converging on the same node merge energy rather than re-expanding.

Each result carries an explanation path: which dimensions matched, which edges it traversed, the contributing channels, and any risk hints such as "a newer `Correction` edge revises part of this conclusion." An agent should not blindly trust retrieval results; the explanation path lets it decide which memories to cite directly and which to treat cautiously.

## Consolidation, Reinforcement, Decay, Forgetting

### Hebbian Reinforcement

When several memories are co-activated in a retrieval and the result is used by the agent or confirmed by the user, the connections between them are reinforced. If two memories are repeatedly co-activated but have no edge, a `CoActivation` edge is created. Usage signals include the result being cited by the agent, the user confirming the answer, the follow-up task succeeding, or the same combination recurring in high-quality answers.

### Decay

Long-unused, low-confidence, low-value edges slowly decay. Edges below a threshold with no retention reason are archived or deleted. Certain edges are protected from easy deletion: explicit decision rationales, causal chains, long-standing user preferences, correction records, compliance- or safety-relevant memories, and high-confidence contradiction records. Edge governance runs in the background and is triggered by observable thresholds (per-node weak-degree limits, per-node total out-degree limits, edges below `min_retained_strength` that have been inactive for a long time). Governance policies are per-type — a single threshold must not be applied to causal, correction, contradiction, long-term preference, and ordinary semantic-similarity edges alike.

### Compression and Merging

Large numbers of similar low-level memories are compressed into summary memories. The original evidence remains traceable through a `covers` field, preventing summary hallucination from contaminating the fact layer.

### Contradiction and Correction

When a new memory conflicts with an old one, the old memory is not simply deleted. Instead, a relationship is recorded: `Contradiction` (the two conflict), `Correction` (the new one explicitly revises the old), `Supersedes` (the new one overrides on time or authority), or `Deprecated` (the old one is stale but kept for historical continuity). Real long-term collaboration means users, projects, and judgments all change — the system must understand "this was once true and is no longer."

## Storage and Index Layout

The storage layer serves high-throughput writes, fast multi-dimensional recall, native association edges, rebuildable indexes, crash recovery, observability, and background compaction. The recommended layout:

```
HIPPMEM Store
  memory_log          append-only MemoryUnit records
  memory_kv           memory_id -> MemoryUnit, plus entity/topic/goal/event inverted indexes
  fulltext_index      BM25 / keyword / phrase search
  semantic_index      dense or quantized vectors, binary semantic codes, topic clusters
  association_graph   outgoing links, incoming links, link evidence
  activation_log      retrieval traces and co-activation events
  consolidation_queue pending background jobs
```

Original memories are stored append-only; mutable state lives in overlays (`link_overlay` for strengths, activations, and decay; `summary_overlay` for summaries and supersedence; `correction_overlay` for corrections, conflicts, and deprecation). This keeps memory content traceable while letting interpretation, weights, and lifecycle evolve. Mature libraries (embedded KV such as redb or SQLite, Tantivy for full text, HNSW-style vector indexes, local model runtimes) may be used internally, but the external surface is always `MemoryUnit`, `AssociationLink`, and activation traces — never the concepts of the underlying library.

## Model Strategy

Embeddings, rerankers, and LLMs are enhancement capabilities, not the memory system itself. HIPPMEM combines several complementary signals: BM25 for keywords, code names, and proper nouns; dense embeddings for semantic generalization; binary semantic codes for fast approximate matching; entity indexes for stable anchors; the association graph for multi-hop association; a reranker for final ordering; and an LLM judge for complex causality, contradictions, summaries, and explanations.

A `SemanticSignature` bundles several fingerprints — lexical SimHash, a dense embedding reference, a binary code, and a topic MinHash — so that the weaknesses of any single signal are covered by the others.

A local or otherwise controllable LLM may extract structured understanding at write time, judge causality and contradictions, generate summaries and natural-language explanations, merge memories during consolidation, and generate evaluation questions. LLM outputs always carry confidence and evidence and are never allowed to directly rewrite the fact layer.

## Design Philosophy: Performance Serves Memory Effectiveness

Microsecond writes and millisecond retrievals are excellent targets, but they are not the only targets. As long as the interactive experience on a server or desktop is fast enough, the system may accept partial background processing, progressive indexing, and deferred consolidation in exchange for materially better memory quality.

Indicative targets for a single-node deployment serving on the order of 100K to 1M memories are: ordinary writes returning synchronously in under 50ms with heavy work queued; simple retrieval under 100ms; spreading-activation retrieval under 300ms; reranked retrieval under 1s; background consolidation fully asynchronous. These numbers are not constitutional. If a target conflicts with memory effectiveness, the correct response is to preserve the full memory semantics and solve the performance problem with async processing, pruning, queue governance, and observable tuning — not to cut a memory dimension.

The same philosophy governs cold start. Cold-start difficulty is real, but it is never solved by deleting dimensions — that also deletes the long-range associative signal long-term memory depends on. The solution is conservative default weights, multi-dimensional corroboration before promoting a candidate to a strong edge, observation zones for low-confidence candidates, observable thresholds, and a fixed evaluation set for replay-based tuning.

## What HIPPMEM Will Not Do

The current focus is server-side and desktop deployment with modern CPUs, SSDs, local file systems, background threads, and optional local models or accelerators. The following are explicitly out of scope for the current version and are not used as reasons to trim memory semantics:

- Mobile and browser deployment; minimal binary size; ultra-low-power edge inference.
- Self-trained embedding models; a full causal reasoning engine; full personality modeling.
- Large-scale distributed storage; multi-tenant permission systems.
- A graphical management UI; full timeline visualization.
- Microsecond-level optimization that would sacrifice memory quality.

Equally, HIPPMEM will not ship as a thin vector-search wrapper, will not delegate all relationships to an external graph database, will not keep only summaries and discard raw evidence, will not let an LLM rewrite facts without evidence, and will not claim "good memory" without a measurement against baselines.

## How Success Is Measured

HIPPMEM's success is judged by the agent's actual memory behavior, not by architectural elegance. It is compared against pure BM25, pure embedding search, BM25-plus-embedding hybrid, and conventional RAG with summary memory, all run through the same interface. Evaluation tasks cover fact recall, preference recall, cross-session project continuity, causal tracing, contradiction identification, state-change tracking, implicit association (recall without keyword overlap), noise resistance, long-tail recall of important but old memories, and explanation quality. Core metrics include `Recall@K`, `Precision@K`, explanation accuracy, contradiction awareness, long-tail recall, end-to-end user outcome, latency, and cost.

## Conclusion

HIPPMEM is a native association memory engine, not a variant of conventional RAG. Its core belief is:

> Memory is not searched out — memory is activated out.

To achieve a step-change in effectiveness, the system establishes a complete memory-network loop from the first version: write-time understanding, write-time association, native edge storage, multi-channel recall, spreading activation, explanation paths, usage feedback, background consolidation, contradiction and correction, and continuous evaluation. The performance numbers can be tuned later; the principles — that a memory is a network node, that associations are first-class, that retrieval is multi-channel activation, that results are explainable, and that memory must be able to evolve — are not negotiable.
