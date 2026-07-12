# Algorithms

> This document describes HIPPMEM's retrieval and memory-evolution algorithms: multi-channel seed recall, rank-based fusion, spreading activation, reranking, Hebbian consolidation, decay, and compaction. It covers the formulas and their default parameters, not the Rust types (see the [Data Model](data-model.md)).

HIPPMEM retrieves memories by combining many weak signals rather than trusting a single similarity score. A query is understood into the same multi-dimensional key structure used at write time, then multiple **recall channels** run in parallel to produce seed candidates. Seeds are fused by **Reciprocal Rank Fusion (RRF)**, energy is propagated across the association graph by **spreading activation**, and the final list is optionally **reranked**. In the background, **Hebbian consolidation** strengthens useful links, **decay** weakens unused ones, and **compaction** merges clusters.

All numeric parameters are configurable defaults; none is a hard-coded constant. Every formula that needs "now" reads from an injectable `Clock`, and every random source is seed-injected, so the entire pipeline is deterministic and reproducible in tests.

---

## 1. Parameter overview

Parameters are grouped and exposed through an `AlgoParams` configuration struct. The most important groups:

| Group | Key parameters | Default |
|-------|---------------|---------|
| Association weights | `w_entity` … `w_importance` (10 weights) | sum ≈ 1.10, normalized after scoring |
| Multi-dimension corroboration | `multi_dim_bonus`, `multi_dim_min_dims` | 0.15, 3 |
| Edge building | `strong_edge_threshold`, `strong_edge_max`, `weak_edge_max`, `edge_build_min_score` | 0.55, 8, 24, 0.25 |
| Observation zone | `observation_enter_max`, `observation_window_ms` | 0.55, 14 days |
| Seed activation | `a_query_match`, `b_context_match`, `c_importance`, `d_freshness`, `e_reliability` | 0.40, 0.20, 0.15, 0.15, 0.10 |
| Spreading | `decay_factor`, `min_propagation_energy`, `fan_out_default`, `max_hops_default` | 0.55, 0.05, 6, 2 |
| Hebbian | `hebbian_learning_rate`, `coactivation_create_threshold`, `strength_max` | 0.08, 3, 1.0 |
| Decay | `decay_per_cycle`, `min_retained_strength`, `stale_unactivated_ms` | 0.97, 0.12, 30 days |
| Rerank | `rerank_top_n`, `seed_per_channel` | 50, 20 |
| RRF | `rrf_k`, `rrf_w_topic`, `rrf_w_temporal` | 1.0, 0.3, 0.3 (others 1.0) |
| Cold start | `cold_start_count`, `single_semantic_penalty` | 500, 0.60 |

---

## 2. Association scoring (write time)

When a new memory `M_new` is written, the engine scores it against existing memories `M_old` to decide which edges to build. The score is a weighted sum across ten match dimensions, each normalized to `[0, 1]`.

### Per-dimension sub-scores

| Dimension | Formula |
|-----------|---------|
| `entity_score` | Jaccard of entity keys |
| `semantic_score` | max(cosine(dense), 1 − hamming(binary_code)/128, SimHash similarity) |
| `temporal_score` | Shared-bucket weighting (session > hour > day); else `exp(−Δt / τ)`, τ = 7 days |
| `topic_score` | Jaccard or MinHash-estimated topic overlap |
| `goal_score` / `event_score` | Jaccard of goal/event keys |
| `emotion_score` | 1 if primary emotions match and intensities are close; else vector distance |
| `causal_score` | Cross-hit of causal keys (one's cause hits the other's effect) |
| `co_context_score` | Proportion of shared conversation/session/project/task IDs |
| `importance_bonus` | `M_old.understanding.importance` |

### Composite score

```
raw = Σ_dim (dim_score * w_dim)
hit_dims = count of dimensions with dim_score > dim_hit_threshold   (default 0.20)
if hit_dims >= multi_dim_min_dims:                                  (default 3)
    raw += multi_dim_bonus * (hit_dims / total_dims)
score = clamp01(raw / NORM)          # NORM = Σ w_dim + multi_dim_bonus
```

### Cold-start guardrail

When total memory count is below `cold_start_count` (default 500), or whenever a single dimension dominates:

- If `hit_dims == 1` and that dimension is `Semantic`, `score` is multiplied by `single_semantic_penalty` (default 0.60). This keeps a lone semantic match from forming a strong edge and routes it to the observation zone instead.
- Only multi-dimension corroboration (`hit_dims >= multi_dim_min_dims`) may promote a memory into the strong-edge band.
- After cold start ends, the penalty linearly returns to 1.0.

This guardrail exists because pure semantic similarity is the noisiest channel early in a memory store's life.

---

## 3. Edge building and the observation zone

Candidates are processed in descending score order:

```
for cand in candidates (sorted by score desc):
    if score >= strong_edge_threshold and strong_count < strong_edge_max:
        build Confirmed edge, strength = init_strength(cand), strong_count++
    elif score >= observation_enter_max and weak_count < weak_edge_max:
        build Confirmed weak edge, weak_count++
    elif edge_build_min_score <= score < observation_enter_max:
        build Observing edge (enter observation zone), since = now
    else:
        discard
# ensure at least strong_edge_min strong edges when enough candidates exist
```

`init_strength(cand) = clamp01(init_strength_base + 0.4 * (score − strong_edge_threshold))`, so stronger matches start with higher strength.

**Link-type assignment** follows the dominant hit dimension (entity-dominant → `EntityOverlap`, causal hit → `Causal`, …). When multiple dimensions hit, the priority is: `Correction`/`Contradiction` > `Causal` > `SameGoal`/`SameEvent` > `Semantic`/`Entity` > `Temporal`/`Topic`/`Emotion`.

**Evidence requirement:** `Causal`, `Contradiction`, and `Correction` edges must populate `evidence.text_spans`. An edge that cannot produce evidence is downgraded or not built.

**Edge deduplication:** if `(target_id, link_type)` already exists, the edge is merged — `strength = max(old, new)`, evidence is unioned, `confidence = max` — rather than duplicated.

### Observation-zone promotion and cleanup

An `Observing` edge is promoted to `Confirmed` (with strength raised to the weak-edge baseline) when, within `observation_window_ms`, it is retrieved, referenced, or co-activated at least once. If the window expires with no activation, compaction deletes it. Retrieval modes treat observation edges differently: `Balanced` excludes them from spreading seeds but may surface them as evidence; `Deep`/`Diagnostic` include them; `Fast` ignores them.

---

## 4. Multi-channel seed recall

A retrieval starts by understanding the query — extracting its entities, topics, temporal keys, goals, and causal claims using the same `Extractor` and `AssociationKeys` generation used at write time. Each recall channel then runs in parallel and returns `Vec<(MemoryId, raw_score)>`, with `raw_score` normalized to `[0, 1]`.

| Channel | Candidate construction | raw_score |
|---------|------------------------|-----------|
| `Bm25` | Tokenized query → Tantivy BM25 top-N | BM25 / batch max |
| `EntityInverted` | Query entities → entity index union | hit entities / query entities |
| `SemanticDense` | `embed(query)` → HNSW KNN top-N | cosine similarity |
| `SemanticBinary` | Query `binary_code` → Hamming scan top-N | 1 − hamming/128 |
| `Temporal` | Query/session temporal keys → temporal index | shared-bucket weighting |
| `TopicCluster` | Query topic keys → topic index union | Jaccard (IDF-weighted) |
| `Goal` | Query goal keys → goal index | Jaccard |
| `Event` | Query event keys → event index | Jaccard |
| `Causal` | Query causal keys / connectives → causal edge index | hit ratio |
| `RecentActivation` | Neighbors of recent memory IDs + recent high-frequency activations | normalized frequency |
| `GraphSpreading` | Not a seed channel — it is the spreading step itself, recorded for attribution | — |

Each channel returns its own top-N (default `seed_per_channel = 20`). Channels with no relevant input (e.g. a query with no goals) simply return nothing — they do not error. `RetrievalMode::Fast` runs only BM25 + Entity + SemanticDense.

### IDF weighting for Topic and Entity channels

Topic and Entity channels match inverted indices where every tag is treated equally — a tag shared by 30 of 30 memories carries the same weight as one shared by 2. An IDF correction fixes this:

```
tag_score(tag) = 1 / log(1 + doc_freq(tag))
topic_score = Σ matched_tags tag_score(tag)
```

Common tags (high `doc_freq`) are down-weighted; rare tags keep their score. The same correction applies to the Entity channel.

---

## 5. Seed fusion by Reciprocal Rank Fusion

A memory may be recalled as a seed by several channels. Rather than adding raw scores across channels (which would require making BM25's 0.7 and SemanticDense's 0.7 commensurable — they are not), HIPPMEM fuses by **rank**.

### Per-channel ranking

After all channels return their seeds, each channel is ranked internally by descending `raw_score` (0-indexed). Rank 0 is the channel's top seed.

### RRF formula

```
rrf_score(id) = Σ_c  w_c / (k + rank_c(id))      # k = rrf_k (default 1.0)
norm_rrf(id)  = rrf_score(id) / max_rrf          # normalized to [0, 1]
```

- `w_c` is the channel's **precision weight**, chosen by the channel's intrinsic mechanism — not by empirical tuning.
- `k > 0` enables RRF fusion (rank 0 contributes far more than rank 5). `k ≤ 0` degenerates to winner-take-all.
- Multi-channel consensus and per-channel precision are encoded simultaneously: a BM25 rank-0 seed (w = 1.0) contributes roughly 3.3× a Topic rank-0 seed (w = 0.3).

### Why only Topic and Temporal are down-weighted

Eight of the eleven seed channels keep `w_c = 1.0` because their internal scoring already encodes precision:

- BM25 carries IDF internally.
- Entity matches are named entities, not bag-of-words tokens.
- SemanticDense uses a real embedding model.
- Goal / Event / Causal / RecentActivation / SemanticBinary all have structurally meaningful scoring.

Only Topic and Temporal lack an intrinsic precision mechanism:

- **Topic** — single-token labels carry no IDF before the correction above; even after IDF, the bag-of-labels signal is coarser than BM25.
- **Temporal** — time-bucket overlap is the weakest possible signal; a shared day bucket says almost nothing about relevance.

Both get `w_c = 0.3`.

### Seed energy

The normalized RRF score becomes the query-match component of the seed's initial energy:

```
seed_energy = clamp(
    norm_rrf_score        * a_query_match     +
    context_match_score   * b_context_match   +
    freshness_score       * d_freshness       +
    reliability_score     * e_reliability,
    0, seed_energy_cap)                        # cap = 1.0
```

`importance` is intentionally excluded from seed energy — it already contributes through association scoring at write time. The context/freshness/reliability terms are wired in the configuration but currently evaluate to 0 in the implementation; they exist as extension points.

---

## 6. Spreading activation

Seeds carry their initial energy into a bounded graph traversal that propagates energy across association links.

### Propagation formula

```
propagated_energy = source_energy
                  * link.strength
                  * link.confidence
                  * decay_factor ^ hop
                  * type_modifier(link.link_type)
```

`decay_factor` (default 0.55) is raised to the hop count, so each successive hop loses roughly half the energy. `min_propagation_energy` (default 0.05) prunes any path below the threshold.

### Type modifiers

Each link type modulates propagation. The modifier encodes whether a link is a strong factual path or a weak associative hint.

| LinkType | type_modifier | Direction rule |
|----------|--------------|----------------|
| `Causal` | 1.30 | Forward-preferring |
| `Correction` | 1.20 | Boosts the corrector |
| `Supersedes` | 1.15 | Boosts the new |
| `CoActivation` | 1.05 | Bidirectional (habitual association) |
| `SameGoal` | 1.10 | Bidirectional |
| `SameEvent` | 1.10 | Bidirectional |
| `EntityOverlap` | 1.00 | Bidirectional |
| `Elaboration` | 1.00 | Forward |
| `SemanticSimilar` | 0.90 | Bidirectional |
| `TopicRelated` | 0.85 | Bidirectional |
| `EmotionalResonance` | 0.70 | Bidirectional |
| `TemporalAdjacent` | 0.60 | Bidirectional, fast-decaying |
| `Contradiction` | 0.50 | Signal only, not a fact path |
| `Deprecated` | 0.40 | Suppresses the old |

`Contradiction` edges are deliberately weak: they flag risk but do not spread factual energy.

### Traversal control

```
visited: Set<MemoryId>
energy:  Map<MemoryId, f32>     # multi-path energy merge
frontier = seeds (with initial_energy)
for hop in 1..=max_hops:
    next = []
    for node in frontier:
        for link in top_fanout(node.out_links):    # top fan_out_default per edge type
            e = propagated_energy(node, link)
            if e < min_propagation_energy: continue
            if link.target in current path: continue    # cycle breaking
            energy[target] = merge(energy[target], e)
            record ActivationStep
            next.push(target with energy[target])
    frontier = dedup(next)
```

**Cycle breaking** uses path-state: a node already on the current path is not re-expanded. A node reached by multiple paths is **merged**, not re-expanded.

**Energy merge function:**

```
merge(existing, new) = max(existing, new) + merge_secondary_weight * min(existing, new)
```

The `merge_secondary_weight` (default 0.30) gives a small bonus when multiple independent paths converge on the same memory, encoding "multi-path corroboration" without letting the secondary path dominate.

**Stopping conditions:** hop count reaches `max_hops` (Fast = 1, Balanced = 2, Deep/Diagnostic = 3), the frontier is empty, or every candidate is below `min_propagation_energy`.

**Fan-out:** at each node, edges are grouped by type and the top `fan_out_default` (default 6) by `strength * confidence` are expanded. This bounds the branching factor.

---

## 7. Reranking and final scoring

After spreading completes, candidates are ranked.

```
base = energy[m]
if mode enables rerank and len(candidates) <= rerank_top_n:    # default 50
    final_score = 0.5 * normalize(base) + 0.5 * normalize(rerank_score)
else:
    final_score = normalize(base)
return top_k by final_score desc
```

The reranker (when enabled) is a separate model trait — see [Model Backends](model-backends.md). `Fast` mode skips reranking entirely; `Balanced` uses a light reranker; `Deep` and `Diagnostic` always rerank.

For each result, the engine attaches:

- `matched_dimensions` — which match dimensions contributed;
- `activation_trace` — the `ActivationStep` sequence recorded during spreading;
- `warnings` — see below.

---

## 8. Warnings and risk signals

After scoring, each result is checked for risk conditions:

- A `Correction` or `Supersedes` in-edge → `HasCorrection` / `Superseded`.
- A `Contradiction` edge → `HasContradiction`.
- `lifecycle == Deprecated` or `Negated` → `Deprecated`.
- `understanding.confidence < low_conf_threshold` (default 0.35) → `LowConfidence`.
- `freshness_score < stale_threshold` (default 0.20) → `StaleFreshness`.

Memories flagged with `Contradiction` are never ranked as high-confidence facts — the `type_modifier` of 0.50 already suppressed them during spreading, and the warning makes the suppression visible to the caller.

---

## 9. Hebbian consolidation

Hebbian learning strengthens links that turn out to be useful. It triggers when a retrieval result is adopted by the agent or confirmed by the user (reported via the feedback API).

```
usage_signal ∈ { referenced = 0.5, user_confirmed = 1.0, task_success = 0.8 }   # stackable, capped at 1.0
for each co-activated and adopted pair (A, B):
    if edge e(A, B) exists:
        e.strength = min(strength_max, e.strength + hebbian_learning_rate * usage_signal)
        e.activation_count += 1
        e.last_activated_at = now
    elif no edge and A.co_activations[B].count >= coactivation_create_threshold:   # default 3
        create CoActivation edge, strength = init_strength_base
also update each memory's ActivationState.usage_score (sliding accumulation, capped at 1.0)
```

The co-activation counter is the authoritative source — Hebbian reads `ActivationState.co_activations` rather than re-deriving it. The `co_activation` list is bounded (default 16 entries); the oldest entry is evicted when full, which prevents unbounded growth. `strength` is capped at `strength_max` (1.0), so learning can never run away.

---

## 10. Decay

Decay runs on each background consolidation cycle and weakens edges that are not being used.

```
for each edge:
    if link_type ∈ protected_set:
        skip strong decay (very slow only)
    else:
        e.strength *= decay_per_cycle        # default 0.97
    if e.strength < min_retained_strength    # default 0.12
       and (now - last_activated_at) > stale_unactivated_ms   # default 30 days
       and no retention reason:
        archive or delete the edge
```

**Protected set** (edges that decay very slowly and are never casually deleted): `Causal`, `Correction`, `Contradiction`, `Supersedes`, long-lived `Preference`-source edges, and compliance/safety-related edges (marked by `ContentType` or provenance).

Different edge types use different thresholds during governance — the protected set is treated strictly.

---

## 11. Compaction

Compaction is the governance pass that keeps the graph healthy. It triggers per node when any of these conditions holds:

- weak edges exceed `weak_degree_limit` (default 32);
- total out-degree exceeds `node_degree_limit` (default 64);
- an edge has `strength < min_retained_strength` and has been inactive past `stale_unactivated_ms`;
- an observation edge has exceeded `observation_window_ms` without activation;
- an edge type is flagged by evaluation as consistently noisy.

Governance actions include weak-edge decay, edge archival, candidate-edge deletion, duplicate-edge merging (see edge deduplication), and observation-edge promotion. Edge type is always distinguished — the protected set is handled strictly.

---

## 12. Summarization and merging

When a topic cluster or entity accumulates more than `summary_trigger_count` (default 12) similar low-importance memories, the engine compresses them:

```
trigger: similar low-layer memories under a topic/entity > summary_trigger_count
S = Summarizer.summarize(sources)
create new summary MemoryUnit (content_type = Reflection, generated_by = Consolidation)
S.covers → each source memory's lifecycle = Compressed { into: S.id }
sources remain in memory_log (traceable); retrieval returns the summary by default, with drill-down to originals
```

The summary must not lose the evidence chain — `covers` links back to every source. If the summary's confidence is below threshold, the originals are not replaced.

---

## 13. Determinism and testing

Every algorithm has deterministic tests with fixed inputs and fixed outputs:

- **Association scoring:** construct known dimension hits, assert the score lands in the expected band, assert the multi-dimension bonus fires, assert the single-semantic penalty fires.
- **Spreading:** build a small graph, assert energy values, cycle breaking, fan-out pruning, merge behavior, and stopping conditions.
- **Hebbian / decay:** assert strength changes monotonically, respects caps and floors, and that the protected set is never deleted.

All tests use an injected `Clock` with fixed time and an injected RNG source, so the entire pipeline is reproducible. This is what lets the deterministic fallback backend produce non-trivial, testable results offline.

---

## Further reading

- [Data Model](data-model.md) — the Rust types these algorithms operate on.
- [Model Backends](model-backends.md) — the `Embedder`, `Extractor`, `Reranker`, and `Summarizer` traits that feed the pipeline.
- [Configuration Reference](../configuration.md) — every `AlgoParams` field with tuning advice.
