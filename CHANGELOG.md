# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.4.3] — 2026-08-21

### Changed
- Confirmation now learns *context-answer links* instead of global memory heat: a confirmed memory is bound to the query's entity/topic fingerprint, and only later queries whose fingerprint intersects the recorded links lift it — a hot memory borrows nothing in unrelated queries, and repeated confirmation accumulates (multi-round learning replaces the one-shot +15% cap)
- Confirming a non-seed answer (reached through the graph) strengthens the guide→answer edge that carried it, so propagation paths become learnable bridges
- Edge decay follows a forgetting curve: strength halves per half-life (1 day base), and each review (activation) doubles the half-life — spaced reviews consolidate edges into long-term memory while unused edges fade per the curve (replaces the fixed ×0.97/day decay)

[0.4.3]: https://github.com/hippmem/hippmem/releases/tag/v0.4.3



### Added
- Semantic index persistence: dense embedding vectors are now stored with each memory (DENSE_VECTORS table) and the SemanticDense / SemanticBinary indexes are rebuilt locally on open — previously the in-memory vector indexes were lost on restart, silently disabling semantic recall for every existing store (a non-empty store with no vectors is now reported via `semantic_index_degraded` on retrieve, and `consolidate("reindex")` rebuilds the dense index). Verified end-to-end: C scenario MRR 0.778 → 0.889, Hit@1 6/9 → 8/9 once semantic recall is live.

[0.4.2]: https://github.com/hippmem/hippmem/releases/tag/v0.4.2

## [0.4.1] — 2026-08-17

### Added
- Multi-entity queries ("what is the relationship between X and Y?") now prefer memories that cover more of the query's entities. The entity channel scores a memory by how many query entities it covers (0.2 / 0.35 / 0.5 for k = 1 / 2 / 3+), and after rerank a candidate covering k of the query's N ≥ 2 entities is multiplied by (1 + 0.2·k/N). The answer to such a query must involve both entities, so a full-coverage memory now overtakes single-entity decoys that only share one entity word with the query — the C-scenario case where an unrelated "Xiaoming is a student" memory ranked above the only correct answer. Single-entity queries are unaffected.

### Changed
- A `user_rejected` feedback with an empty `used_memory_ids` is no longer a result-set reject: it no longer lowers the usage scores of the returned memories, and it no longer removes them from the recent channel. Trap questions (queries with no answer in the store) trigger this signal by construction while retrieval must still return a list, so the 0.4.0 behavior permanently suppressed innocent memories that merely appeared in the rejected result set — even after they were explicitly confirmed. An empty rejection is now a retrieval-quality signal with no memory-side effects; it is still recorded in the activation log for audit. Targeted rejections (non-empty `used_memory_ids`) are unchanged: they still weaken the named memories' association edges during the next consolidation.
- Confirmed memories no longer surface in unrelated queries: confirmation frequency previously seeded retrieval in *every* query (a recently confirmed memory could crowd out more relevant ones in queries it had nothing to do with). It is now a small multiplicative tie-break inside the candidate set only — a confirmed memory that already matched a query scores up to 15% higher than it would have, so it can overtake memories within that relative distance but cannot enter the results of an unrelated query. `RetrieveContext.recent_memory_ids` remains the explicit working-memory interface (caller-declared recent context is still seeded directly).

### Fixed
- Chinese entity extraction no longer fuses an out-of-vocabulary proper name with the copula: jieba's HMM new-word discovery tagged the fused form "Li Hua shi" (person name + "is") as a single person name. The trailing copula is now stripped, so canonical-exact entity matching (entity index, multi-entity coverage boost) sees the same entity in the memory and the query — before the fix, a memory containing the fused form did not match the query entity "Li Hua", and full-coverage memories silently degraded to partial coverage.

[0.4.1]: https://github.com/hippmem/hippmem/releases/tag/v0.4.1

## [0.4.0] — 2026-08-12

### Added
- A `user_rejected` feedback with an empty `used_memory_ids` now lowers the usage score of the whole result set returned by that retrieval (a weaker signal than a targeted rejection) — trap questions and noisy stores can now teach the engine "none of these were right".

### Fixed
- When consolidation compresses a cluster, edges pointing at the compressed sources are now redirected to the summary: previously they stayed as "ghost edges" to units that could no longer participate, silently losing the association. The summary also becomes reachable through the graph as an upward view. Edge changes made during consolidation are now written to the graph store, so reinforcement and rejection actually affect retrieval.
- Retrieval is now deterministic: the same store state and query produce bit-identical scores. Previously, iteration-order randomness in channel ranking and multi-path energy merging made some scores vary between calls.
- Consolidation no longer duplicates co-activation edges: running consolidation repeatedly no longer grows the edge count without bound.
- Sources of a summary are now fully excluded from retrieval, including from the seed stage — a compressed source can no longer feed energy into its summary during search.

### Changed
- Summaries created by consolidation no longer hit the retrieval channels directly: their text is a concatenation of the source memories, so they used to rank above the concrete, correct memories in every related query. Summaries and their compressed sources are now excluded from retrieval seeds; the sources remain in storage and inspectable, and a drill-down path is planned.
- Retrieval no longer reinforces itself: merely running a query no longer boosts the memories it returns — only explicit feedback does. This stops frequently-returned memories from snowballing across queries.
- Confirmed memories no longer score higher in every query (the global usage weighting is removed): feedback now works through association edges, so it only lifts a memory in queries that reach it through its associations. Edge reinforcement now also scales with signal strength (a confirmation strengthens more than a reference).
- A targeted `user_rejected` feedback now weakens the rejected memory's association edges during the next consolidation, so it becomes harder to retrieve through its connections; an empty `used_memory_ids` triggers the result-set reject above. Rejections never strengthen any memory.

[0.4.0]: https://github.com/hippmem/hippmem/releases/tag/v0.4.0

## [0.3.0] — 2026-08-10

### Added
- Feedback now updates each memory's usage score: confirmations and references raise it, rejections lower it, bounded to [0, 1]. Memories with high usage are weighted higher in retrieval energy.
- New configuration options: `summary_similarity_threshold` (0.7), `summary_low_importance_threshold` (0.5), and `c_usage` (0.5). All defaults preserve existing retrieval behavior.
- Retrieval traces now report real hop counts and latencies; `max_hops` is honored by the graph traversal.

### Changed
- Summarization now groups similar low-importance memories into clusters before creating a summary (previously the whole store was treated as one candidate set). Only clusters above the configured size with mostly low-importance members trigger a summary.
- After a summary is created, its source memories are marked compressed and hidden from retrieval results; the summary itself is returned instead (sources remain in storage and inspectable).
- New summaries exclude memories already covered by an existing summary.
- The recent-activity recall channel and consolidation co-activation learning now ignore rejected feedback, so negative signals no longer strengthen memories.

### Fixed
- Feedback had no observable effect on retrieval: memory ids were truncated when recorded in the activation log, so reinforcement channels never matched the real memories. Ids are now stored in full.
- Summaries created by consolidation were not searchable (missing from all recall indexes) and were lost by reindex; they are now fully indexed and persisted to the memory log.
- `user_rejected` feedback previously boosted the rejected memories via the recent-activity channel; it now only lowers their usage score.

[0.3.0]: https://github.com/hippmem/hippmem/releases/tag/v0.3.0

## [0.2.1] — 2026-08-07

### Added
- `RetrieveOutput` now exposes a `retrieval_id` so callers can link an
  `Engine::feedback` record back to the retrieval that produced it.

[0.2.1]: https://github.com/hippmem/hippmem/releases/tag/v0.2.1

## [0.2.0] — 2026-07-30

### Changed
- **Embedder naming**: `Deterministic` → **`Hash`** (256d SimHash, offline default),
  `OpenAiCompatible` → **`Neural`** (API-based, higher semantic accuracy).
- **No more feature flags**: both embedders are always compiled. Choose at runtime
  via config (`provider = "hash" | "neural"`), env vars, or CLI (`--embedding-provider`).
- CLI default model: `text-embedding-v4` (DashScope) → `text-embedding-3-small` (OpenAI).
- All user documentation updated to the Hash/Neural terminology (EN + ZH).

### Fixed
- Removed dead-code `#[cfg(feature = "api-backends")]` guards; API-dependent tests
  now skip at runtime when `OPENAI_API_KEY` is missing.
- Mock HTTP test now performs a real round-trip against a local server.

### Added
- `hippmem-eval` bumped to 0.2.0 (aligned MINOR with the engine).

[0.2.0]: https://github.com/hippmem/hippmem/releases/tag/v0.2.0

## [0.1.0] — 2026-07-12

Initial public release of HIPPMEM.

### Added
- Native association memory engine with write-time association discovery.
- Multi-channel recall: BM25 + entity + semantic + temporal + topic + graph.
- Spreading activation retrieval with explanation traces.
- Hebbian consolidation, decay, and compaction.
- RRF (Reciprocal Rank Fusion) channel fusion.
- Deterministic fallback backend — fully offline, zero external API dependencies.
- gRPC server and CLI.
- Evaluation framework with 10 task types across 50+ test corpora.
- Tiered licensing: Apache 2.0 (infrastructure crates) / AGPL-3.0-only (algorithm + product crates).
- Commercial license option for proprietary use cases.

[0.1.0]: https://github.com/hippmem/hippmem/releases/tag/v0.1.0
