//! Engine::retrieve — retrieval API assembly.
//!
//! Corresponds to 05#retrieve, 09 §4.2. Wires seed recall→energy→spreading→rerank→warnings→explain.

use crate::{Engine, EngineResult, RetrieveInput, RetrieveOutput};
use hippmem_core::hash::stable_hash64;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{ActivationStep, RecallChannel, RetrievalResult};
use hippmem_core::model::unit::{GeneratedBy, MemoryLifecycle, MemoryUnit};
use hippmem_core::time::Clock;
use hippmem_model::lang::active_locales;
use hippmem_retrieval::explain::deduce_dimensions;
use hippmem_retrieval::seeds::{multi_channel_seeds, rrf_fuse};
use hippmem_retrieval::spreading::spread_multi_hop_fused;
use hippmem_retrieval::warnings::check_warnings;
use hippmem_store::activation_log::ActivationLogger;
use hippmem_store::kv::InvertedIndex;
use hippmem_store::semantic::vector_index::BinaryIndex;
use hippmem_store::semantic::vector_index::VectorIndex;
use std::collections::HashMap;

impl Engine {
    /// Retrieves memories: multi-channel seeds→activation energy→spreading→rerank→warnings.
    pub fn retrieve(&self, input: RetrieveInput) -> EngineResult<RetrieveOutput> {
        let start = std::time::Instant::now();
        let params = self.params.read();

        // 1. Lightweight understanding of the query (extract entities/topics for index lookup)
        let extractor = self.extractor.as_ref();
        let query_content = hippmem_core::model::unit::MemoryContent {
            raw: input.query.clone(),
            summary: None,
            normalized: None,
            language: hippmem_core::model::unit::Language::Zh,
            content_type: hippmem_core::model::enums::ContentType::UserStatement,
        };
        let understanding = extractor
            .extract_immediate_sync(&query_content)
            .unwrap_or_else(|_| hippmem_model::traits::ImmediateExtraction {
                entities: vec![],
                topics: vec![],
                explicit_causals: vec![],
                language: hippmem_core::model::unit::Language::Zh,
                content_type: None,
                importance: hippmem_core::score::UnitScore::new(0.0),
            });

        // 2. Multi-channel seed recall: query candidate IDs from the store index
        let inverted = InvertedIndex::new(self.store.db_arc());

        // 2a. Entity: from query entities → entity_index
        // entity-coverage-query-boost (0.4.2): a memory covering more query
        // entities gets a higher per-hit score (0.2 / 0.35 / 0.5 by covered
        // count) — for multi-entity queries ("what is the relationship between
        // X and Y") the answer must involve both entities, so full coverage
        // should rank first in this channel. Count first (seeds are deduped by
        // (id, channel) downstream, so coverage must be aggregated here).
        let mut entity_hits: Vec<(MemoryId, f32)> = {
            let mut covered: std::collections::HashMap<MemoryId, u32> =
                std::collections::HashMap::new();
            for em in &understanding.entities {
                let key = hippmem_core::hash::stable_hash64(&em.canonical);
                if let Ok(ids) = inverted.get_entity(&key) {
                    for id in ids {
                        *covered.entry(MemoryId(id)).or_default() += 1;
                    }
                }
            }
            covered
                .into_iter()
                .map(|(id, k)| (id, entity_coverage_score(k)))
                .collect()
        };
        // Determinism: HashMap iteration is randomized; sort by score desc
        // (id as tie-break) so same-tier hits resolve to a fixed channel rank.
        entity_hits.sort_by(|(a_id, a_s), (b_id, b_s)| {
            b_s.partial_cmp(a_s)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a_id.cmp(b_id))
        });

        // 2b. Topic: from query topics → topic_index
        let topic_hits: Vec<(MemoryId, f32)> = understanding
            .topics
            .iter()
            .filter_map(|t| {
                let key = hippmem_core::hash::stable_hash64(&t.label);
                inverted.get_topic(&key).ok().map(|ids| {
                    ids.into_iter()
                        .map(|id| (MemoryId(id), 0.15f32))
                        .collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect();

        // 2c. Temporal: query-time-aware retrieval (proposal
        // query-time-aware-retrieval, confirmed 2026-08-27) — when the
        // query carries a temporal expression ("3月5日", "上周"), target its
        // day buckets; otherwise the current-time hour/day/week buckets.
        let now = hippmem_core::time::SystemClock.now();
        let temporal_keys = match hippmem_model::time_query::parse_temporal_query(&input.query, now)
        {
            Some(spec) => temporal_keys_for_spec(spec),
            None => temporal_bucket_keys(now),
        };
        let mut temporal_hit_ids = std::collections::HashSet::new();
        for tk in &temporal_keys {
            if let Ok(ids) = inverted.get_temporal(tk) {
                for id in ids {
                    temporal_hit_ids.insert(MemoryId(id));
                }
            }
        }
        // Determinism: HashSet iteration order is randomized; sort by id so tied
        // scores within the channel resolve to a fixed rank (RRF depends on rank).
        let mut temporal_hit_vec: Vec<MemoryId> = temporal_hit_ids.into_iter().collect();
        temporal_hit_vec.sort();
        let temporal_hits: Vec<(MemoryId, bool)> =
            temporal_hit_vec.into_iter().map(|id| (id, true)).collect();

        // 2d. BM25: Tantivy fulltext search (03 §4.5), score normalized to [0,1] via tanh
        let mut bm25_hits: Vec<(MemoryId, f32)> = self
            .fulltext_index
            .lock()
            .search(&input.query, params.seed_per_channel as usize)
            .unwrap_or_default()
            .into_iter()
            .map(|(id, score)| {
                let norm = (score / params.bm25_norm_factor).tanh();
                (MemoryId(id), norm)
            })
            .collect();
        // Determinism: Tantivy's top-k order is unspecified for tied scores;
        // sort by score descending (id as tie-break) so channel ranks are stable.
        bm25_hits.sort_by(|(a_id, a_s), (b_id, b_s)| {
            b_s.partial_cmp(a_s)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a_id.cmp(b_id))
        });

        // 2e. SemanticDense: dense vector HNSW/FlatVectorIndex recall (03 §4.5)
        let semantic_hits: Vec<(MemoryId, f32)> = {
            let query_texts = vec![input.query.clone()];
            self.embedder
                .embed_sync(&query_texts)
                .ok()
                .and_then(|vectors| vectors.first().cloned())
                .map(|query_vec| {
                    let idx = self.dense_vector_index.lock();
                    idx.search(&query_vec, params.seed_per_channel as usize)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(id, l2_dist)| {
                            // L2 distance → cosine similarity: 1/(1+l2_dist), distance 0 → similarity 1
                            let cos_sim = 1.0 / (1.0 + l2_dist);
                            (MemoryId(id), cos_sim)
                        })
                        .filter(|(_, sim)| *sim > 0.0)
                        .collect()
                })
                .unwrap_or_default()
        };

        // 2f. SemanticBinary: binary_code Hamming distance recall (03 §4.5)
        let binary_hits: Vec<(MemoryId, f32)> = {
            let query_bc = query_binary_code(&input.query);
            let idx = self.binary_code_index.lock();
            idx.search(&query_bc, params.seed_per_channel as usize)
                .unwrap_or_default()
                .into_iter()
                .map(|(id, hamming)| {
                    let sim = 1.0 - (hamming as f32 / 128.0);
                    (MemoryId(id), sim.max(0.0))
                })
                .filter(|(_, sim)| *sim > 0.0)
                .collect()
        };

        // 2g. Goal: from query goal keywords → goal_index (03 §4.5)
        let query_goals = extract_query_goals(&input.query);
        let goal_hits: Vec<(MemoryId, usize)> = query_goals
            .iter()
            .filter_map(|goal| {
                let key = stable_hash64(goal);
                inverted.get_goal(&key).ok().map(|ids| {
                    ids.into_iter()
                        .map(|id| (MemoryId(id), 1))
                        .collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect();

        // 2h. Event: from query event keywords → event_index (03 §4.5)
        let query_events = extract_query_events(&input.query);
        let event_hits: Vec<(MemoryId, usize)> = query_events
            .iter()
            .filter_map(|event| {
                let key = stable_hash64(event);
                inverted.get_event(&key).ok().map(|ids| {
                    ids.into_iter()
                        .map(|id| (MemoryId(id), 1))
                        .collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect();

        // 2i. Causal: from query explicit causals → causal_index (03 §4.5)
        let causal_hits: Vec<(MemoryId, usize)> = understanding
            .explicit_causals
            .iter()
            .filter_map(|c| {
                let causal_str = format!("{} -> {}", c.cause, c.effect);
                let key = stable_hash64(&causal_str);
                inverted.get_causal(&key).ok().map(|ids| {
                    ids.into_iter()
                        .map(|id| (MemoryId(id), 1))
                        .collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect();

        // 2j. RecentActivation: recent_memory_ids graph neighbors (03 §4.5).
        //     Explicit caller context is a reliable seed source (+0.3 direct,
        //     +0.15 per graph neighbor). Confirmation frequency no longer seeds
        //     here — it is applied as a candidate-set tie-break after rerank
        //     (recency-candidate-correction proposal, 0.4.1).
        let recent_hits: Vec<(MemoryId, f32)> = {
            let mut recent_map: HashMap<MemoryId, f32> = HashMap::new();

            // Take directly from recent_memory_ids (each +0.3 base score)
            for mid in &input.context.recent_memory_ids {
                recent_map
                    .entry(*mid)
                    .and_modify(|s| *s = (*s + 0.3).min(1.0))
                    .or_insert(0.3);
            }

            // Supplement with graph neighbors of recent_memory_ids (neighbor +0.15)
            let graph = hippmem_store::graph::GraphStore::new(self.store.db_arc());
            for mid in &input.context.recent_memory_ids {
                if let Ok(links) = graph.get_outgoing(mid) {
                    for link in links.iter().take(8) {
                        recent_map
                            .entry(link.target_id)
                            .and_modify(|s| *s = (*s + 0.15).min(1.0))
                            .or_insert(0.15);
                    }
                }
            }

            let mut hits: Vec<(MemoryId, f32)> = recent_map.into_iter().collect();
            // Determinism: stable sort keeps tied scores in iteration order; sort by
            // id first so tied scores resolve to a fixed rank (RRF depends on rank).
            hits.sort_by_key(|(id, _)| *id);
            hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            hits.truncate(params.seed_per_channel as usize);
            hits
        };

        // 2j2. (removed in 0.4.3, memory-learning-mechanism) The global
        //      confirmation-frequency correction (recency-candidate-correction,
        //      0.4.1) learned "memory heat" without binding it to the query
        //      context: a hot memory lifted itself in every candidate set it
        //      happened to enter, could not accumulate across rounds, and
        //      diluted others via the max-frequency denominator. It is replaced
        //      by the context-answer correction (step 7d): confirmation binds
        //      a memory to the query's entity/topic fingerprint, and only
        //      queries whose fingerprint intersects the recorded links lift it.
        //      The context strengths are computed lazily inside step 7d (after
        //      rerank), so no per-query work is done here.

        let seed_result = multi_channel_seeds(
            &input.query,
            &entity_hits,
            &temporal_hits,
            &semantic_hits,
            &topic_hits,
            &bm25_hits,
            &binary_hits,
            &goal_hits,
            &event_hits,
            &causal_hits,
            &recent_hits,
            params.seed_per_channel as usize,
        );

        // 3. RRF rank fusion (V9): multi-channel seeds → fuse into a single score per MemoryId
        let mut fused_scores: HashMap<MemoryId, (f32, RecallChannel)> =
            if seed_result.seeds.is_empty() {
                // Fallback: no channel hits; take a few memories as RecentActivation seeds
                // F3/B1 (0.4.0): exclude compressed sources AND summaries.
                let fallback = load_limited_units(self.store.db_arc(), 50)
                    .into_iter()
                    .filter(|u| !is_retrieval_seed_excluded(u))
                    .collect::<Vec<_>>();
                fallback
                    .into_iter()
                    .map(|u| (u.id, (0.3_f32, RecallChannel::RecentActivation)))
                    .collect()
            } else {
                rrf_fuse(&seed_result.seeds, &params)
            };

        // 4. Load on demand: seed units + seed outgoing edges + neighbor prefetch (supports 2-hop)
        let mut unit_map: HashMap<MemoryId, MemoryUnit> = HashMap::new();
        for unit in load_units_by_ids(
            self.store.db_arc(),
            &fused_scores.keys().cloned().collect::<Vec<_>>(),
        ) {
            unit_map.insert(unit.id, unit);
        }
        // F3/B1 (0.4.0): exclude compressed sources and summaries from the seed
        // set, not just from the final results (7c). Compressed sources must not
        // start a traversal (they would inject energy into their summary);
        // summaries must not hit the direct channels at all (they crowd out
        // concrete memories). Consistent with the fallback path above.
        fused_scores.retain(|id, _| !unit_map.get(id).is_some_and(is_retrieval_seed_excluded));
        let seed_ids: Vec<MemoryId> = fused_scores.keys().cloned().collect();

        // 4a. Build importance map from the loaded seed units
        let importance_map: HashMap<MemoryId, f32> = unit_map
            .iter()
            .map(|(id, unit)| (*id, unit.understanding.importance.value()))
            .collect();
        // 4a2. usage_map：feedback 驱动的 usage_score（0.5 = 中性）
        let usage_map: HashMap<MemoryId, f32> = unit_map
            .iter()
            .map(|(id, unit)| (*id, unit.activation.usage_score.value()))
            .collect();

        let graph = hippmem_store::graph::GraphStore::new(self.store.db_arc());
        let mut links_map: HashMap<MemoryId, Vec<hippmem_core::model::links::AssociationLink>> =
            HashMap::new();

        // Round 1: seed outgoing edges
        for sid in &seed_ids {
            if let Ok(links) = graph.get_outgoing(sid) {
                links_map.insert(*sid, links);
            }
        }

        // Round 2: prefetch outgoing edges of direct neighbors (GraphStore), and load their MemoryUnit (for rerank)
        let neighbor_ids: Vec<MemoryId> = links_map
            .values()
            .flatten()
            .map(|l| l.target_id)
            .filter(|tid| !links_map.contains_key(tid))
            .collect();
        // Load neighbor units first: their lifecycle decides whether they may expand.
        for unit in load_units_by_ids(self.store.db_arc(), &neighbor_ids) {
            unit_map.entry(unit.id).or_insert(unit);
        }
        // F3 (0.3.1): a compressed neighbor is a dead end — load no outgoing edges
        // for it, so it cannot propagate energy onward (it is still reachable as a
        // result candidate and removed by the 7c filter).
        for nid in &neighbor_ids {
            // F3/B1: compressed sources and summaries are dead ends — load no
            // outgoing edges, so they cannot propagate energy onward.
            if unit_map.get(nid).is_some_and(is_retrieval_seed_excluded) {
                continue;
            }
            if let Ok(links) = graph.get_outgoing(nid) {
                links_map.insert(*nid, links);
            }
        }

        // 5. Spreading activation (P2 回归：max_hops 必须透传；merged_count 如实报告)
        let (activated, merged_count) = spread_multi_hop_fused(
            &fused_scores,
            &links_map,
            &params,
            &importance_map,
            &usage_map,
            input.max_hops.map(|h| h as u32),
        );
        let max_k = input.top_k.min(activated.len());

        // 6. Load additional nodes discovered by spreading (for rerank)
        let extra_ids: Vec<MemoryId> = activated
            .iter()
            .map(|(id, _, _)| *id)
            .filter(|id| !unit_map.contains_key(id))
            .collect();
        for unit in load_units_by_ids(self.store.db_arc(), &extra_ids) {
            unit_map.insert(unit.id, unit);
        }

        // 7. Rerank: requires the MemoryUnit of all activated nodes
        let loaded_units: Vec<MemoryUnit> = activated
            .iter()
            .filter_map(|(id, _, _)| unit_map.get(id).cloned())
            .collect();
        let mut reranked = hippmem_retrieval::rerank::rerank_by_energy(&activated, &loaded_units);

        // 7b. Question-type aware boost: detect the question type of the query, and apply a moderate score boost to matching answer patterns.
        //     Compensates for the deterministic embedder's inability, under a bag-of-tokens mechanism, to capture the "why"↔"because" semantic relation.
        apply_question_aware_boost(&input.query, &mut reranked, &params);

        // 7c. 0.3.0: 过滤已压缩（Compressed）单元 —— 检索默认返回摘要，源记忆不直接命中（03 §8）
        reranked.retain(|(_, _, _, unit)| {
            !matches!(unit.lifecycle, MemoryLifecycle::Compressed { .. })
        });

        // 7d. Context-answer correction (memory-learning-mechanism, 0.4.3):
        //     confirmation binds a memory to the query's entity/topic
        //     fingerprint (written to QUERY_CONTEXT at retrieve time, applied
        //     in feedback). Here we lift a candidate only when the current
        //     query's fingerprint intersects the links recorded for it —
        //     exact set intersection, so the lift is scoped to similar queries
        //     and a hot memory borrows nothing in unrelated queries (no global
        //     heat). Multiplicative (score × (1 + α·strength)), bounded by
        //     LINK_STRENGTH_CAP; strengths accumulate across confirmations
        //     (multi-round learning). Applied after the compressed/summary
        //     filter so summaries never gain from it. Re-sort with the
        //     deterministic pattern (id first, then stable by energy).
        {
            use hippmem_store::context_links::{collect_link_strengths, QueryContext};
            let ctx = QueryContext {
                entity_hashes: understanding
                    .entities
                    .iter()
                    .map(|em| stable_hash64(&em.canonical))
                    .collect(),
                topic_hashes: understanding
                    .topics
                    .iter()
                    .map(|t| stable_hash64(&t.label))
                    .collect(),
            };
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            let context_strengths = collect_link_strengths(self.store.db_arc(), &ctx, now_ms);
            if !context_strengths.is_empty() {
                for (id, energy, _, _) in reranked.iter_mut() {
                    if let Some(s) = context_strengths.get(&id.0) {
                        *energy *= 1.0 + CONTEXT_BOOST_ALPHA * s;
                    }
                }
                reranked.sort_by_key(|(id, _, _, _)| *id);
                reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        // 7d2. Entity coverage correction (entity-coverage-query-boost, 0.4.2):
        //     for multi-entity queries (N ≥ 2), lift candidates that cover more
        //     of the query's entities — a multiplicative tie-break within the
        //     candidate set, same tier as the recency correction. Coverage is a
        //     query-side signal (|memory entities ∩ query entities| / N), so it
        //     never reaches outside the query's own relevance scope. Single-
        //     entity queries (N = 1) are untouched: coverage has no
        //     discrimination there. Re-sort with the deterministic pattern.
        let query_entity_count = understanding.entities.len();
        if query_entity_count >= 2 {
            let query_entities: std::collections::HashSet<&str> = understanding
                .entities
                .iter()
                .map(|em| em.canonical.as_str())
                .collect();
            let n = query_entity_count as f32;
            for (_, energy, _, unit) in reranked.iter_mut() {
                let covered = unit
                    .understanding
                    .entities
                    .iter()
                    .filter(|em| query_entities.contains(em.canonical.as_str()))
                    .count();
                if covered > 0 {
                    *energy *= 1.0 + ENTITY_COVERAGE_BETA * (covered as f32 / n);
                }
            }
            reranked.sort_by_key(|(id, _, _, _)| *id);
            reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // 8. Build results
        let results: Vec<RetrievalResult> = reranked
            .iter()
            .take(max_k)
            .map(|(_id, energy, trace, unit)| {
                let matched = deduce_dimensions(trace);
                let warns = check_warnings(unit, *energy);
                RetrievalResult {
                    memory: unit.clone(),
                    final_score: *energy,
                    activation_trace: trace.clone(),
                    matched_dimensions: matched,
                    warnings: warns,
                }
            })
            .collect();

        // 9. Channel contributions
        // F3 (0.3.1): report only the seeds actually used for spreading —
        // compressed sources were dropped from fused_scores and must not be
        // counted here either.
        let channel_contributions: Vec<(RecallChannel, u32)> = {
            let mut map: HashMap<RecallChannel, u32> = HashMap::new();
            for seed in seed_result
                .seeds
                .iter()
                .filter(|s| !unit_map.get(&s.id).is_some_and(is_retrieval_seed_excluded))
            {
                *map.entry(seed.channel).or_default() += 1;
            }
            map.into_iter().collect()
        };

        // 10. Record activation log (for the RecentActivation channel and Hebbian)
        //     and surface the retrieval_id to the caller for feedback.
        let retrieval_id = {
            let act_log = ActivationLogger::new(self.store.db_arc());
            // P1 回归：记录完整 u128 id（截断会导致 RecentActivation/Hebbian 指向幽灵 id）
            let used_ids: Vec<u128> = results.iter().map(|r| r.memory.id.0).collect();
            let now_ms =
                if let Ok(t) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    t.as_millis() as i64
                } else {
                    0
                };
            let _ = act_log.record(&hippmem_store::activation_log::ActivationRecord {
                retrieval_id: now_ms as u64,
                used_memory_ids: used_ids,
                signal: "retrieve".into(),
                recorded_at_ms: now_ms,
            });
            // 0.4.3 memory-learning-mechanism: record the query context
            // fingerprint so a later confirmation can bind its answers to
            // this query's entity/topic context (exact set intersection at
            // lift time — the query itself is never stored as a memory).
            {
                use hippmem_store::context_links::{write_query_context, QueryContext};
                let ctx = QueryContext {
                    entity_hashes: understanding
                        .entities
                        .iter()
                        .map(|em| stable_hash64(&em.canonical))
                        .collect(),
                    topic_hashes: understanding
                        .topics
                        .iter()
                        .map(|t| stable_hash64(&t.label))
                        .collect(),
                };
                let _ = write_query_context(self.store.db_arc(), now_ms as u64, &ctx);
            }
            // 0.4.3 memory-learning-mechanism: record the propagation paths of
            // every non-seed result (the guide memory that spread energy to
            // it), so a later confirmation can strengthen the "guide →
            // answer" edge — the bridge that let the query reach the answer
            // through the graph.
            {
                use hippmem_store::context_links::{write_retrieval_paths, RetrievalPath};
                let paths: Vec<RetrievalPath> = activated
                    .iter()
                    .filter_map(|(_, _, steps)| {
                        steps
                            .iter()
                            .find(|s| s.from.is_some())
                            .map(|s| RetrievalPath {
                                from: s.from.expect("checked is_some").0,
                                to: s.to.0,
                            })
                    })
                    .collect();
                let _ = write_retrieval_paths(self.store.db_arc(), now_ms as u64, &paths);
            }
            now_ms as u64
        };

        // 10b. E7 (0.4.0): update ActivationState of the returned memories —
        // retrieval_count/last_retrieved_at/co_activations (03 §6: "每次检索后
        // 由检索侧累加"). Bounded co-activation list (co_activation_keep);
        // usage_score is owned by feedback and not touched here.
        {
            let kv = hippmem_store::kv::KvStore::new(self.store.db_arc());
            let now_ts = hippmem_core::time::Timestamp::from_millis(retrieval_id as i64);
            let max_co = params.co_activation_keep as usize;
            for r in &results {
                let raw = kv
                    .get(&r.memory.id.0)
                    .map_err(|e| crate::EngineError::Store(e.to_string()))?;
                let Some(raw) = raw else { continue };
                let (mut unit, _): (MemoryUnit, _) =
                    bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                        .map_err(|e| crate::EngineError::Internal(e.to_string()))?;
                unit.activation.retrieval_count = unit.activation.retrieval_count.saturating_add(1);
                unit.activation.last_retrieved_at = Some(now_ts);
                // Co-activation with the other memories of this result set.
                let mut co: Vec<hippmem_core::model::links::CoActivationCount> =
                    unit.activation.co_activations.clone();
                for other in &results {
                    if other.memory.id == r.memory.id {
                        continue;
                    }
                    if let Some(existing) = co.iter_mut().find(|c| c.with == other.memory.id) {
                        existing.count = existing.count.saturating_add(1);
                        existing.last_at = now_ts;
                    } else {
                        co.push(hippmem_core::model::links::CoActivationCount {
                            with: other.memory.id,
                            count: 1,
                            last_at: now_ts,
                        });
                    }
                }
                // Bound the list, dropping the least recently co-activated.
                if co.len() > max_co {
                    co.sort_by_key(|c| std::cmp::Reverse(c.last_at));
                    co.truncate(max_co);
                }
                unit.activation.co_activations = co;
                let encoded = bincode::serde::encode_to_vec(&unit, bincode::config::standard())
                    .map_err(|e| crate::EngineError::Internal(e.to_string()))?;
                kv.put(r.memory.id.0, &encoded)
                    .map_err(|e| crate::EngineError::Store(e.to_string()))?;
            }
        }

        Ok(RetrieveOutput {
            retrieval_id,
            results,
            trace: crate::RetrievalTrace {
                seeds: seed_result
                    .seeds
                    .iter()
                    // F3/B1 (0.4.0): mirror the actual seed set used for spreading.
                    .filter(|s| !unit_map.get(&s.id).is_some_and(is_retrieval_seed_excluded))
                    .map(|s| crate::SeedRecord {
                        id: s.id,
                        channel: s.channel,
                        initial_energy: s.score,
                        rank_in_channel: s.rank_in_channel,
                    })
                    .collect(),
                steps: activated
                    .iter()
                    .flat_map(|(_, _, trace)| trace.clone())
                    .collect(),
                hops_used: activated
                    .iter()
                    .flat_map(|(_, _, trace)| trace.iter())
                    .map(|s| s.hop)
                    .max()
                    .unwrap_or(0),
                merged_count,
            },
            diagnostics: crate::RetrievalDiagnostics {
                channel_contributions,
                reranked: true,
                pruned_branches: 0,
                backend_used: crate::BackendUsage {
                    embedder: self.embedder.backend_id().to_string(),
                    reranker: Some("rule".into()),
                },
                latency_ms: start.elapsed().as_millis() as u32,
            },
        })
    }
}

// ── Helpers ──

/// Context-answer boost coefficient (memory-learning-mechanism, 0.4.3):
/// a candidate with context-link strength s scores ×(1 + this·s). With
/// CONFIRM_LINK_DELTA = 0.15 per confirmation and a strength cap of 1.0,
/// a memory confirmed ~6 times in the same context reaches s ≈ 0.9 →
/// ×(1 + 0.3·0.9) ≈ +27%, enough to overtake an initial gap of up to ~20%
/// while remaining bounded (×1.3 max). Calibrated against batch2 hard
/// questions; revisit with the forgetting-curve work (P3).
const CONTEXT_BOOST_ALPHA: f32 = 0.5;

/// Entity-coverage query boost (entity-coverage-query-boost proposal, 0.4.2):
/// rerank multiplier for multi-entity queries — a candidate covering k of the
/// query's N entities scores ×(1 + β·k/N), at most +20%. Multiplicative, like
/// the recency correction, so relative gaps between candidates are preserved;
/// bounded (≤ ×1.2), so the score-ceiling guarantee (§8 of the concepts docs)
/// is not weakened.
const ENTITY_COVERAGE_BETA: f32 = 0.2;

/// Per-hit entity channel score by how many query entities a memory covers
/// (2a). k=1 keeps the historical flat 0.2; each additional covered query
/// entity raises the hit score (cap 0.5), so a memory covering every query
/// entity ranks first in the entity channel. Seeds are deduped by
/// (id, channel) downstream, so this is a single seed per memory.
fn entity_coverage_score(covered_entities: u32) -> f32 {
    match covered_entities {
        0 | 1 => 0.2,
        2 => 0.35,
        _ => 0.5,
    }
}

// ── Question-type aware boost (§4.5) ──

/// Question type: detected from the query text, used to activate answer-pattern boosts.
#[derive(Debug, Clone, Copy, PartialEq)]
enum QuestionType {
    /// Why-type queries: expects causal/explanatory answers
    Why,
    /// How-type queries: expects process/method answers
    How,
    /// What-type queries: expects factual/enumeration answers
    What,
    /// Correction/change queries: expects Correction-type memories
    Correction,
    /// Preference queries: expects Preference-type memories
    Preference,
    /// No clear question type detected
    None,
}

/// Detects the question type from the query text using locale-parametrized patterns.
///
/// Patterns for each locale are tried in order (zh first, then en fallback).
/// Within each locale, priority is Correction > Preference > Why > How > What.
/// The first matching pattern wins.
fn detect_question_type(query: &str) -> QuestionType {
    let q = query.to_lowercase();

    // Special case: change_pair signals a change/correction in any locale
    for lang in active_locales() {
        if let Some((before, after)) = lang.change_pair {
            if q.contains(before) && q.contains(after) {
                return QuestionType::Correction;
            }
        }
    }

    // Try each locale's patterns. Priority order preserved from active_locales().
    // Chinese first (higher specificity for CJK queries), then English as a broad fallback.
    // Within each priority category, zh patterns are checked before en.
    for lang in active_locales() {
        for keyword in lang.q_correction {
            if q.contains(keyword) {
                return QuestionType::Correction;
            }
        }
    }
    for lang in active_locales() {
        for keyword in lang.q_preference {
            if q.contains(keyword) {
                return QuestionType::Preference;
            }
        }
    }
    for lang in active_locales() {
        for keyword in lang.q_why {
            if q.contains(keyword) {
                return QuestionType::Why;
            }
        }
    }
    for lang in active_locales() {
        for keyword in lang.q_how {
            if q.contains(keyword) {
                return QuestionType::How;
            }
        }
    }
    for lang in active_locales() {
        for keyword in lang.q_what {
            if q.contains(keyword) {
                return QuestionType::What;
            }
        }
    }
    QuestionType::None
}

/// Detects the strength of explanatory patterns in the text (range [0, 0.20]).
fn explanatory_pattern_score(text: &str) -> f32 {
    let mut score = 0.0f32;
    for lang in active_locales() {
        for (pattern, boost) in lang.explanatory {
            if text.contains(pattern) {
                score += boost;
            }
        }
    }
    score.min(0.20) // Hard cap, prevents boost from over-dominating ranking
}

/// Returns a per-ContentType boost map based on the detected query intent.
///
/// Core idea: embedding cannot distinguish "decision" from "correction of a decision",
/// nor "preference" from "identity description"; but ContentType is a strong signal fixed
/// at write time. By detecting intent keywords in the query, a moderate energy boost is
/// applied to memories of the matching ContentType, compensating for the granularity gap
/// of pure semantic channels.
///
/// Boost cap 0.12, ensures the boost only flips borderline cases (#2→#1) without dominating ranking.
fn content_type_boost(query: &str) -> Vec<(hippmem_core::model::unit::ContentType, f32)> {
    let qt = detect_question_type(query);
    let mut boosts = Vec::new();

    match qt {
        QuestionType::Correction => {
            // Correction queries: Correction memory +0.12; can pull it back even if embedding ranks it behind the decision
            boosts.push((hippmem_core::model::unit::ContentType::Correction, 0.12));
        }
        QuestionType::Preference => {
            // Preference queries: Preference memory +0.08, enough to distinguish "prefers PostgreSQL" from "the project uses redb"
            boosts.push((hippmem_core::model::unit::ContentType::Preference, 0.08));
            // Decisions are often preference-related (+0.04)
            boosts.push((hippmem_core::model::unit::ContentType::Decision, 0.04));
        }
        QuestionType::Why => {
            // Causal: Decision and TaskState often explain the reason
            boosts.push((hippmem_core::model::unit::ContentType::Decision, 0.08));
            boosts.push((hippmem_core::model::unit::ContentType::TaskState, 0.08));
        }
        QuestionType::How => {
            // Method: TaskState (contains process descriptions such as fix/resolve verbs)
            boosts.push((hippmem_core::model::unit::ContentType::TaskState, 0.08));
        }
        QuestionType::What => {
            // What-type ("what is") queries prefer project knowledge.
            // V9 precision weight (rrf_w_topic=0.3) lowers the Topic channel contribution; definition memories need moderate compensation.
            // Boost value 0.15: enough to flip adjacent weak differences, but not enough to let a RRF-bottom ProjectKnowledge
            // overtake a strongly-matching memory of another type (e.g. the correct Decision answer for a "what is the license" query).
            // The second stage also adds the precondition "query subject must appear in memory content" to further suppress false positives.
            boosts.push((
                hippmem_core::model::unit::ContentType::ProjectKnowledge,
                0.15,
            ));
        }
        QuestionType::None => {
            // No question type detected: no per-type boost, rely on semantic channels
        }
    }

    // Generic correction-keyword detection (even if the main intent is not Correction, give Correction a boost when correction words are present)
    if qt != QuestionType::Correction {
        let q = query.to_lowercase();
        let has_correction_signal = active_locales().iter().any(|lang| {
            lang.q_correction.iter().any(|kw| q.contains(kw))
                || lang
                    .change_pair
                    .is_some_and(|(b, a)| q.contains(b) && q.contains(a))
        });
        if has_correction_signal {
            boosts.push((hippmem_core::model::unit::ContentType::Correction, 0.10));
        }
    }

    boosts
}

/// Applies question-type aware boosts to the reranked candidate list.
///
/// Currently supports:
/// - Why queries → documents with explanatory markers receive an `explanatory_pattern_score` boost
/// - Correction queries → Correction ContentType receives a content-type boost
/// - Preference queries → Preference ContentType receives a content-type boost
/// - How/What queries → reserved extension points
///
/// After boosts, re-sorts by adjusted energy descending.
fn apply_question_aware_boost(
    query: &str,
    reranked: &mut [(MemoryId, f32, Vec<ActivationStep>, MemoryUnit)],
    params: &hippmem_core::config::AlgoParams,
) {
    let qt = detect_question_type(query);
    let ct_boosts = content_type_boost(query);
    let cap = params.seed_energy_cap;
    // Subject of the What query (used as the content-match precondition for the stage-2 PK boost)
    let what_subject: Option<String> = if qt == QuestionType::What {
        extract_subject_for_what_query(query)
    } else {
        None
    };

    // Stage 1: question-type logic boost
    match qt {
        QuestionType::Why => {
            for (_, energy, _, unit) in reranked.iter_mut() {
                let boost = explanatory_pattern_score(&unit.content.raw);
                if boost > 0.0 {
                    *energy = (*energy + boost).min(cap);
                }
            }
        }
        QuestionType::Correction
        | QuestionType::Preference
        | QuestionType::How
        | QuestionType::What
        | QuestionType::None => {
            // Content-type boost is applied uniformly in stage 2
        }
    }

    // Stage 2: ContentType-aware boost (applies to all question types)
    // For the What-query ProjectKnowledge boost, require the query subject to appear in the memory content,
    // to prevent a what-is-the-license query from pushing an unrelated project-definition memory to the top (false positive).
    if !ct_boosts.is_empty() {
        for (_, energy, _, unit) in reranked.iter_mut() {
            for (ct, boost) in &ct_boosts {
                if unit.content.content_type != *ct {
                    continue;
                }
                // What + ProjectKnowledge: subject-match precondition
                if qt == QuestionType::What
                    && *ct == hippmem_core::model::unit::ContentType::ProjectKnowledge
                {
                    if let Some(ref subject) = what_subject {
                        let content_lower = unit.content.raw.to_lowercase();
                        if !content_lower.contains(&subject.to_lowercase()) {
                            break; // Subject not in content; no boost
                        }
                    }
                }
                *energy = (*energy + boost).min(cap);
                break; // At most one type boost per memory
            }
        }
    }

    // Stage 3: rare-keyword overlap bonus (+0.04 per keyword per memory, cap +0.08)
    // Extracts high-information words from the query (English abbreviations / proper nouns),
    // and gives a small boost to memories containing them.
    // Used to distinguish a query mentioning a specific term (e.g. "OOM") → the OOM memory,
    // vs a query that merely describes fixing something without naming the term.
    let keywords = extract_discriminative_keywords(query);
    if !keywords.is_empty() {
        for (_, energy, _, unit) in reranked.iter_mut() {
            let mut kw_bonus = 0.0f32;
            let content_lower = unit.content.raw.to_lowercase();
            for kw in &keywords {
                if content_lower.contains(&kw.to_lowercase()) {
                    kw_bonus += 0.04;
                }
            }
            if kw_bonus > 0.0 {
                *energy = (*energy + kw_bonus.min(0.08)).min(cap);
            }
        }
    }

    // Stage 4: definition-pattern detection (a "what is X" query → prefer "X is ..." definitions)
    // When the query is a what-is-X form, detect whether results contain definition patterns
    // (subject followed by a copular/usage/based-on/adopts verb). Apply a moderate +0.05 boost
    // to matching memories; not enough to dominate ranking but enough to flip adjacent results.
    if qt == QuestionType::What {
        if let Some(ref subject) = extract_subject_for_what_query(query) {
            let subject_lower = subject.to_lowercase();
            for (_, energy, _, unit) in reranked.iter_mut() {
                let content_lower = unit.content.raw.to_lowercase();
                let has_definition = active_locales().iter().any(|lang| {
                    lang.definition_patterns
                        .iter()
                        .any(|pat| content_lower.contains(&format!("{} {pat}", subject_lower)))
                });
                if has_definition {
                    *energy = (*energy + 0.05).min(cap);
                }
            }
        }
    }

    // Re-sort by adjusted energy descending
    reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}

/// Extracts the subject X from a "what is X" query (locale-driven).
///
/// Uses locale-specific what-delimiters (e.g., "是什么" for zh, "what is" for en)
/// and possessive particles ("的" for zh, None for en).
/// For the "A's B is what" form, takes the last segment "B" as the subject
/// (stripping the qualifier "A's"), to avoid merging the qualifier into the subject
/// and breaking later content matching.
/// Returns None when no what-is pattern is detected or the subject is too short (< 2 chars).
fn extract_subject_for_what_query(query: &str) -> Option<String> {
    let q = query.to_lowercase();
    for lang in active_locales() {
        for delimiter in lang.what_delimiters {
            if let Some(pos) = q.find(delimiter) {
                let prefix = &q[..pos];
                let subject = if let Some(particle) = lang.possessive_particle {
                    // First split on the possessive marker and take the last segment
                    // (strip qualifier), then split on whitespace/question mark and take the last segment
                    prefix
                        .rsplit(particle)
                        .next()
                        .unwrap_or("")
                        .rsplit(|c: char| c.is_whitespace() || c == '？' || c == '?')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string()
                } else {
                    prefix
                        .rsplit(|c: char| c.is_whitespace() || c == '？' || c == '?')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string()
                };
                if subject.len() >= 2 {
                    return Some(subject);
                }
                return None;
            }
        }
    }
    None
}

/// Extracts high-information keywords (English abbreviations, technical terms, proper nouns) from the query.
///
/// Filters out common question words and stop words, keeping only discriminative tokens.
/// Returns a deduplicated keyword list (max 5).
///
/// Stop words are multilingual: Chinese (zh) function words and question particles
/// are filtered alongside English equivalents so that CJK queries yield meaningful keywords.
fn extract_discriminative_keywords(query: &str) -> Vec<String> {
    // Multilingual stop words: collected from all active locales
    let stop_words: Vec<&str> = active_locales()
        .iter()
        .flat_map(|lang| lang.stop_words.iter().copied())
        .collect();

    let mut keywords: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // 1. Extract English abbreviations/words (all-caps or camelCase, e.g. OOM/HNSW/BM25/redb/gRPC)
    for word in query.split(|c: char| !c.is_alphanumeric()) {
        let is_keyword = (word.len() >= 2 && word.chars().any(|c| c.is_uppercase()))
            || (word.chars().all(|c| c.is_ascii_alphabetic()) && word.len() >= 3);
        if is_keyword
            && !stop_words.contains(&word.to_lowercase().as_str())
            && seen.insert(word.to_string())
        {
            keywords.push(word.to_string());
        }
    }

    // 2. Extract Chinese keywords (>=2 chars, not stop words, not question words)
    for word in query
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation() || c == '？' || c == '?')
    {
        let trimmed = word.trim();
        if trimmed.chars().count() >= 2
            && trimmed.chars().all(|c| c as u32 > 0x2E80) // CJK range
            && !stop_words.contains(&trimmed)
            && seen.insert(trimmed.to_string())
        {
            keywords.push(trimmed.to_string());
        }
    }

    keywords.truncate(5); // At most 5 keywords
    keywords
}

/// Generates the 16 bytes of binary_code for the query text ([u64;2]→LE), isomorphic to write_api::build_semantic_signature.
fn query_binary_code(text: &str) -> [u8; 16] {
    let bc0 = stable_hash64(&format!("bc_0_{}", text));
    let bc1 = stable_hash64(&format!("bc_1_{}", text));
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&bc0.to_le_bytes());
    bytes[8..].copy_from_slice(&bc1.to_le_bytes());
    bytes
}

/// Temporal bucket keys for a parsed query spec:
/// - single day → [day-1, day, day+1] (local dates straddle the UTC day
///   boundary; neighbours absorb the offset, RRF weights demote noise);
/// - range → every day in the range (caller caps the enumeration).
fn temporal_keys_for_spec(spec: hippmem_model::time_query::TemporalQuerySpec) -> Vec<u32> {
    match spec {
        hippmem_model::time_query::TemporalQuerySpec::SingleDay { day } => {
            vec![(day - 1) as u32, day as u32, (day + 1) as u32]
        }
        hippmem_model::time_query::TemporalQuerySpec::Range { start_day, end_day } => {
            (start_day..=end_day).map(|d| d as u32).collect()
        }
    }
}

/// Generates temporal bucket keys (hour/day/week) for the current time, consistent with write time.
fn temporal_bucket_keys(ts: hippmem_core::time::Timestamp) -> Vec<u32> {
    let ms = ts.0;
    vec![
        (ms / 3_600_000) as u32,   // Hour bucket
        (ms / 86_400_000) as u32,  // Day bucket
        (ms / 604_800_000) as u32, // Week bucket
    ]
}

pub(crate) fn load_all_units(db: std::sync::Arc<redb::Database>) -> Vec<MemoryUnit> {
    use redb::ReadableDatabase;
    use redb::ReadableTable;
    let mut units = Vec::new();
    let read_txn = db.begin_read().expect("read transaction should succeed");
    let table = read_txn
        .open_table(hippmem_store::store::MEMORY_KV)
        .expect("memory_kv table should exist");
    let iter = table.iter().expect("iter should succeed");
    for entry in iter.flatten() {
        let (_key, value) = entry;
        if let Ok((unit, _)) = bincode::serde::decode_from_slice::<MemoryUnit, _>(
            value.value(),
            bincode::config::standard(),
        ) {
            units.push(unit);
        }
    }
    units
}

/// Batch-loads MemoryUnit entries from the MEMORY_KV table by an ID list (single transaction).
fn load_units_by_ids(db: std::sync::Arc<redb::Database>, ids: &[MemoryId]) -> Vec<MemoryUnit> {
    if ids.is_empty() {
        return vec![];
    }
    use redb::ReadableDatabase;
    let mut units = Vec::new();
    let read_txn = db.begin_read().expect("read transaction should succeed");
    let table = read_txn
        .open_table(hippmem_store::store::MEMORY_KV)
        .expect("memory_kv table should exist");
    for id in ids {
        if let Some(value) = table.get(id.0).expect("get should succeed") {
            if let Ok((unit, _)) = bincode::serde::decode_from_slice::<MemoryUnit, _>(
                value.value(),
                bincode::config::standard(),
            ) {
                units.push(unit);
            }
        }
    }
    units
}

/// Extracts goal keywords from the query text (deterministic rules, locale-driven).
fn extract_query_goals(text: &str) -> Vec<String> {
    let mut goals = Vec::new();
    for lang in active_locales() {
        for m in lang.goal_markers {
            if text.contains(m) {
                goals.push(format!("goal_marker:{m}"));
            }
        }
    }
    goals
}

/// Extracts event keywords from the query text (deterministic rules, locale-driven).
fn extract_query_events(text: &str) -> Vec<String> {
    let mut events = Vec::new();
    for lang in active_locales() {
        for m in lang.event_markers {
            if text.contains(m) {
                events.push(format!("event_marker:{m}"));
            }
        }
    }
    events
}

/// Loads at most `limit` memories from the MEMORY_KV table (for fallback, not a full scan).
/// A unit that must not act as a retrieval seed (B1, 0.4.0):
/// - `Compressed`: summary sources (F3) — a compressed source must not start
///   a traversal, otherwise it injects energy into its summary via the
///   Elaboration edges;
/// - `GeneratedBy::Consolidation`: summaries themselves (B1) — a summary text
///   is a concatenation of its sources, so it matches every query about them
///   through the semantic/BM25/entity channels and crowds out the concrete
///   memories (2026-08-11 test report P1-2). Summaries stay reachable via
///   graph edges once an upward-rollout channel exists (planned with B5).
fn is_retrieval_seed_excluded(unit: &MemoryUnit) -> bool {
    matches!(unit.lifecycle, MemoryLifecycle::Compressed { .. })
        || unit.provenance.generated_by == GeneratedBy::Consolidation
}

fn load_limited_units(db: std::sync::Arc<redb::Database>, limit: usize) -> Vec<MemoryUnit> {
    use redb::ReadableDatabase;
    use redb::ReadableTable;
    let mut units = Vec::new();
    let read_txn = db.begin_read().expect("read transaction should succeed");
    let table = read_txn
        .open_table(hippmem_store::store::MEMORY_KV)
        .expect("memory_kv table should exist");
    let iter = table.iter().expect("iter should succeed");
    for entry in iter.flatten().take(limit) {
        let (_key, value) = entry;
        if let Ok((unit, _)) = bincode::serde::decode_from_slice::<MemoryUnit, _>(
            value.value(),
            bincode::config::standard(),
        ) {
            units.push(unit);
        }
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_coverage_score_tiers() {
        // k=1 keeps the historical flat 0.2; each extra covered query entity
        // raises the hit score, capped at 0.5 (entity-coverage-query-boost).
        assert_eq!(entity_coverage_score(0), 0.2);
        assert_eq!(entity_coverage_score(1), 0.2);
        assert_eq!(entity_coverage_score(2), 0.35);
        assert_eq!(entity_coverage_score(3), 0.5);
        assert_eq!(entity_coverage_score(7), 0.5);
    }
}
