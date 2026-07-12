//! Engine::retrieve — retrieval API assembly.
//!
//! Corresponds to 05#retrieve, 09 §4.2. Wires seed recall→energy→spreading→rerank→warnings→explain.

use crate::{Engine, EngineResult, RetrieveInput, RetrieveOutput};
use hippmem_core::hash::stable_hash64;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{ActivationStep, RecallChannel, RetrievalResult};
use hippmem_core::model::unit::MemoryUnit;
use hippmem_core::time::Clock;
use hippmem_model::deterministic::extract::DeterministicExtractor;
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
        let params = self.params.read();

        // 1. Lightweight understanding of the query (extract entities/topics for index lookup)
        let extractor = DeterministicExtractor;
        let query_content = hippmem_core::model::unit::MemoryContent {
            raw: input.query.clone(),
            summary: None,
            normalized: None,
            language: hippmem_core::model::unit::Language::Zh,
            content_type: hippmem_core::model::enums::ContentType::UserStatement,
        };
        let understanding = extractor
            .extract_sync_immediate(&query_content)
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
        let entity_hits: Vec<(MemoryId, f32)> = understanding
            .entities
            .iter()
            .filter_map(|em| {
                let key = hippmem_core::hash::stable_hash64(&em.canonical);
                inverted.get_entity(&key).ok().map(|ids| {
                    ids.into_iter()
                        .map(|id| (MemoryId(id), 0.2f32))
                        .collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect();

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

        // 2c. Temporal: from current time bucket keys → temporal_index
        let now = hippmem_core::time::SystemClock.now();
        let temporal_keys = temporal_bucket_keys(now);
        let mut temporal_hit_ids = std::collections::HashSet::new();
        for tk in &temporal_keys {
            if let Ok(ids) = inverted.get_temporal(tk) {
                for id in ids {
                    temporal_hit_ids.insert(MemoryId(id));
                }
            }
        }
        let temporal_hits: Vec<(MemoryId, bool)> =
            temporal_hit_ids.into_iter().map(|id| (id, true)).collect();

        // 2d. BM25: Tantivy fulltext search (03 §4.5), score normalized to [0,1] via tanh
        let bm25_hits: Vec<(MemoryId, f32)> = self
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

        // 2j. RecentActivation: recent_memory_ids graph neighbors + activation_log (03 §4.5)
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

            // Take recently frequent memories from activation_log
            let act_log = ActivationLogger::new(self.store.db_arc());
            if let Ok(records) = act_log.read_all() {
                let mut freq: HashMap<MemoryId, u32> = HashMap::new();
                for rec in records.iter() {
                    for mid_u64 in &rec.used_memory_ids {
                        *freq.entry(MemoryId(*mid_u64 as u128)).or_default() += 1;
                    }
                }
                let max_freq = freq.values().max().copied().unwrap_or(1) as f32;
                for (mid, count) in freq {
                    let score = (count as f32 / max_freq) * 0.25;
                    recent_map
                        .entry(mid)
                        .and_modify(|s| *s = (*s + score).min(1.0))
                        .or_insert(score);
                }
            }

            let mut hits: Vec<(MemoryId, f32)> = recent_map.into_iter().collect();
            hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            hits.truncate(params.seed_per_channel as usize);
            hits
        };

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
        let fused_scores: HashMap<MemoryId, (f32, RecallChannel)> = if seed_result.seeds.is_empty()
        {
            // Fallback: no channel hits; take a few memories as RecentActivation seeds
            let fallback = load_limited_units(self.store.db_arc(), 50);
            fallback
                .into_iter()
                .map(|u| (u.id, (0.3_f32, RecallChannel::RecentActivation)))
                .collect()
        } else {
            rrf_fuse(&seed_result.seeds, &params)
        };

        // 4. Load on demand: seed units + seed outgoing edges + neighbor prefetch (supports 2-hop)
        let seed_ids: Vec<MemoryId> = fused_scores.keys().cloned().collect();
        let mut unit_map: HashMap<MemoryId, MemoryUnit> = HashMap::new();
        for unit in load_units_by_ids(self.store.db_arc(), &seed_ids) {
            unit_map.insert(unit.id, unit);
        }

        // 4a. Build importance map from the loaded seed units
        let importance_map: HashMap<MemoryId, f32> = unit_map
            .iter()
            .map(|(id, unit)| (*id, unit.understanding.importance.value()))
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
        for nid in &neighbor_ids {
            if let Ok(links) = graph.get_outgoing(nid) {
                links_map.insert(*nid, links);
            }
        }
        // Load neighbor units on demand as well
        for unit in load_units_by_ids(self.store.db_arc(), &neighbor_ids) {
            unit_map.entry(unit.id).or_insert(unit);
        }

        // 5. Spreading activation
        let activated = spread_multi_hop_fused(&fused_scores, &links_map, &params, &importance_map);
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
        let channel_contributions: Vec<(RecallChannel, u32)> = {
            let mut map: HashMap<RecallChannel, u32> = HashMap::new();
            for seed in &seed_result.seeds {
                *map.entry(seed.channel).or_default() += 1;
            }
            map.into_iter().collect()
        };

        // 10. Record activation log (for the RecentActivation channel and Hebbian)
        {
            let act_log = ActivationLogger::new(self.store.db_arc());
            let used_ids: Vec<u64> = results.iter().map(|r| r.memory.id.0 as u64).collect();
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
        }

        Ok(RetrieveOutput {
            results,
            trace: crate::RetrievalTrace {
                seeds: seed_result
                    .seeds
                    .iter()
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
                hops_used: 0,
                merged_count: 0,
            },
            diagnostics: crate::RetrievalDiagnostics {
                channel_contributions,
                reranked: true,
                pruned_branches: 0,
                backend_used: crate::BackendUsage {
                    embedder: self.embedder.backend_id().to_string(),
                    reranker: Some("rule".into()),
                },
                latency_ms: 0,
            },
        })
    }
}

// ── Helpers ──

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
