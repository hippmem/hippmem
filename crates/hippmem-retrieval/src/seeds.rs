//! Multi-channel seed recall + spreading activation + rerank (03 §4).

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{
    ActivationStep, MatchDimension, MemoryWarning, RecallChannel, RetrievalResult,
};
use hippmem_core::model::unit::MemoryUnit;

// ── Seed ──

#[derive(Debug, Clone)]
pub struct Seed {
    pub id: MemoryId,
    pub channel: RecallChannel,
    pub score: f32,
    /// V9: This seed's rank within its recall channel (0-indexed, score descending). None = not computed.
    pub rank_in_channel: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct SeedResult {
    pub seeds: Vec<Seed>,
    pub channel_scores: Vec<(RecallChannel, f32)>,
}

#[allow(clippy::too_many_arguments)]
pub fn multi_channel_seeds(
    _query_text: &str,
    entity_hits: &[(MemoryId, f32)],
    temporal_hits: &[(MemoryId, bool)],
    semantic_hits: &[(MemoryId, f32)],
    topic_hits: &[(MemoryId, f32)],
    bm25_hits: &[(MemoryId, f32)],
    binary_hits: &[(MemoryId, f32)],
    goal_hits: &[(MemoryId, usize)],
    event_hits: &[(MemoryId, usize)],
    causal_hits: &[(MemoryId, usize)],
    recent_hits: &[(MemoryId, f32)],
    per_channel_limit: usize,
) -> SeedResult {
    let mut seeds = Vec::new();
    let mut channel_scores = Vec::new();

    // BM25
    for (id, score) in bm25_hits.iter().take(per_channel_limit) {
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::Bm25,
            score: *score,
            rank_in_channel: None,
        });
    }
    if !bm25_hits.is_empty() {
        let max = bm25_hits.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        channel_scores.push((RecallChannel::Bm25, max));
    }

    // SemanticBinary
    for (id, sim) in binary_hits.iter().take(per_channel_limit) {
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::SemanticBinary,
            score: *sim,
            rank_in_channel: None,
        });
    }
    if !binary_hits.is_empty() {
        let max = binary_hits.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        channel_scores.push((RecallChannel::SemanticBinary, max));
    }

    // EntityInverted (V9: score already normalized per-hit, no IDF)
    for (id, score) in entity_hits.iter().take(per_channel_limit) {
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::EntityInverted,
            score: *score,
            rank_in_channel: None,
        });
    }
    if !entity_hits.is_empty() {
        channel_scores.push((RecallChannel::EntityInverted, entity_hits[0].1));
    }

    // TopicCluster (V9: flat 0.15 per match, w=0.3 in RRF, no IDF)
    for (id, score) in topic_hits.iter().take(per_channel_limit) {
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::TopicCluster,
            score: *score,
            rank_in_channel: None,
        });
    }
    if !topic_hits.is_empty() {
        let max_score = topic_hits.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        channel_scores.push((RecallChannel::TopicCluster, max_score));
    }

    // Temporal
    for (id, matched) in temporal_hits.iter().take(per_channel_limit) {
        if *matched {
            seeds.push(Seed {
                id: *id,
                channel: RecallChannel::Temporal,
                score: 0.3,
                rank_in_channel: None,
            });
        }
    }
    if temporal_hits.iter().any(|(_, m)| *m) {
        channel_scores.push((RecallChannel::Temporal, 0.3));
    }

    // Goal
    for (id, overlap) in goal_hits.iter().take(per_channel_limit) {
        let score = (*overlap as f32 * 0.15).min(1.0);
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::Goal,
            score,
            rank_in_channel: None,
        });
    }
    if !goal_hits.is_empty() {
        let max_score = goal_hits
            .iter()
            .map(|(_, o)| *o as f32 * 0.15)
            .fold(0.0f32, f32::max);
        channel_scores.push((RecallChannel::Goal, max_score.min(1.0)));
    }

    // Event
    for (id, overlap) in event_hits.iter().take(per_channel_limit) {
        let score = (*overlap as f32 * 0.15).min(1.0);
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::Event,
            score,
            rank_in_channel: None,
        });
    }
    if !event_hits.is_empty() {
        let max_score = event_hits
            .iter()
            .map(|(_, o)| *o as f32 * 0.15)
            .fold(0.0f32, f32::max);
        channel_scores.push((RecallChannel::Event, max_score.min(1.0)));
    }

    // Causal
    for (id, overlap) in causal_hits.iter().take(per_channel_limit) {
        let score = (*overlap as f32 * 0.15).min(1.0);
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::Causal,
            score,
            rank_in_channel: None,
        });
    }
    if !causal_hits.is_empty() {
        let max_score = causal_hits
            .iter()
            .map(|(_, o)| *o as f32 * 0.15)
            .fold(0.0f32, f32::max);
        channel_scores.push((RecallChannel::Causal, max_score.min(1.0)));
    }

    // RecentActivation
    for (id, score) in recent_hits.iter().take(per_channel_limit) {
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::RecentActivation,
            score: *score,
            rank_in_channel: None,
        });
    }
    if !recent_hits.is_empty() {
        let max = recent_hits.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        channel_scores.push((RecallChannel::RecentActivation, max));
    }

    // SemanticDense
    for (id, sim) in semantic_hits.iter().take(per_channel_limit) {
        seeds.push(Seed {
            id: *id,
            channel: RecallChannel::SemanticDense,
            score: *sim,
            rank_in_channel: None,
        });
    }
    if !semantic_hits.is_empty() {
        channel_scores.push((RecallChannel::SemanticDense, semantic_hits[0].1));
    }

    // ── V9: per-channel rank computation (for RRF fusion) ──
    {
        let mut channel_groups: std::collections::HashMap<RecallChannel, Vec<&mut Seed>> =
            std::collections::HashMap::new();
        for seed in seeds.iter_mut() {
            channel_groups.entry(seed.channel).or_default().push(seed);
        }
        for (_channel, group) in channel_groups.iter_mut() {
            group.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (rank, seed) in group.iter_mut().enumerate() {
                seed.rank_in_channel = Some(rank);
            }
        }
    }

    seeds.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    seeds.retain(|s| seen.insert((s.id, s.channel)));

    SeedResult {
        seeds,
        channel_scores,
    }
}

// ── RRF rank fusion (V9) ──

/// Perform RRF fusion on multi-channel seeds.
///
/// `k > 0`: RRF `fused(id) = Σ_c 1/(k + rank_c(id))`
/// `k ≤ 0`: winner-take-all (max score across channels)
///
/// Returns (fused score, top-contributing channel) for each MemoryId.
pub fn rrf_fuse(
    seeds: &[Seed],
    params: &hippmem_core::config::AlgoParams,
) -> std::collections::HashMap<MemoryId, (f32, RecallChannel)> {
    let mut fused: std::collections::HashMap<MemoryId, (f32, RecallChannel)> =
        std::collections::HashMap::new();
    let k = params.rrf_k;

    if k <= 0.0 {
        for seed in seeds {
            fused
                .entry(seed.id)
                .and_modify(|(score, ch)| {
                    if seed.score > *score {
                        *score = seed.score;
                        *ch = seed.channel;
                    }
                })
                .or_insert((seed.score, seed.channel));
        }
    } else {
        let mut best_contrib: std::collections::HashMap<MemoryId, (f32, RecallChannel)> =
            std::collections::HashMap::new();
        for seed in seeds {
            let rank = seed.rank_in_channel.unwrap_or(99) as f32;
            let w = params.rrf_channel_weight(seed.channel);
            let contrib = w / (k + rank);
            fused
                .entry(seed.id)
                .and_modify(|(score, _ch)| {
                    *score += contrib;
                })
                .or_insert((contrib, seed.channel));
            best_contrib
                .entry(seed.id)
                .and_modify(|(max_c, ch)| {
                    if contrib > *max_c {
                        *max_c = contrib;
                        *ch = seed.channel;
                    }
                })
                .or_insert((contrib, seed.channel));
        }
        for (id, (_max_c, best_ch)) in best_contrib.iter() {
            if let Some(entry) = fused.get_mut(id) {
                entry.1 = *best_ch;
            }
        }
    }

    fused
}

// ── Rerank ──

pub fn rerank_results(
    activated: &[(MemoryId, f32, Vec<ActivationStep>)],
    units: &[MemoryUnit],
    _query_text: &str,
) -> Vec<RetrievalResult> {
    activated
        .iter()
        .filter_map(|(id, energy, trace)| {
            let unit = units.iter().find(|u| u.id == *id)?;
            let matched_dimensions = deduce_dimensions(trace);
            let warnings = generate_warnings(unit, energy);
            Some(RetrievalResult {
                memory: unit.clone(),
                final_score: *energy,
                activation_trace: trace.clone(),
                matched_dimensions,
                warnings,
            })
        })
        .collect()
}

fn deduce_dimensions(trace: &[ActivationStep]) -> Vec<MatchDimension> {
    let mut dims = Vec::new();
    for step in trace {
        if let Some(edge) = step.via_link {
            match edge {
                hippmem_core::model::links::LinkType::EntityOverlap => {
                    dims.push(MatchDimension::Entity)
                }
                hippmem_core::model::links::LinkType::Causal => dims.push(MatchDimension::Causal),
                hippmem_core::model::links::LinkType::TopicRelated => {
                    dims.push(MatchDimension::Topic)
                }
                hippmem_core::model::links::LinkType::TemporalAdjacent => {
                    dims.push(MatchDimension::Temporal)
                }
                hippmem_core::model::links::LinkType::SemanticSimilar => {
                    dims.push(MatchDimension::Semantic)
                }
                hippmem_core::model::links::LinkType::SameGoal => dims.push(MatchDimension::Goal),
                hippmem_core::model::links::LinkType::SameEvent => dims.push(MatchDimension::Event),
                _ => {}
            }
        }
    }
    dims.dedup();
    dims
}

fn generate_warnings(unit: &MemoryUnit, energy: &f32) -> Vec<MemoryWarning> {
    let mut w = Vec::new();
    if *energy < 0.2 {
        w.push(MemoryWarning::LowConfidence);
    }
    if unit.lifecycle == hippmem_core::model::unit::MemoryLifecycle::Deprecated {
        w.push(MemoryWarning::Deprecated);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_multi_channel() {
        let e = vec![(MemoryId(1), 0.6f32)];
        let r = multi_channel_seeds("q", &e, &[], &[], &[], &[], &[], &[], &[], &[], &[], 20);
        assert!(!r.seeds.is_empty());
    }

    #[test]
    fn seeds_new_channels_goal_event_causal_recent() {
        let goal = vec![(MemoryId(10), 2)];
        let event = vec![(MemoryId(20), 1)];
        let causal = vec![(MemoryId(30), 3)];
        let recent = vec![(MemoryId(40), 0.5)];
        let r = multi_channel_seeds(
            "q",
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &goal,
            &event,
            &causal,
            &recent,
            20,
        );
        assert!(!r.seeds.is_empty());
        let channels: Vec<_> = r.seeds.iter().map(|s| s.channel).collect();
        assert!(channels.contains(&RecallChannel::Goal));
        assert!(channels.contains(&RecallChannel::Event));
        assert!(channels.contains(&RecallChannel::Causal));
        assert!(channels.contains(&RecallChannel::RecentActivation));
    }

    #[test]
    fn rrf_fuse_two_channels_beats_single() {
        let mut s1 = Seed {
            id: MemoryId(1),
            channel: RecallChannel::Bm25,
            score: 0.9,
            rank_in_channel: Some(0),
        };
        let s2 = Seed {
            id: MemoryId(1),
            channel: RecallChannel::SemanticDense,
            score: 0.7,
            rank_in_channel: Some(2),
        };
        let s3 = Seed {
            id: MemoryId(2),
            channel: RecallChannel::Bm25,
            score: 0.5,
            rank_in_channel: Some(1),
        };
        s1.rank_in_channel = Some(0);
        let seeds = vec![s1, s2, s3];
        let params = hippmem_core::config::AlgoParams {
            rrf_k: 1.0,
            ..Default::default()
        };
        let fused = rrf_fuse(&seeds, &params);
        let (score1, _) = fused.get(&MemoryId(1)).unwrap();
        let (score2, _) = fused.get(&MemoryId(2)).unwrap();
        assert!(
            *score1 > *score2,
            "dual-channel RRF ({:.3}) should be > single-channel ({:.3})",
            score1,
            score2
        );
    }

    #[test]
    fn rrf_fuse_k_zero_falls_back_to_wta() {
        let s1 = Seed {
            id: MemoryId(1),
            channel: RecallChannel::Bm25,
            score: 0.9,
            rank_in_channel: Some(0),
        };
        let s2 = Seed {
            id: MemoryId(1),
            channel: RecallChannel::SemanticDense,
            score: 0.3,
            rank_in_channel: Some(0),
        };
        let params = hippmem_core::config::AlgoParams {
            rrf_k: 0.0,
            ..Default::default()
        };
        let fused = rrf_fuse(&[s1, s2], &params);
        let (score, ch) = fused.get(&MemoryId(1)).unwrap();
        assert!(
            (*score - 0.9).abs() < 0.01,
            "k=0 should take max score (0.9), actual={:.3}",
            score
        );
        assert_eq!(*ch, RecallChannel::Bm25);
    }

    #[test]
    fn per_channel_ranks_correctly_assigned() {
        let entity = vec![(MemoryId(1), 0.6f32), (MemoryId(2), 0.2f32)]; // score: 0.6 > 0.2 → rank 0/1
        let r = multi_channel_seeds(
            "q",
            &entity,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            20,
        );
        let entity_seeds: Vec<_> = r
            .seeds
            .iter()
            .filter(|s| s.channel == RecallChannel::EntityInverted)
            .collect();
        assert_eq!(entity_seeds.len(), 2);
        // rank 0 = highest score (0.6), rank 1 = second (0.2)
        let (rank0, rank1) = if entity_seeds[0].score > entity_seeds[1].score {
            (
                entity_seeds[0].rank_in_channel,
                entity_seeds[1].rank_in_channel,
            )
        } else {
            (
                entity_seeds[1].rank_in_channel,
                entity_seeds[0].rank_in_channel,
            )
        };
        assert_eq!(rank0, Some(0));
        assert_eq!(rank1, Some(1));
    }
}
