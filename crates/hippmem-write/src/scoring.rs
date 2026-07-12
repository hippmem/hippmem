//! Association scoring: multi-dimensional scoring function (03 §2).

use crate::candidates::CandidateResult;
use hippmem_core::config::AlgoParams;
use hippmem_core::score::UnitScore;

/// Lexical/binary are not yet split out in AlgoParams; keep temporary constants.
const W_LEXICAL: f32 = 0.18;
const W_BINARY: f32 = 0.10;

/// Compute the cold-start warm-up factor (03 §2.3).
///
/// When the total memory count < `cold_start_count`, the single-semantic
/// penalty is active (= `single_semantic_penalty`);
/// as memories grow, the factor ramps linearly from `single_semantic_penalty`
/// back up to 1.0;
/// when `total >= 2 * cold_start_count`, the penalty disappears entirely.
fn cold_start_factor(total_memories: u32, params: &AlgoParams) -> f32 {
    let cold = params.cold_start_count;
    if total_memories < cold {
        params.single_semantic_penalty
    } else if total_memories >= cold * 2 {
        1.0
    } else {
        let t = (total_memories - cold) as f32 / cold as f32;
        params.single_semantic_penalty + (1.0 - params.single_semantic_penalty) * t
    }
}

/// Compute the association score (range [0, 1]).
///
/// - `candidate`: candidate discovery result (includes Jaccard similarity).
/// - `total_dim_hits`: total number of dimensions hit.
/// - `total_memory_count`: current total memory count (used for cold-start
///   warm-up, 03 §2.3).
/// - `params`: configurable algorithm parameters (weights/bonuses/penalties).
pub fn associate_score(
    c: &CandidateResult,
    total_dim_hits: usize,
    total_memory_count: u32,
    params: &AlgoParams,
) -> UnitScore {
    let n = total_dim_hits.max(1) as f32;

    // Entity/Topic/Goal/Event: use Jaccard similarity (03 §2.1)
    // Temporal/Causal/Emotion: use overlap count normalized
    let mut raw = params.w_entity * c.entity_jaccard
        + params.w_topic * c.topic_jaccard
        + params.w_goal * c.goal_jaccard
        + params.w_event * c.event_jaccard
        + params.w_temporal * overlap_ratio(c.temporal_overlap, n)
        + params.w_causal * overlap_ratio(c.causal_overlap, n)
        + params.w_emotion * overlap_ratio(c.emotion_overlap, n)
        + params.w_importance * c.importance_value
        + params.w_context * c.co_context_score;

    // lexical/binary participate only when other dimensions hit (avoid pure
    // noise signals)
    if !c.matched_dimensions.is_empty() {
        raw += W_LEXICAL * c.lexical_similarity + W_BINARY * c.semantic_binary_similarity;
    }

    // Multi-dimensional bonus
    let dim_count = c.matched_dimensions.len();
    if dim_count >= params.multi_dim_min_dims as usize {
        raw += params.multi_dim_bonus;
    }

    // Single-semantic penalty: penalize only when the Semantic dimension alone
    // hits; the factor warms up with total memory count (03 §2.3)
    if dim_count == 1
        && c.matched_dimensions
            .contains(&hippmem_core::model::links::MatchDimension::Semantic)
    {
        raw *= cold_start_factor(total_memory_count, params);
    }

    UnitScore::new(raw.clamp(0.0, 1.0))
}

fn overlap_ratio(overlap: usize, norm: f32) -> f32 {
    if overlap == 0 {
        0.0
    } else {
        (overlap as f32 / norm).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::model::links::MatchDimension;

    #[test]
    fn zero_overlap_gives_low_score() {
        let c = CandidateResult {
            matched_dimensions: vec![],
            entity_jaccard: 0.0,
            topic_jaccard: 0.0,
            temporal_overlap: 0,
            goal_jaccard: 0.0,
            event_jaccard: 0.0,
            causal_overlap: 0,
            emotion_overlap: 0,
            importance_value: 0.0,
            co_context_score: 0.0,
            lexical_similarity: 0.0,
            semantic_binary_similarity: 0.0,
        };
        let s = associate_score(&c, 1, 1000, &AlgoParams::default());
        assert_eq!(s.value(), 0.0);
    }

    #[test]
    fn jaccard_exact_values() {
        // Entity Jaccard: |A∩B|=2, |A|=3, |B|=3 → union=4 → 2/4=0.5
        let c = CandidateResult {
            matched_dimensions: vec![MatchDimension::Entity],
            entity_jaccard: 0.5, // |{a,b,c} ∩ {a,b,d}| / |{a,b,c} ∪ {a,b,d}| = 2/4
            topic_jaccard: 0.0,
            temporal_overlap: 0,
            goal_jaccard: 0.0,
            event_jaccard: 0.0,
            causal_overlap: 0,
            emotion_overlap: 0,
            importance_value: 0.0,
            co_context_score: 0.0,
            lexical_similarity: 0.0,
            semantic_binary_similarity: 0.0,
        };
        let s = associate_score(&c, 1, 1000, &AlgoParams::default());
        let expected = 0.20 * 0.5; // W_ENTITY * 0.5 = 0.10
        assert!(
            (s.value() - expected).abs() < 0.001,
            "entity_jaccard=0.5 → score ≈ {} (W_ENTITY={})",
            expected,
            0.20
        );
    }

    #[test]
    fn multi_dim_boost() {
        let c = CandidateResult {
            matched_dimensions: vec![
                MatchDimension::Entity,
                MatchDimension::Topic,
                MatchDimension::Temporal,
            ],
            entity_jaccard: 0.6,
            topic_jaccard: 0.5,
            temporal_overlap: 1,
            goal_jaccard: 0.0,
            event_jaccard: 0.0,
            causal_overlap: 0,
            emotion_overlap: 0,
            importance_value: 0.0,
            co_context_score: 0.0,
            lexical_similarity: 0.8,
            semantic_binary_similarity: 0.0,
        };
        let s = associate_score(&c, 3, 1000, &AlgoParams::default());
        // Multi-dim bonus should make the score > the no-bonus case
        assert!(s.value() > 0.3, "multi-dim should get a bonus");
    }

    // ── Cold-start warm-up tests (03 §2.3) ──

    fn make_semantic_only() -> CandidateResult {
        CandidateResult {
            matched_dimensions: vec![MatchDimension::Semantic],
            entity_jaccard: 0.0,
            topic_jaccard: 0.0,
            temporal_overlap: 0,
            goal_jaccard: 0.0,
            event_jaccard: 0.0,
            causal_overlap: 0,
            emotion_overlap: 0,
            importance_value: 0.0,
            co_context_score: 0.0,
            lexical_similarity: 0.5,
            semantic_binary_similarity: 0.0,
        }
    }

    #[test]
    fn cold_start_below_threshold_penalty_applies() {
        let params = AlgoParams::default();
        let c = make_semantic_only();
        // total=100 < cold_start_count=500 → factor = 0.6
        let s = associate_score(&c, 1, 100, &params);
        // score = (lexical 0.18 * 0.5) * 0.6 = 0.09 * 0.6 = 0.054
        assert!(s.value() > 0.0);
        assert!(
            s.value() < 0.1,
            "single-semantic should be penalized during cold start"
        );
    }

    #[test]
    fn cold_start_at_threshold_penalty_applies() {
        let params = AlgoParams::default();
        let c = make_semantic_only();
        // total == cold_start_count=500 → factor = 0.6
        let s_cold = associate_score(&c, 1, 500, &params);
        assert!(s_cold.value() > 0.0);
        assert!(s_cold.value() < 0.1);
    }

    #[test]
    fn cold_start_warming_up_linear() {
        let params = AlgoParams::default();
        let c = make_semantic_only();
        // total=750 (midpoint between 500 and 1000)
        // factor = 0.6 + 0.4 * (250/500) = 0.6 + 0.2 = 0.8
        let s_mid = associate_score(&c, 1, 750, &params);
        // total=500 → factor=0.6
        let s_cold = associate_score(&c, 1, 500, &params);
        // Score during warm-up should be greater than during cold start
        assert!(
            s_mid.value() > s_cold.value(),
            "warm-up score({}) > cold-start score({})",
            s_mid.value(),
            s_cold.value()
        );
    }

    #[test]
    fn cold_start_fully_warmed_no_penalty() {
        let params = AlgoParams::default();
        let c = make_semantic_only();
        // total=1000 (== 2*cold_start_count) → factor = 1.0
        let s_warm = associate_score(&c, 1, 1000, &params);
        // total=100 (cold start) → factor = 0.6
        let s_cold = associate_score(&c, 1, 100, &params);
        // Fully-warmed score = cold-start score / 0.6 (because factor goes from
        // 0.6 to 1.0)
        let expected_ratio = s_warm.value() / s_cold.value();
        assert!(
            expected_ratio > 1.5,
            "fully-warmed score should be much greater than cold-start: warm={}, cold={}, ratio={}",
            s_warm.value(),
            s_cold.value(),
            expected_ratio
        );
    }

    #[test]
    fn cold_start_no_effect_on_multi_dim() {
        let params = AlgoParams::default();
        let c = CandidateResult {
            matched_dimensions: vec![MatchDimension::Entity, MatchDimension::Topic],
            entity_jaccard: 0.6,
            topic_jaccard: 0.5,
            temporal_overlap: 0,
            goal_jaccard: 0.0,
            event_jaccard: 0.0,
            causal_overlap: 0,
            emotion_overlap: 0,
            importance_value: 0.0,
            co_context_score: 0.0,
            lexical_similarity: 0.8,
            semantic_binary_similarity: 0.0,
        };
        // Multi-dim hit: cold start should not affect it (dim_count >= 2)
        let s_cold = associate_score(&c, 2, 100, &params);
        let s_warm = associate_score(&c, 2, 1000, &params);
        assert!(
            (s_cold.value() - s_warm.value()).abs() < 0.001,
            "cold start should not affect score on multi-dim hit"
        );
    }
}
