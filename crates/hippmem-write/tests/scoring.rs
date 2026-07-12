//! acceptance tests: multi-dimensional candidate discovery +
//! association scoring

use hippmem_core::config::AlgoParams;
use hippmem_core::model::links::{
    AssociationKeys, LexicalSignature, MatchDimension, SemanticSignature,
};
use hippmem_write::candidates::discover_candidates;
use hippmem_write::scoring::associate_score;

fn make_keys(entity_count: u64, topic_count: u64) -> AssociationKeys {
    use hippmem_core::ids::{EntityKey, TopicKey};
    AssociationKeys {
        entity_keys: (0..entity_count)
            .map(|i| i + 100)
            .collect::<Vec<EntityKey>>(),
        temporal_keys: vec![42],
        lexical_signature: LexicalSignature {
            simhash: [1, 2, 3, 4],
        },
        semantic_signature: SemanticSignature {
            lexical_simhash: [1, 2, 3, 4],
            dense_embedding_ref: None,
            binary_code: [0xABCD, 0x1234],
            topic_minhash: [0u32; 16],
        },
        topic_keys: (0..topic_count).map(|i| i + 200).collect::<Vec<TopicKey>>(),
        emotion_keys: vec![],
        goal_keys: vec![300],
        event_keys: vec![],
        causal_keys: vec![],
    }
}

/// Discover candidates when two AssociationKeys share multiple dimensions.
#[test]
fn candidate_discovery_multi_dim() {
    let a = make_keys(3, 2);
    let b = make_keys(3, 2);

    let result = discover_candidates(&a, &b);
    // Should hit entity (3 shared), topic (2 shared), and temporal (identical)
    let matched_dims = result.matched_dimensions;
    assert!(matched_dims.len() >= 2, "at least 2 dimensions should hit");
}

/// Two keys with no shared dimensions return a zero score.
#[test]
fn no_shared_dimensions_zero_score() {
    let zero_sig = SemanticSignature {
        lexical_simhash: [0; 4],
        dense_embedding_ref: None,
        binary_code: [0, 0],
        topic_minhash: [0u32; 16],
    };
    let a = AssociationKeys {
        entity_keys: vec![1],
        temporal_keys: vec![1],
        lexical_signature: LexicalSignature { simhash: [0; 4] },
        semantic_signature: zero_sig.clone(),
        topic_keys: vec![10],
        emotion_keys: vec![],
        goal_keys: vec![],
        event_keys: vec![],
        causal_keys: vec![],
    };
    let b = AssociationKeys {
        entity_keys: vec![999],
        temporal_keys: vec![99],
        lexical_signature: LexicalSignature { simhash: [0; 4] },
        semantic_signature: zero_sig,
        topic_keys: vec![888],
        emotion_keys: vec![],
        goal_keys: vec![],
        event_keys: vec![],
        causal_keys: vec![],
    };

    let result = discover_candidates(&a, &b);
    assert!(
        result.matched_dimensions.is_empty(),
        "no shared dimensions should not hit"
    );
    let score = associate_score(&result, 0, 1000, &AlgoParams::default());
    assert!(
        score.value() < 0.1,
        "score with no shared dimensions should be low"
    );
}

/// Multi-dim bonus: >=3 dimensions hit yields a score bonus.
#[test]
fn multi_dim_bonus() {
    let a = AssociationKeys {
        entity_keys: vec![1, 2, 3],
        temporal_keys: vec![100],
        topic_keys: vec![10, 20],
        goal_keys: vec![5],
        ..make_keys(0, 0)
    };
    let b = AssociationKeys {
        entity_keys: vec![1, 2, 3], // entity hit
        temporal_keys: vec![100],   // temporal hit
        topic_keys: vec![10, 20],   // topic hit
        goal_keys: vec![5],         // goal hit
        ..a.clone()
    };

    let result = discover_candidates(&a, &b);
    // At least 4 dimensions match -> multi_dim_bonus should trigger
    assert!(result.matched_dimensions.len() >= 3);
}

/// Score is in the [0, 1] range.
#[test]
fn score_in_expected_range() {
    let a = make_keys(3, 2);
    let b = make_keys(3, 2);

    let result = discover_candidates(&a, &b);
    let score = associate_score(&result, 2, 1000, &AlgoParams::default());
    let v = score.value();
    assert!((0.0..=1.0).contains(&v), "score {v} should be in [0,1]");
}

// ═══════════════════════════════════════════════════════════════════
// per-dimension exact verification of the 10 weights
// (03 §0 parameter table, §2 association scoring)
// ═══════════════════════════════════════════════════════════════════

use hippmem_write::candidates::CandidateResult;

/// Helper: build a CandidateResult containing only the specified dimension,
/// with all values set to 0.5
fn single_dim_cand(dim: MatchDimension) -> CandidateResult {
    let mut c = CandidateResult {
        matched_dimensions: vec![dim],
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
    // Set the field corresponding to the dimension to 0.5
    match dim {
        MatchDimension::Entity => c.entity_jaccard = 0.5,
        MatchDimension::Topic => c.topic_jaccard = 0.5,
        MatchDimension::Temporal => c.temporal_overlap = 1,
        MatchDimension::Goal => c.goal_jaccard = 0.5,
        MatchDimension::Event => c.event_jaccard = 0.5,
        MatchDimension::Causal => c.causal_overlap = 1,
        MatchDimension::Semantic => c.lexical_similarity = 0.5,
        // Emotion/Context/Importance use dedicated fields
        _ => {}
    }
    c
}

#[test]
fn weight_entity_exact() {
    let params = AlgoParams::default();
    let c = single_dim_cand(MatchDimension::Entity);
    // w_entity=0.20, jaccard=0.5, no other dimensions
    // raw = 0.20*0.5 = 0.10
    let s = associate_score(&c, 1, 1000, &params);
    assert!(
        (s.value() - 0.10).abs() < 0.001,
        "entity: expected 0.10, got {}",
        s.value()
    );
}

#[test]
fn weight_topic_exact() {
    let params = AlgoParams::default();
    let c = single_dim_cand(MatchDimension::Topic);
    // w_topic=0.10, jaccard=0.5
    let s = associate_score(&c, 1, 1000, &params);
    assert!(
        (s.value() - 0.05).abs() < 0.001,
        "topic: expected 0.05, got {}",
        s.value()
    );
}

#[test]
fn weight_goal_exact() {
    let params = AlgoParams::default();
    let c = single_dim_cand(MatchDimension::Goal);
    // w_goal=0.12, jaccard=0.5
    let s = associate_score(&c, 1, 1000, &params);
    assert!(
        (s.value() - 0.06).abs() < 0.001,
        "goal: expected 0.06, got {}",
        s.value()
    );
}

#[test]
fn weight_event_exact() {
    let params = AlgoParams::default();
    let c = single_dim_cand(MatchDimension::Event);
    // w_event=0.10, jaccard=0.5
    let s = associate_score(&c, 1, 1000, &params);
    assert!(
        (s.value() - 0.05).abs() < 0.001,
        "event: expected 0.05, got {}",
        s.value()
    );
}

#[test]
fn weight_temporal_exact() {
    let params = AlgoParams::default();
    let c = single_dim_cand(MatchDimension::Temporal);
    // w_temporal=0.10, overlap=1, n=1 → overlap_ratio=1.0
    let s = associate_score(&c, 1, 1000, &params);
    assert!(
        (s.value() - 0.10).abs() < 0.001,
        "temporal: expected 0.10, got {}",
        s.value()
    );
}

#[test]
fn weight_causal_exact() {
    let params = AlgoParams::default();
    let c = single_dim_cand(MatchDimension::Causal);
    // w_causal=0.10, overlap=1, n=1 → overlap_ratio=1.0
    let s = associate_score(&c, 1, 1000, &params);
    assert!(
        (s.value() - 0.10).abs() < 0.001,
        "causal: expected 0.10, got {}",
        s.value()
    );
}

#[test]
fn weight_emotion_exact() {
    let params = AlgoParams::default();
    let c = CandidateResult {
        matched_dimensions: vec![MatchDimension::Emotion],
        entity_jaccard: 0.0,
        topic_jaccard: 0.0,
        temporal_overlap: 0,
        goal_jaccard: 0.0,
        event_jaccard: 0.0,
        causal_overlap: 0,
        emotion_overlap: 2, // overlap=2, n=1 → ratio=1.0
        importance_value: 0.0,
        co_context_score: 0.0,
        lexical_similarity: 0.0,
        semantic_binary_similarity: 0.0,
    };
    // dim_count=1, matched_dimensions not empty → lexical/binary added
    // But lexical_similarity=0, so only emotion contributes
    let s = associate_score(&c, 1, 1000, &params);
    // w_emotion=0.05, overlap_ratio(2,1)=1.0 → 0.05
    assert!(
        (s.value() - 0.05).abs() < 0.001,
        "emotion: expected 0.05, got {}",
        s.value()
    );
}

#[test]
fn weight_importance_exact() {
    let params = AlgoParams::default();
    let c = CandidateResult {
        matched_dimensions: vec![MatchDimension::Importance],
        entity_jaccard: 0.0,
        topic_jaccard: 0.0,
        temporal_overlap: 0,
        goal_jaccard: 0.0,
        event_jaccard: 0.0,
        causal_overlap: 0,
        emotion_overlap: 0,
        importance_value: 0.8,
        co_context_score: 0.0,
        lexical_similarity: 0.0,
        semantic_binary_similarity: 0.0,
    };
    let s = associate_score(&c, 1, 1000, &params);
    // w_importance=0.02, value=0.8 → 0.016
    assert!(
        (s.value() - 0.016).abs() < 0.001,
        "importance: expected 0.016, got {}",
        s.value()
    );
}

#[test]
fn weight_context_exact() {
    let params = AlgoParams::default();
    let c = CandidateResult {
        matched_dimensions: vec![MatchDimension::CoContext],
        entity_jaccard: 0.0,
        topic_jaccard: 0.0,
        temporal_overlap: 0,
        goal_jaccard: 0.0,
        event_jaccard: 0.0,
        causal_overlap: 0,
        emotion_overlap: 0,
        importance_value: 0.0,
        co_context_score: 0.7,
        lexical_similarity: 0.0,
        semantic_binary_similarity: 0.0,
    };
    let s = associate_score(&c, 1, 1000, &params);
    // w_context=0.03, score=0.7 → 0.021
    assert!(
        (s.value() - 0.021).abs() < 0.001,
        "context: expected 0.021, got {}",
        s.value()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Multi-dim bonus exact verification (03 §2.2)
// ═══════════════════════════════════════════════════════════════════

/// Build a CandidateResult containing only the specified dimensions, with all
/// score fields zeroed
fn multi_dim_cand_only(dims: Vec<MatchDimension>) -> CandidateResult {
    let mut c = CandidateResult {
        matched_dimensions: dims,
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
    for dim in &c.matched_dimensions {
        match dim {
            MatchDimension::Entity => c.entity_jaccard = 0.5,
            MatchDimension::Topic => c.topic_jaccard = 0.5,
            MatchDimension::Temporal => c.temporal_overlap = 1,
            MatchDimension::Goal => c.goal_jaccard = 0.5,
            MatchDimension::Event => c.event_jaccard = 0.5,
            _ => {}
        }
    }
    c
}

#[test]
fn multi_dim_bonus_2_dims_custom_threshold() {
    let params = AlgoParams {
        multi_dim_min_dims: 2,
        ..Default::default()
    };
    let c = multi_dim_cand_only(vec![MatchDimension::Entity, MatchDimension::Topic]);
    // raw = 0.20*0.5 + 0.10*0.5 = 0.10 + 0.05 = 0.15
    // dim_count=2 >= 2 → +multi_dim_bonus(0.15) = 0.30
    let s = associate_score(&c, 2, 1000, &params);
    let expected = 0.20 * 0.5 + 0.10 * 0.5 + 0.15;
    assert!(
        (s.value() - expected).abs() < 0.001,
        "2-dims: expected {expected}, got {}",
        s.value()
    );
}

#[test]
fn multi_dim_bonus_3_dims_default() {
    let params = AlgoParams::default(); // min_dims=3
    let c = multi_dim_cand_only(vec![
        MatchDimension::Entity,
        MatchDimension::Topic,
        MatchDimension::Temporal,
    ]);
    // 0.20*0.5 + 0.10*0.5 + 0.10*(1/3) + bonus(0.15) = 0.10+0.05+0.0333+0.15 = 0.3333
    let s = associate_score(&c, 3, 1000, &params);
    let expected = 0.20 * 0.5 + 0.10 * 0.5 + 0.10 * (1.0 / 3.0) + 0.15;
    assert!(
        (s.value() - expected).abs() < 0.001,
        "3-dims: expected {expected}, got {}",
        s.value()
    );
}

#[test]
fn multi_dim_bonus_4_dims() {
    let params = AlgoParams::default();
    let c = multi_dim_cand_only(vec![
        MatchDimension::Entity,
        MatchDimension::Topic,
        MatchDimension::Temporal,
        MatchDimension::Goal,
    ]);
    // 0.20*0.5 + 0.10*0.5 + 0.10*(1/4) + 0.12*0.5 + 0.15
    // = 0.10 + 0.05 + 0.025 + 0.06 + 0.15 = 0.385
    let s = associate_score(&c, 4, 1000, &params);
    let expected = 0.20 * 0.5 + 0.10 * 0.5 + 0.10 * (1.0 / 4.0) + 0.12 * 0.5 + 0.15;
    assert!(
        (s.value() - expected).abs() < 0.001,
        "4-dims: expected {expected}, got {}",
        s.value()
    );
}

#[test]
fn multi_dim_bonus_5_dims() {
    let params = AlgoParams::default();
    let c = multi_dim_cand_only(vec![
        MatchDimension::Entity,
        MatchDimension::Topic,
        MatchDimension::Temporal,
        MatchDimension::Goal,
        MatchDimension::Event,
    ]);
    // 0.20*0.5 + 0.10*0.5 + 0.10*(1/5) + 0.12*0.5 + 0.10*0.5 + 0.15
    // = 0.10 + 0.05 + 0.02 + 0.06 + 0.05 + 0.15 = 0.43
    let s = associate_score(&c, 5, 1000, &params);
    let expected = 0.20 * 0.5 + 0.10 * 0.5 + 0.10 * (1.0 / 5.0) + 0.12 * 0.5 + 0.10 * 0.5 + 0.15;
    assert!(
        (s.value() - expected).abs() < 0.001,
        "5-dims: expected {expected}, got {}",
        s.value()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Single-semantic penalty + cold start (03 §2.3) — supplementary tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn single_semantic_penalty_cold_start() {
    let params = AlgoParams::default();
    let c = CandidateResult {
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
    };
    // total=100 < cold_start_count=500 → factor = 0.6
    // lexical 0.18*0.5 = 0.09, * 0.6 = 0.054
    let s_cold = associate_score(&c, 1, 100, &params);
    let expected_cold = 0.18 * 0.5 * 0.6;
    assert!(
        (s_cold.value() - expected_cold).abs() < 0.001,
        "semantic+cold_start: expected {expected_cold}, got {}",
        s_cold.value()
    );

    // total=1000 >= 2*cold_start → factor = 1.0
    let s_warm = associate_score(&c, 1, 1000, &params);
    let expected_warm = 0.18 * 0.5 * 1.0;
    assert!(
        (s_warm.value() - expected_warm).abs() < 0.001,
        "semantic+warm: expected {expected_warm}, got {}",
        s_warm.value()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Score clamp boundary values (03 §2.2)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn score_clamped_to_one() {
    let params = AlgoParams::default();
    let c = CandidateResult {
        matched_dimensions: vec![
            MatchDimension::Entity,
            MatchDimension::Topic,
            MatchDimension::Temporal,
            MatchDimension::Goal,
            MatchDimension::Event,
        ],
        entity_jaccard: 1.0,
        topic_jaccard: 1.0,
        temporal_overlap: 10,
        goal_jaccard: 1.0,
        event_jaccard: 1.0,
        causal_overlap: 0,
        emotion_overlap: 0,
        importance_value: 1.0,
        co_context_score: 1.0,
        lexical_similarity: 1.0,
        semantic_binary_similarity: 1.0,
    };
    let s = associate_score(&c, 10, 1000, &params);
    assert!(
        s.value() <= 1.0,
        "score should not exceed 1.0, got={}",
        s.value()
    );
}

#[test]
fn score_not_negative() {
    let params = AlgoParams::default();
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
    let s = associate_score(&c, 0, 1000, &params);
    assert!(
        s.value() >= 0.0,
        "score should not be negative, got={}",
        s.value()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Semantic-channel weights participate only when other dimensions hit
// (03 §2.2)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn lexical_only_with_other_dims() {
    let params = AlgoParams::default();
    // No matched_dimensions -> lexical does not participate
    let c_empty = CandidateResult {
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
        lexical_similarity: 1.0,         // high lexical
        semantic_binary_similarity: 1.0, // high binary
    };
    let s = associate_score(&c_empty, 0, 1000, &params);
    assert_eq!(
        s.value(),
        0.0,
        "lexical/binary should not participate without other dimensions"
    );

    // Has matched_dimensions -> lexical participates
    let c_has = CandidateResult {
        matched_dimensions: vec![MatchDimension::Entity],
        entity_jaccard: 0.0, // entity=0, but lexical will participate
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
    };
    let s2 = associate_score(&c_has, 1, 1000, &params);
    // entity=0, but lexical=0.18*0.5=0.09
    assert!(
        s2.value() > 0.0,
        "lexical should participate when other dimensions are present"
    );
    assert!(
        (s2.value() - 0.09).abs() < 0.001,
        "lexical: expected 0.09, got {}",
        s2.value()
    );
}
