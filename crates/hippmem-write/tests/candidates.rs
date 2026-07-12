//! acceptance tests: candidate-discovery algorithm
//! (Jaccard/SimHash/Binary/thresholds/empty sets)
//!
//! Verifies the correctness of 03 §1 (AssociationKeys generation) and
//! §2.1 (per-dimension sub-scores).

use hippmem_core::model::links::{
    AssociationKeys, LexicalSignature, MatchDimension, SemanticSignature,
};
use hippmem_write::candidates::discover_candidates;

/// Build a minimal AssociationKeys (all fields empty).
fn empty_keys() -> AssociationKeys {
    AssociationKeys {
        entity_keys: vec![],
        temporal_keys: vec![],
        lexical_signature: LexicalSignature { simhash: [0; 4] },
        semantic_signature: SemanticSignature {
            lexical_simhash: [0; 4],
            dense_embedding_ref: None,
            binary_code: [0; 2],
            topic_minhash: [0u32; 16],
        },
        topic_keys: vec![],
        emotion_keys: vec![],
        goal_keys: vec![],
        event_keys: vec![],
        causal_keys: vec![],
    }
}

// ═══════════════════════════════════════════════════════════════════
// Empty keys -> no dimension hit
// ═══════════════════════════════════════════════════════════════════

#[test]
fn empty_keys_no_dimensions() {
    let a = empty_keys();
    let b = empty_keys();
    let result = discover_candidates(&a, &b);
    assert!(
        result.matched_dimensions.is_empty(),
        "empty keys should not hit any dimension"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Jaccard correctness: known sets -> exact values
// ═══════════════════════════════════════════════════════════════════

#[test]
fn jaccard_entity_exact() {
    let a = AssociationKeys {
        entity_keys: vec![1, 2, 3], // |A|=3
        ..empty_keys()
    };
    let b = AssociationKeys {
        entity_keys: vec![2, 3, 4], // |B|=3
        ..empty_keys()
    };
    // intersect={2,3}=2, union={1,2,3,4}=4, jaccard=2/4=0.5
    let result = discover_candidates(&a, &b);
    assert!(
        (result.entity_jaccard - 0.5).abs() < 0.001,
        "entity_jaccard: expected 0.5, got {}",
        result.entity_jaccard
    );
    assert!(result.matched_dimensions.contains(&MatchDimension::Entity));
}

#[test]
fn jaccard_topic_exact() {
    let a = AssociationKeys {
        topic_keys: vec![10, 20, 30, 40], // |A|=4
        ..empty_keys()
    };
    let b = AssociationKeys {
        topic_keys: vec![30, 40, 50], // |B|=3
        ..empty_keys()
    };
    // intersect={30,40}=2, union={10,20,30,40,50}=5, jaccard=2/5=0.4
    let result = discover_candidates(&a, &b);
    assert!(
        (result.topic_jaccard - 0.4).abs() < 0.001,
        "topic_jaccard: expected 0.4, got {}",
        result.topic_jaccard
    );
}

#[test]
fn jaccard_goal_exact() {
    let a = AssociationKeys {
        goal_keys: vec![100, 200], // |A|=2
        ..empty_keys()
    };
    let b = AssociationKeys {
        goal_keys: vec![200], // |B|=1
        ..empty_keys()
    };
    // intersect={200}=1, union={100,200}=2, jaccard=1/2=0.5
    let result = discover_candidates(&a, &b);
    assert!(
        (result.goal_jaccard - 0.5).abs() < 0.001,
        "goal_jaccard: expected 0.5, got {}",
        result.goal_jaccard
    );
}

#[test]
fn jaccard_event_exact() {
    let a = AssociationKeys {
        event_keys: vec![1, 2, 3, 4, 5], // |A|=5
        ..empty_keys()
    };
    let b = AssociationKeys {
        event_keys: vec![1, 2], // |B|=2
        ..empty_keys()
    };
    // intersect={1,2}=2, union={1,2,3,4,5}=5, jaccard=2/5=0.4
    let result = discover_candidates(&a, &b);
    assert!(
        (result.event_jaccard - 0.4).abs() < 0.001,
        "event_jaccard: expected 0.4, got {}",
        result.event_jaccard
    );
}

#[test]
fn jaccard_no_overlap_is_zero() {
    let a = AssociationKeys {
        entity_keys: vec![1, 2],
        ..empty_keys()
    };
    let b = AssociationKeys {
        entity_keys: vec![3, 4],
        ..empty_keys()
    };
    // intersect=0, union=4, jaccard=0/4=0.0
    let result = discover_candidates(&a, &b);
    assert_eq!(
        result.entity_jaccard, 0.0,
        "Jaccard should be 0 with no intersection"
    );
    assert!(!result.matched_dimensions.contains(&MatchDimension::Entity));
}

// ═══════════════════════════════════════════════════════════════════
// SimHash similarity: identical = 1.0, completely different ~= 0.0
// ═══════════════════════════════════════════════════════════════════

#[test]
fn simhash_identical_is_one() {
    let sig = LexicalSignature {
        simhash: [0xABCD1234ABCD1234, 0x5678, 0x9ABC, 0xDEF0],
    };
    let a = AssociationKeys {
        lexical_signature: sig.clone(),
        ..empty_keys()
    };
    let b = AssociationKeys {
        lexical_signature: sig,
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    assert!(
        (result.lexical_similarity - 1.0).abs() < 0.001,
        "identical simhash similarity should be 1.0, got {}",
        result.lexical_similarity
    );
}

#[test]
fn simhash_different_is_low() {
    let a_sig = LexicalSignature {
        simhash: [u64::MAX, u64::MAX, 0, 0],
    };
    let b_sig = LexicalSignature {
        simhash: [0, 0, u64::MAX, u64::MAX],
    };
    let a = AssociationKeys {
        lexical_signature: a_sig,
        ..empty_keys()
    };
    let b = AssociationKeys {
        lexical_signature: b_sig,
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    // 0 bits match -> 0.0
    assert!(
        (result.lexical_similarity - 0.0).abs() < 0.001,
        "completely different simhash similarity should be 0.0, got {}",
        result.lexical_similarity
    );
}

// ═══════════════════════════════════════════════════════════════════
// Binary-code similarity: all same = 1.0, all different = 0.0
// ═══════════════════════════════════════════════════════════════════

#[test]
fn binary_code_identical_is_one() {
    let a = AssociationKeys {
        semantic_signature: SemanticSignature {
            binary_code: [0xABCD1234ABCD1234, 0x5678ABCD],
            ..empty_keys().semantic_signature
        },
        ..empty_keys()
    };
    let b = AssociationKeys {
        semantic_signature: SemanticSignature {
            binary_code: [0xABCD1234ABCD1234, 0x5678ABCD],
            ..empty_keys().semantic_signature
        },
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    assert!(
        (result.semantic_binary_similarity - 1.0).abs() < 0.001,
        "identical binary_code similarity should be 1.0, got {}",
        result.semantic_binary_similarity
    );
}

#[test]
fn binary_code_opposite_is_low() {
    let a = AssociationKeys {
        semantic_signature: SemanticSignature {
            binary_code: [0, 0],
            ..empty_keys().semantic_signature
        },
        ..empty_keys()
    };
    let b = AssociationKeys {
        semantic_signature: SemanticSignature {
            binary_code: [u64::MAX, u64::MAX],
            ..empty_keys().semantic_signature
        },
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    // 0 vs u64::MAX -> all 128 bits differ -> 0/128 = 0.0
    assert!(
        (result.semantic_binary_similarity - 0.0).abs() < 0.001,
        "complementary binary_code similarity should be 0.0, got {}",
        result.semantic_binary_similarity
    );
}

// ═══════════════════════════════════════════════════════════════════
// Threshold check: do not add the Semantic dimension when
// simhash_sim <= 0.7
// ═══════════════════════════════════════════════════════════════════

#[test]
fn simhash_above_threshold_adds_semantic() {
    // 3 u64s match -> 192/256 = 0.75 > 0.7 -> should add Semantic
    let a_sig = LexicalSignature {
        simhash: [u64::MAX, u64::MAX, u64::MAX, 0],
    };
    let b_sig = LexicalSignature {
        simhash: [u64::MAX, u64::MAX, u64::MAX, 0], // identical -> 1.0 > 0.7
    };
    let a = AssociationKeys {
        lexical_signature: a_sig,
        ..empty_keys()
    };
    let b = AssociationKeys {
        lexical_signature: b_sig,
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    // simhash = 1.0 > 0.7, and both sides have signal (non-zero) -> Semantic
    assert!(
        result.lexical_similarity > 0.7,
        "simhash={} should be > 0.7",
        result.lexical_similarity
    );
    assert!(
        result
            .matched_dimensions
            .contains(&MatchDimension::Semantic),
        "simhash > 0.7 should add the Semantic dimension"
    );
}

#[test]
fn simhash_below_threshold_no_semantic() {
    // 0 matches -> 0/256 = 0.0 < 0.7 -> do not add Semantic
    let a_sig = LexicalSignature {
        simhash: [u64::MAX, u64::MAX, 0, 0],
    };
    let b_sig = LexicalSignature {
        simhash: [0, 0, u64::MAX, u64::MAX],
    };
    let a = AssociationKeys {
        lexical_signature: a_sig,
        ..empty_keys()
    };
    let b = AssociationKeys {
        lexical_signature: b_sig,
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    assert!(
        result.lexical_similarity < 0.7,
        "simhash={} should be < 0.7",
        result.lexical_similarity
    );
    assert!(
        !result
            .matched_dimensions
            .contains(&MatchDimension::Semantic),
        "simhash < 0.7 should not add Semantic"
    );
}

#[test]
fn simhash_all_zeros_no_semantic() {
    // Even if the simhashes are identical, as long as they are all zero (no
    // signal), do not add Semantic
    let sig = LexicalSignature { simhash: [0; 4] };
    let a = AssociationKeys {
        lexical_signature: sig.clone(),
        ..empty_keys()
    };
    let b = AssociationKeys {
        lexical_signature: sig,
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    // simhash=1.0 but both sides have no signal (all zero) -> do not add
    assert!(
        !result
            .matched_dimensions
            .contains(&MatchDimension::Semantic),
        "all-zero simhash (no signal) should not add Semantic"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Temporal/Causal/Emotion: intersection-size tests (not Jaccard)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn temporal_overlap_count() {
    let a = AssociationKeys {
        temporal_keys: vec![1, 2, 3, 4],
        ..empty_keys()
    };
    let b = AssociationKeys {
        temporal_keys: vec![3, 4, 5],
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    assert_eq!(
        result.temporal_overlap, 2,
        "temporal intersection should be 2"
    );
    assert!(result
        .matched_dimensions
        .contains(&MatchDimension::Temporal));
}

#[test]
fn causal_overlap_count() {
    let a = AssociationKeys {
        causal_keys: vec![10, 20],
        ..empty_keys()
    };
    let b = AssociationKeys {
        causal_keys: vec![20, 30],
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    assert_eq!(result.causal_overlap, 1, "causal intersection should be 1");
    assert!(result.matched_dimensions.contains(&MatchDimension::Causal));
}

#[test]
fn emotion_overlap_count() {
    let a = AssociationKeys {
        emotion_keys: vec![1, 2, 3],
        ..empty_keys()
    };
    let b = AssociationKeys {
        emotion_keys: vec![2, 3, 4],
        ..empty_keys()
    };
    let result = discover_candidates(&a, &b);
    assert_eq!(
        result.emotion_overlap, 2,
        "emotion intersection should be 2"
    );
    assert!(result.matched_dimensions.contains(&MatchDimension::Emotion));
}
