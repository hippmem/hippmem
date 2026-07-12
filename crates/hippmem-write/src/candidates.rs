//! Multi-dimensional candidate discovery: match on multiple dimensions based
//! on AssociationKeys (03 §2).

use hippmem_core::model::links::{AssociationKeys, MatchDimension};
use std::collections::HashSet;

/// Candidate discovery result: includes matched dimensions and similarity info.
#[derive(Debug, Clone)]
pub struct CandidateResult {
    /// List of dimensions that hit.
    pub matched_dimensions: Vec<MatchDimension>,
    /// Entity Jaccard similarity: |A∩B| / |A∪B| (03 §2.1).
    pub entity_jaccard: f32,
    /// Topic Jaccard similarity.
    pub topic_jaccard: f32,
    /// Temporal key intersection size (temporal proximity is not Jaccard,
    /// 03 §2.1).
    pub temporal_overlap: usize,
    /// Goal Jaccard similarity.
    pub goal_jaccard: f32,
    /// Event Jaccard similarity.
    pub event_jaccard: f32,
    /// Causal key intersection size (causal crossover is not Jaccard, 03 §2.1).
    pub causal_overlap: usize,
    /// Emotion key intersection size (count of same emotion categories,
    /// 03 §2.1).
    pub emotion_overlap: usize,
    /// Importance value of the existing memory (filled in by the caller from
    /// MemoryUnit, 03 §2.1).
    pub importance_value: f32,
    /// Context sharing score (shared session/conversation/project/task,
    /// 03 §2.1).
    /// Filled in by the caller after computing; discover_candidates only sets
    /// the default value 0.0.
    pub co_context_score: f32,
    /// SimHash Hamming similarity (0.0 = completely different, 1.0 = identical).
    pub lexical_similarity: f32,
    /// Semantic binary-code Hamming similarity.
    pub semantic_binary_similarity: f32,
}

/// Discover multi-dimensional candidate matches between two AssociationKeys.
pub fn discover_candidates(a: &AssociationKeys, b: &AssociationKeys) -> CandidateResult {
    let mut dims = Vec::new();

    let set = |v: &[u64]| -> HashSet<u64> { v.iter().copied().collect() };

    // Entity: Jaccard similarity (03 §2.1)
    let a_ent = set(&a.entity_keys);
    let b_ent = set(&b.entity_keys);
    let ent_intersect = a_ent.intersection(&b_ent).count();
    let ent_union = a_ent.len() + b_ent.len() - ent_intersect;
    let entity_jaccard = if ent_union > 0 {
        ent_intersect as f32 / ent_union as f32
    } else {
        0.0
    };
    if entity_jaccard > 0.0 {
        dims.push(MatchDimension::Entity);
    }

    // Topic: Jaccard similarity
    let a_top = set(&a.topic_keys);
    let b_top = set(&b.topic_keys);
    let top_intersect = a_top.intersection(&b_top).count();
    let top_union = a_top.len() + b_top.len() - top_intersect;
    let topic_jaccard = if top_union > 0 {
        top_intersect as f32 / top_union as f32
    } else {
        0.0
    };
    if topic_jaccard > 0.0 {
        dims.push(MatchDimension::Topic);
    }

    // Temporal: time-bucket overlap (not Jaccard, 03 §2.1)
    let a_tmp: HashSet<u32> = a.temporal_keys.iter().copied().collect();
    let b_tmp: HashSet<u32> = b.temporal_keys.iter().copied().collect();
    let temporal_overlap = a_tmp.intersection(&b_tmp).count();
    if temporal_overlap > 0 {
        dims.push(MatchDimension::Temporal);
    }

    // Goal: Jaccard similarity
    let a_goal = set(&a.goal_keys);
    let b_goal = set(&b.goal_keys);
    let goal_intersect = a_goal.intersection(&b_goal).count();
    let goal_union = a_goal.len() + b_goal.len() - goal_intersect;
    let goal_jaccard = if goal_union > 0 {
        goal_intersect as f32 / goal_union as f32
    } else {
        0.0
    };
    if goal_jaccard > 0.0 {
        dims.push(MatchDimension::Goal);
    }

    // Event: Jaccard similarity
    let a_evt = set(&a.event_keys);
    let b_evt = set(&b.event_keys);
    let evt_intersect = a_evt.intersection(&b_evt).count();
    let evt_union = a_evt.len() + b_evt.len() - evt_intersect;
    let event_jaccard = if evt_union > 0 {
        evt_intersect as f32 / evt_union as f32
    } else {
        0.0
    };
    if event_jaccard > 0.0 {
        dims.push(MatchDimension::Event);
    }

    // Causal: causal-key crossover (not Jaccard, 03 §2.1)
    let a_cau = set(&a.causal_keys);
    let b_cau = set(&b.causal_keys);
    let causal_overlap = a_cau.intersection(&b_cau).count();
    if causal_overlap > 0 {
        dims.push(MatchDimension::Causal);
    }

    // Emotion: emotion-category overlap (u8 direct comparison, 03 §2.1)
    let a_emo: HashSet<u8> = a.emotion_keys.iter().copied().collect();
    let b_emo: HashSet<u8> = b.emotion_keys.iter().copied().collect();
    let emotion_overlap = a_emo.intersection(&b_emo).count();
    if emotion_overlap > 0 {
        dims.push(MatchDimension::Emotion);
    }

    let lexical_similarity =
        simhash_similarity(&a.lexical_signature.simhash, &b.lexical_signature.simhash);
    // Only count as a semantic match when both SimHashes have signal (not all
    // zero) and are similar enough
    let a_has_signal = a.lexical_signature.simhash.iter().any(|&x| x != 0);
    let b_has_signal = b.lexical_signature.simhash.iter().any(|&x| x != 0);
    if a_has_signal && b_has_signal && lexical_similarity > 0.7 {
        dims.push(MatchDimension::Semantic);
    }

    let binary_sim = binary_similarity(
        &a.semantic_signature.binary_code,
        &b.semantic_signature.binary_code,
    );

    CandidateResult {
        matched_dimensions: dims,
        entity_jaccard,
        topic_jaccard,
        temporal_overlap,
        goal_jaccard,
        event_jaccard,
        causal_overlap,
        emotion_overlap,
        importance_value: 0.0,
        co_context_score: 0.0,
        lexical_similarity,
        semantic_binary_similarity: binary_sim,
    }
}

pub(crate) fn simhash_similarity(a: &[u64; 4], b: &[u64; 4]) -> f32 {
    let same: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| 64 - (x ^ y).count_ones())
        .sum();
    same as f32 / 256.0
}

fn binary_similarity(a: &[u64; 2], b: &[u64; 2]) -> f32 {
    let same: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| 64 - (x ^ y).count_ones())
        .sum();
    same as f32 / 128.0
}
