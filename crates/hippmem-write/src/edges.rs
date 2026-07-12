//! Edge construction logic: strong/weak edges, observation zone, evidence,
//! bidirectional registration, dedup (03 §3).

use crate::candidates::CandidateResult;
use crate::scoring::associate_score;
use hippmem_core::config::AlgoParams;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{
    AssociationLink, LinkDirection, LinkEvidence, LinkType, MatchDimension, ObservationState,
};
use hippmem_core::time::Timestamp;

/// Edge construction parameters.
pub struct EdgeBuildParams {
    pub strong_threshold: f32,
    pub strong_max: usize,
    pub weak_max: usize,
    pub min_score: f32,
    pub observation_max: f32,
    /// Max number of candidates compared per edge build (0 = no limit).
    /// Default 0 (backward compatible).
    /// Used to control O(n^2) edge-construction cost: only build edges with the
    /// N most similar existing memories.
    pub max_candidates: usize,
}

impl Default for EdgeBuildParams {
    fn default() -> Self {
        Self {
            strong_threshold: 0.55,
            strong_max: 8,
            weak_max: 24,
            min_score: 0.25,
            observation_max: 0.55,
            max_candidates: 0, // 0 = no limit, backward compatible
        }
    }
}

pub struct EdgeBuildResult {
    pub created_links: Vec<AssociationLink>,
}

/// Build edges: dedup, no self-loop, bilateral registration.
///
/// - `edge_params`: edge thresholds (strong/weak/observation zone boundaries).
/// - `algo_params`: configurable algorithm parameters (dimension weights /
///   bonuses / penalties, passed to the scoring function).
#[allow(clippy::too_many_arguments)]
pub fn build_edges(
    source_id: MemoryId,
    target_id: MemoryId,
    candidate: &CandidateResult,
    dim_count: usize,
    edge_params: &EdgeBuildParams,
    algo_params: &AlgoParams,
    existing_links: &[AssociationLink],
    now: Timestamp,
    total_memory_count: u32,
) -> EdgeBuildResult {
    if source_id == target_id {
        return EdgeBuildResult {
            created_links: vec![],
        };
    }

    let score = associate_score(candidate, dim_count, total_memory_count, algo_params);
    let sv = score.value();

    if sv < edge_params.min_score {
        return EdgeBuildResult {
            created_links: vec![],
        };
    }

    let link_type = determine_link_type(candidate);

    // Dedup: (target, link_type)
    if existing_links
        .iter()
        .any(|l| l.target_id == target_id && l.link_type == link_type)
    {
        return EdgeBuildResult {
            created_links: vec![],
        };
    }

    let observation = if sv <= edge_params.observation_max {
        ObservationState::Observing { since: now }
    } else {
        ObservationState::Confirmed
    };

    let evidence = LinkEvidence {
        contributing_dimensions: candidate.matched_dimensions.clone(),
        score_breakdown: candidate
            .matched_dimensions
            .iter()
            .map(|&d| (d, sv))
            .collect(),
        text_spans: vec![],
        note: None,
    };

    let link = AssociationLink {
        target_id,
        link_type,
        direction: LinkDirection::Forward,
        strength: score,
        confidence: score,
        evidence,
        formed_at: now,
        last_activated_at: Some(now),
        activation_count: 0,
        observation,
    };

    EdgeBuildResult {
        created_links: vec![link],
    }
}

fn determine_link_type(candidate: &CandidateResult) -> LinkType {
    let dims = &candidate.matched_dimensions;
    // Detect the more specific association type first
    if dims.contains(&MatchDimension::Causal) {
        LinkType::Causal
    } else if dims.contains(&MatchDimension::Entity) {
        LinkType::EntityOverlap
    } else if dims.contains(&MatchDimension::Temporal) {
        LinkType::TemporalAdjacent
    } else if dims.contains(&MatchDimension::Topic) {
        LinkType::TopicRelated
    } else if dims.contains(&MatchDimension::Goal) {
        LinkType::SameGoal
    } else if dims.contains(&MatchDimension::Event) {
        LinkType::SameEvent
    } else {
        LinkType::SemanticSimilar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::score::UnitScore;

    fn make_cand(dims: Vec<MatchDimension>, n: usize) -> CandidateResult {
        CandidateResult {
            matched_dimensions: dims,
            entity_jaccard: if n >= 1 { 0.6 } else { 0.0 },
            topic_jaccard: if n >= 2 { 0.5 } else { 0.0 },
            temporal_overlap: if n >= 3 { 1 } else { 0 },
            goal_jaccard: 0.0,
            event_jaccard: 0.0,
            causal_overlap: 0,
            emotion_overlap: 0,
            importance_value: 0.0,
            co_context_score: 0.0,
            lexical_similarity: 0.8,
            semantic_binary_similarity: 0.0,
        }
    }

    #[test]
    fn no_self_loop() {
        let c = make_cand(vec![MatchDimension::Entity], 1);
        let r = build_edges(
            MemoryId(1),
            MemoryId(1),
            &c,
            2,
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
            &[],
            Timestamp(0),
            1000,
        );
        assert!(r.created_links.is_empty());
    }

    #[test]
    fn dedup_skips_existing() {
        let c = make_cand(vec![MatchDimension::Entity], 2);
        let existing = vec![AssociationLink {
            target_id: MemoryId(2),
            link_type: LinkType::EntityOverlap,
            direction: LinkDirection::Forward,
            strength: UnitScore::new(0.5),
            confidence: UnitScore::new(0.5),
            evidence: LinkEvidence {
                contributing_dimensions: vec![],
                score_breakdown: vec![],
                text_spans: vec![],
                note: None,
            },
            formed_at: Timestamp(0),
            last_activated_at: None,
            activation_count: 0,
            observation: ObservationState::Confirmed,
        }];
        let r = build_edges(
            MemoryId(1),
            MemoryId(2),
            &c,
            2,
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
            &existing,
            Timestamp(0),
            1000,
        );
        assert!(r.created_links.is_empty());
    }

    #[test]
    fn strong_edge_with_multi_dim() {
        let c = make_cand(
            vec![
                MatchDimension::Entity,
                MatchDimension::Topic,
                MatchDimension::Temporal,
            ],
            3,
        );
        let r = build_edges(
            MemoryId(1),
            MemoryId(2),
            &c,
            3,
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
            &[],
            Timestamp(0),
            1000,
        );
        assert!(!r.created_links.is_empty());
    }
}
