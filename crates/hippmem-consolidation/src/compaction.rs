//! Edge Compaction: weak-edge archiving + node degree governance (03 §7).
//!
//! Weak edges below the threshold are archived (not physically deleted);
//! when the degree limit is exceeded, the strongest edges are retained.

use hippmem_core::model::links::{AssociationLink, LinkType, ObservationState};

/// Compaction parameters.
pub struct CompactionParams {
    pub min_retained_strength: f32,
    pub degree_limit: usize,
}

impl Default for CompactionParams {
    fn default() -> Self {
        Self {
            min_retained_strength: 0.12,
            degree_limit: 64,
        }
    }
}

/// Protected set: edge types not subject to compaction.
const PROTECTED_TYPES: &[LinkType] = &[
    LinkType::Causal,
    LinkType::Correction,
    LinkType::Contradiction,
    LinkType::Supersedes,
];

/// Performs edge compaction:
/// - Returns (kept edges, archived edges)
/// - Protected edges are unaffected
/// - Non-protected edges below `min_retained_strength` → archived
/// - When `degree_limit` is exceeded → the highest-strength edges are retained
pub fn compact_edges(
    links: Vec<AssociationLink>,
    params: &CompactionParams,
) -> (Vec<AssociationLink>, Vec<AssociationLink>) {
    let mut kept = Vec::new();
    let mut archived = Vec::new();

    for link in links {
        let is_protected = PROTECTED_TYPES.contains(&link.link_type)
            && !matches!(link.observation, ObservationState::Observing { .. });

        let keep = is_protected || link.strength.value() >= params.min_retained_strength;
        if keep {
            kept.push(link);
        } else {
            archived.push(link);
        }
    }

    // Degree limit: retain the highest-strength edges
    if kept.len() > params.degree_limit {
        kept.sort_by(|a, b| {
            b.strength
                .value()
                .partial_cmp(&a.strength.value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let overflow: Vec<_> = kept.drain(params.degree_limit..).collect();
        archived.extend(overflow);
    }

    (kept, archived)
}
