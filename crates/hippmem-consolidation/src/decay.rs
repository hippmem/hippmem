//! Edge decay logic: per-cycle decay with a protected set (03 §7).
//!
//! The protected set (non-observing edges of type Causal/Correction/Supersedes)
//! is not decayed; stale candidates in the observation zone are pruned automatically.

use hippmem_core::model::links::{AssociationLink, LinkType, ObservationState};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;

/// Decay parameters.
pub struct DecayParams {
    pub decay_per_cycle: f32,
    pub min_retained_strength: f32,
}

impl Default for DecayParams {
    fn default() -> Self {
        Self {
            decay_per_cycle: 0.97,
            min_retained_strength: 0.12,
        }
    }
}

/// Protected set: edge types not subject to decay (Constitution C7: decision basis / causal chain / correction record / long-term preferences must not be deleted).
const PROTECTED_TYPES: &[LinkType] = &[
    LinkType::Causal,
    LinkType::Correction,
    LinkType::Contradiction,
    LinkType::Supersedes,
];

/// Returns whether an edge is protected.
fn is_protected(link: &AssociationLink) -> bool {
    // Observation-zone edges are not protected (even if the type matches)
    if matches!(link.observation, ObservationState::Observing { .. }) {
        return false;
    }
    PROTECTED_TYPES.contains(&link.link_type)
}

/// Applies decay to the edge list: protected edges keep their strength,
/// ordinary edges are multiplied by decay_per_cycle, and observing edges below the threshold are removed.
pub fn apply_decay_with_protection(
    links: &mut Vec<AssociationLink>,
    params: &DecayParams,
    now: Timestamp,
) {
    links.retain_mut(|link| {
        if is_protected(link) {
            return true;
        }

        let inactive_duration = now.0 - link.last_activated_at.unwrap_or(link.formed_at).0;
        let raw_decayed = if inactive_duration > 86_400_000 {
            link.strength.value() * params.decay_per_cycle
        } else {
            link.strength.value()
        };

        // Observation-zone edges: drop if decayed value falls below the threshold
        let is_observing = matches!(link.observation, ObservationState::Observing { .. });
        if is_observing && raw_decayed < params.min_retained_strength {
            return false;
        }

        // Ordinary edges: apply decay with a strength floor
        let new_strength = raw_decayed.max(params.min_retained_strength);
        link.strength = UnitScore::new(new_strength);
        true
    });
}
