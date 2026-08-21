//! Edge decay logic: forgetting-curve decay with a protected set (03 §7,
//! memory-learning-mechanism 0.4.3).
//!
//! Strength follows an exponential forgetting curve: `strength × 0.5^(Δt /
//! half_life)`. The half-life grows with each activation (review) — an edge
//! activated N times forgets 2^N times slower — so spaced reviews (repeated
//! co-activation/confirmation) consolidate an edge into long-term memory
//! while unused edges fade per the curve. The protected set
//! (non-observing edges of type Causal/Correction/Contradiction/Supersedes)
//! is not decayed; stale candidates in the observation zone are pruned.

use hippmem_core::model::links::{AssociationLink, LinkType, ObservationState};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;

/// Decay parameters (forgetting-curve model).
pub struct DecayParams {
    /// Base half-life: how long an edge lasts without any review.
    pub half_life_base_ms: i64,
    /// Each activation doubles the half-life; capped at this many doublings.
    pub half_life_max_doublings: u32,
    pub min_retained_strength: f32,
}

impl Default for DecayParams {
    fn default() -> Self {
        Self {
            half_life_base_ms: 86_400_000, // 1 day base half-life
            half_life_max_doublings: 6,    // up to 64 days (2^6)
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

        // Forgetting curve: exponential decay with a review-dependent
        // half-life. Each activation (Hebbian reinforcement or a confirmed
        // retrieval path) doubles the half-life, so repeatedly reviewed edges
        // consolidate into long-term memory while unused edges fade.
        let inactive_ms = (now.0 - link.last_activated_at.unwrap_or(link.formed_at).0).max(0);
        let doublings = link.activation_count.min(params.half_life_max_doublings);
        let half_life = (params.half_life_base_ms as f64) * 2f64.powi(doublings as i32);
        let decay = 0.5f64.powf(inactive_ms as f64 / half_life);
        let raw_decayed = link.strength.value() as f64 * decay;

        // Observation-zone edges: drop if decayed value falls below the threshold
        let is_observing = matches!(link.observation, ObservationState::Observing { .. });
        if is_observing && raw_decayed < params.min_retained_strength as f64 {
            return false;
        }

        // Ordinary edges: apply decay with a strength floor
        let new_strength = raw_decayed.max(params.min_retained_strength as f64) as f32;
        link.strength = UnitScore::new(new_strength);
        true
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::model::links::{LinkDirection, LinkEvidence, LinkType};

    fn make_link(
        strength: f32,
        last_activated: Option<Timestamp>,
        activation_count: u32,
    ) -> AssociationLink {
        AssociationLink {
            target_id: hippmem_core::ids::MemoryId(2),
            link_type: LinkType::EntityOverlap,
            direction: LinkDirection::Forward,
            strength: UnitScore::new(strength),
            confidence: UnitScore::new(0.5),
            evidence: LinkEvidence {
                contributing_dimensions: vec![],
                score_breakdown: vec![],
                text_spans: vec![],
                note: None,
            },
            formed_at: Timestamp(0),
            last_activated_at: last_activated,
            activation_count,
            observation: ObservationState::Confirmed,
        }
    }

    const DAY: i64 = 86_400_000;

    #[test]
    fn unused_edge_fades_along_forgetting_curve() {
        // Base half-life is one day: after one day without review the
        // strength is halved (0.8 → ~0.4).
        let mut links = vec![make_link(0.8, Some(Timestamp(0)), 0)];
        apply_decay_with_protection(&mut links, &DecayParams::default(), Timestamp(DAY));
        assert!(
            (links[0].strength.value() - 0.4).abs() < 0.01,
            "one half-life must halve the strength, got {}",
            links[0].strength.value()
        );
    }

    #[test]
    fn reviewed_edge_forgets_much_slower() {
        // Three activations double the half-life three times: 8 days. After
        // 8 days the reviewed edge is at half strength, while an unreviewed
        // edge (1-day half-life) has decayed to the floor.
        let mut reviewed = vec![make_link(0.8, Some(Timestamp(0)), 3)];
        apply_decay_with_protection(&mut reviewed, &DecayParams::default(), Timestamp(8 * DAY));
        assert!(
            (reviewed[0].strength.value() - 0.4).abs() < 0.01,
            "reviewed edge (half-life 8d) must be ~0.4 after 8 days, got {}",
            reviewed[0].strength.value()
        );

        let mut unreviewed = vec![make_link(0.8, Some(Timestamp(0)), 0)];
        apply_decay_with_protection(&mut unreviewed, &DecayParams::default(), Timestamp(8 * DAY));
        assert_eq!(
            unreviewed[0].strength.value(),
            0.12,
            "unreviewed edge (half-life 1d) must hit the floor after 8 days"
        );
    }
}
