//! Explanation path: produces a human-readable recall path description from the activation trace (03 §4.4, constitution C4).

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{ActivationStep, LinkType, MatchDimension, RecallChannel};

/// Map a recall channel to a match dimension.
///
/// Each `RecallChannel` corresponds to a semantically matching `MatchDimension`.
/// `GraphSpreading` is not a seed channel (it is a spreading attribution marker), and returns `None`.
pub fn recall_channel_to_match_dimension(channel: RecallChannel) -> Option<MatchDimension> {
    match channel {
        RecallChannel::EntityInverted => Some(MatchDimension::Entity),
        RecallChannel::Bm25 => Some(MatchDimension::Semantic),
        RecallChannel::SemanticDense => Some(MatchDimension::Semantic),
        RecallChannel::SemanticBinary => Some(MatchDimension::Semantic),
        RecallChannel::Temporal => Some(MatchDimension::Temporal),
        RecallChannel::TopicCluster => Some(MatchDimension::Topic),
        RecallChannel::Goal => Some(MatchDimension::Goal),
        RecallChannel::Event => Some(MatchDimension::Event),
        RecallChannel::Causal => Some(MatchDimension::Causal),
        RecallChannel::RecentActivation => Some(MatchDimension::CoContext),
        RecallChannel::GraphSpreading => None, // non-seed channel, spreading attribution marker
    }
}

/// Deduce hit dimensions from the activation trace.
pub fn deduce_dimensions(trace: &[ActivationStep]) -> Vec<MatchDimension> {
    let mut dims = Vec::new();
    for step in trace {
        if let Some(link_type) = step.via_link {
            match link_type {
                LinkType::EntityOverlap => dims.push(MatchDimension::Entity),
                LinkType::Causal => dims.push(MatchDimension::Causal),
                LinkType::TopicRelated => dims.push(MatchDimension::Topic),
                LinkType::TemporalAdjacent => dims.push(MatchDimension::Temporal),
                LinkType::SemanticSimilar => dims.push(MatchDimension::Semantic),
                LinkType::SameGoal => dims.push(MatchDimension::Goal),
                LinkType::SameEvent => dims.push(MatchDimension::Event),
                LinkType::EmotionalResonance => dims.push(MatchDimension::Emotion),
                LinkType::CoActivation => dims.push(MatchDimension::CoContext),
                _ => {}
            }
        }
    }
    // Seed channel maps to a MatchDimension (replaces the former hardcoded Importance)
    if let Some(first) = trace.first() {
        if first.hop == 0 {
            if let Some(channel) = first.channel {
                if let Some(dim) = recall_channel_to_match_dimension(channel) {
                    dims.push(dim);
                }
            }
        }
    }
    dims.dedup();
    dims
}

/// Generate a short English explanation text (for logs/diagnostics).
pub fn explain_trace(trace: &[ActivationStep], seed_id: MemoryId) -> String {
    if trace.len() <= 1 {
        return format!("direct seed recall (id={})", seed_id.0);
    }
    let mut parts = vec![format!("seed={}", seed_id.0)];
    for step in trace.iter().skip(1) {
        if let Some(lt) = step.via_link {
            parts.push(format!(
                "-{:?}->{} (e={:.3})",
                lt, step.to.0, step.energy_in
            ));
        }
    }
    parts.join(" ")
}
