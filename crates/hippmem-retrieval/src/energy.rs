//! Activation energy computation: seed initial energy + propagated energy + edge type modifier.
//!
//! Corresponds to 03 §4.1-4.2.

use crate::seeds::Seed;
use hippmem_core::config::AlgoParams;
use hippmem_core::model::links::{AssociationLink, LinkType};

/// Compute the seed initial energy (03 §4.1)
///
/// ```text
/// initial_energy = clamp(
///     query_match * a_query_match   +
///     context_match * b_context_match +
///     importance * c_importance     +
///     freshness * d_freshness       +
///     reliability * e_reliability,
///     0, seed_energy_cap)
/// ```
pub fn initial_energy(
    seed: &Seed,
    query_match: f32,
    context_match: f32,
    importance: f32,
    freshness: f32,
    reliability: f32,
    params: &AlgoParams,
) -> f32 {
    let channel_coeff = params.channel_energy_coeff(seed.channel);
    // Importance amplifies query_match multiplicatively (rather than an additive fixed bonus),
    // ensuring importance only takes effect when the memory is semantically relevant to the query.
    let importance_multiplier = 1.0 + importance * params.c_importance;
    let raw = channel_coeff * query_match * params.a_query_match * importance_multiplier
        + context_match * params.b_context_match
        + freshness * params.d_freshness
        + reliability * params.e_reliability;
    raw.clamp(0.0, params.seed_energy_cap)
}

/// Compute the propagated energy (03 §4.2)
///
/// ```text
/// propagated = source_energy
///            * link.strength
///            * link.confidence
///            * decay_factor^hop
///            * type_modifier(link.link_type)
/// ```
pub fn propagated_energy(
    source_energy: f32,
    link: &AssociationLink,
    hop: u32,
    params: &AlgoParams,
) -> f32 {
    let raw = source_energy
        * link.strength.value()
        * link.confidence.value()
        * params.decay_factor.powi(hop as i32)
        * type_modifier(link.link_type);
    raw.clamp(0.0, 1.0)
}

/// Edge type spreading modifier (03 §4.2 type_modifier table)
///
/// Different edge types contribute differently to spreading. Causal relations promote backtracking,
/// Temporal adjacency decays quickly, and Contradiction should not serve as a main spreading path.
pub fn type_modifier(link_type: LinkType) -> f32 {
    match link_type {
        LinkType::Causal => 1.30,
        LinkType::Correction => 1.20,
        LinkType::Supersedes => 1.15,
        LinkType::SameGoal => 1.10,
        LinkType::SameEvent => 1.10,
        LinkType::CoActivation => 1.05,
        LinkType::EntityOverlap => 1.00,
        LinkType::Elaboration => 1.00,
        LinkType::SemanticSimilar => 0.90,
        LinkType::TopicRelated => 0.85,
        LinkType::EmotionalResonance => 0.70,
        LinkType::TemporalAdjacent => 0.60,
        LinkType::Contradiction => 0.50,
        LinkType::Deprecated => 0.40,
    }
}
