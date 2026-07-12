//! Reranking: applies rule-based or model-based reranking to spreading results (03 §4.4).
//!
//! The fallback backend uses rule-based reranking (deterministic); the API backend uses the Reranker trait.

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::ActivationStep;
use hippmem_core::model::unit::MemoryUnit;

/// Rule-based reranking: sort by energy, plus context-match bonus.
///
/// Used by the fallback backend; no external model dependency, deterministic.
pub fn rerank_by_energy(
    activated: &[(MemoryId, f32, Vec<ActivationStep>)],
    units: &[MemoryUnit],
) -> Vec<(MemoryId, f32, Vec<ActivationStep>, MemoryUnit)> {
    let mut scored: Vec<_> = activated
        .iter()
        .filter_map(|(id, energy, trace)| {
            let unit = units.iter().find(|u| u.id == *id)?;
            Some((*id, *energy, trace.clone(), unit.clone()))
        })
        .collect();

    // Sort by energy descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}
