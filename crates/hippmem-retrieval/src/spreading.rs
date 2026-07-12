//! Spreading activation traversal: starts from seeds and propagates energy along association edges.
//!
//! Corresponds to 03 §4.2-4.3. This module works with [`energy`] to perform activation spreading.

use crate::energy;
use crate::seeds::{rrf_fuse, Seed};
use hippmem_core::config::AlgoParams;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{ActivationStep, AssociationLink, RecallChannel};
use std::collections::{HashMap, HashSet};

/// Single-hop spreading: starts from seeds and propagates energy along each seed's single strongest outgoing edge.
///
/// Returns all reached nodes (MemoryId, final energy, activation trace).
/// Includes the seed itself (hop=0) and one-hop neighbors (hop=1).
/// Neighbors with energy < min_propagation_energy are excluded.
///
/// Multi-seed merge strategy: multi-channel seeds for the same MemoryId are fused via merge_energy
/// (max + merge_secondary_weight × min), so multi-channel consensus rewards accumulate naturally.
pub fn spread_one_hop_fused(
    fused_scores: &HashMap<MemoryId, (f32, RecallChannel)>,
    links_map: &HashMap<MemoryId, Vec<AssociationLink>>,
    params: &AlgoParams,
    importance_map: &HashMap<MemoryId, f32>,
) -> Vec<(MemoryId, f32, Vec<ActivationStep>)> {
    // V9: RRF fused score → seed energy
    let max_fused: f32 = fused_scores
        .values()
        .map(|(s, _)| *s)
        .fold(0.0f32, f32::max);
    let mut seed_energies: Vec<(MemoryId, f32, RecallChannel)> = Vec::new();
    for (seed_id, (fused_score, seed_channel)) in fused_scores.iter() {
        let imp = importance_map.get(seed_id).copied().unwrap_or(0.0);
        let norm_score = if max_fused > 0.0 {
            fused_score / max_fused
        } else {
            0.0
        };
        let seed_energy = (norm_score * params.a_query_match * (1.0 + imp * params.c_importance))
            .min(params.seed_energy_cap);
        if seed_energy >= params.min_propagation_energy {
            seed_energies.push((*seed_id, seed_energy, *seed_channel));
        }
    }

    // Phase 2: build results and propagate from seeds
    let mut results: Vec<(MemoryId, f32, Vec<ActivationStep>)> = Vec::new();
    let mut seen: HashSet<MemoryId> = HashSet::new();

    seed_energies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (seed_id, seed_energy, seed_channel) in &seed_energies {
        // Seed itself
        if seen.insert(*seed_id) {
            let trace = vec![ActivationStep {
                from: None,
                to: *seed_id,
                via_link: None,
                channel: Some(*seed_channel),
                hop: 0,
                energy_in: *seed_energy,
                energy_out: *seed_energy,
            }];
            results.push((*seed_id, *seed_energy, trace));
        }

        // One-hop neighbors
        if let Some(links) = links_map.get(seed_id) {
            let mut sorted_links: Vec<&AssociationLink> = links.iter().collect();
            sorted_links.sort_by(|a, b| {
                let sa = a.strength.value() * a.confidence.value();
                let sb = b.strength.value() * b.confidence.value();
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_links: Vec<&AssociationLink> = sorted_links
                .iter()
                .take(params.fan_out_default as usize)
                .copied()
                .collect();

            for link in &top_links {
                let neighbor_energy = energy::propagated_energy(*seed_energy, link, 1, params);

                if neighbor_energy < params.min_propagation_energy {
                    continue;
                }

                if seen.insert(link.target_id) {
                    let trace = vec![
                        ActivationStep {
                            from: None,
                            to: *seed_id,
                            via_link: None,
                            channel: Some(*seed_channel),
                            hop: 0,
                            energy_in: *seed_energy,
                            energy_out: *seed_energy,
                        },
                        ActivationStep {
                            from: Some(*seed_id),
                            to: link.target_id,
                            via_link: Some(link.link_type),
                            channel: None,
                            hop: 1,
                            energy_in: neighbor_energy,
                            energy_out: neighbor_energy,
                        },
                    ];
                    results.push((link.target_id, neighbor_energy, trace));
                }
            }
        }
    }
    // Sort by energy descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Multi-hop spreading: starts from RRF-fused seeds and propagates energy along association edges over multiple hops (03 §4.3).
///
/// Returns all reached nodes (MemoryId, final energy, activation trace).
/// Implements cycle elimination (expanded set), fan-out pruning, energy threshold filtering,
/// multi-path merge (max + merge_secondary_weight * min),
/// and stopping conditions (max_hops / empty frontier / energy exhausted).
///
/// V9: Seed fusion is already done by the caller via RRF; each MemoryId in `fused_scores` has a single fused score.
/// The internal seed_best/merge_energy deduplication logic is no longer needed.
pub fn spread_multi_hop_fused(
    fused_scores: &HashMap<MemoryId, (f32, RecallChannel)>,
    links_map: &HashMap<MemoryId, Vec<AssociationLink>>,
    params: &AlgoParams,
    importance_map: &HashMap<MemoryId, f32>,
) -> Vec<(MemoryId, f32, Vec<ActivationStep>)> {
    // Set of expanded nodes (cycle elimination: a node is not expanded twice)
    let mut expanded: HashSet<MemoryId> = HashSet::new();
    // Final energy map (multi-path merge)
    let mut energy_map: HashMap<MemoryId, f32> = HashMap::new();
    // Result list (node + energy + activation trace)
    let mut results: Vec<(MemoryId, f32, Vec<ActivationStep>)> = Vec::new();
    // Activation trace map (keeps the first/best path's trace)
    let mut trace_map: HashMap<MemoryId, Vec<ActivationStep>> = HashMap::new();

    // Initialize frontier: RRF fused score is normalized then assigned as energy (V9)
    let mut frontier: Vec<(MemoryId, f32)> = Vec::new();
    // Compute max fused score for normalization
    let max_fused: f32 = fused_scores
        .values()
        .map(|(s, _)| *s)
        .fold(0.0f32, f32::max);
    for (seed_id, (fused_score, seed_channel)) in fused_scores.iter() {
        let imp = importance_map.get(seed_id).copied().unwrap_or(0.0);
        let norm_score = if max_fused > 0.0 {
            fused_score / max_fused
        } else {
            0.0
        };
        let seed_energy = (norm_score * params.a_query_match * (1.0 + imp * params.c_importance))
            .min(params.seed_energy_cap);
        if seed_energy < params.min_propagation_energy {
            continue;
        }
        if expanded.insert(*seed_id) {
            energy_map.insert(*seed_id, seed_energy);
            let trace = vec![ActivationStep {
                from: None,
                to: *seed_id,
                via_link: None,
                channel: Some(*seed_channel),
                hop: 0,
                energy_in: seed_energy,
                energy_out: seed_energy,
            }];
            trace_map.insert(*seed_id, trace);
            results.push((*seed_id, seed_energy, trace_map[seed_id].clone()));
            frontier.push((*seed_id, seed_energy));
        }
    }

    let max_hops = params.max_hops_default;

    for hop in 1..=max_hops {
        if frontier.is_empty() {
            break; // frontier empty, stop
        }

        let mut next_frontier: Vec<(MemoryId, f32)> = Vec::new();
        // Newly reached nodes this round (for merge + dedup)
        let mut round_energy: HashMap<MemoryId, f32> = HashMap::new();
        let mut round_traces: HashMap<MemoryId, Vec<ActivationStep>> = HashMap::new();

        for (node_id, node_energy) in &frontier {
            // Take the node's outgoing edges
            if let Some(links) = links_map.get(node_id) {
                // Take the top fan_out_default edges by strength*confidence descending
                let mut sorted: Vec<&AssociationLink> = links.iter().collect();
                sorted.sort_by(|a, b| {
                    let sa = a.strength.value() * a.confidence.value();
                    let sb = b.strength.value() * b.confidence.value();
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top_links = sorted.iter().take(params.fan_out_default as usize).copied();

                for link in top_links {
                    // Skip already-expanded nodes (cycle elimination)
                    if expanded.contains(&link.target_id) {
                        continue;
                    }

                    let propagated = energy::propagated_energy(*node_energy, link, hop, params);

                    // Energy threshold pruning
                    if propagated < params.min_propagation_energy {
                        continue;
                    }

                    // Multi-path energy merge: merge(existing, new)
                    let merged = match round_energy.get(&link.target_id) {
                        Some(&existing) => merge_energy(existing, propagated, params),
                        None => propagated,
                    };

                    round_energy.insert(link.target_id, merged);

                    // Build activation trace: take source node's trace + append new step
                    let source_trace = trace_map.get(node_id).cloned().unwrap_or_default();
                    let mut new_trace = source_trace;
                    new_trace.push(ActivationStep {
                        from: Some(*node_id),
                        to: link.target_id,
                        via_link: Some(link.link_type),
                        channel: None,
                        hop: hop as u8,
                        energy_in: propagated,
                        energy_out: propagated,
                    });
                    // Keep the trace of the highest-energy path
                    round_traces
                        .entry(link.target_id)
                        .and_modify(|existing_trace| {
                            // If the new energy is larger, replace the trace
                            if merged == propagated {
                                *existing_trace = new_trace.clone();
                            }
                        })
                        .or_insert(new_trace);
                }
            }
        }

        // Add this round's new nodes to results (dedup: those not in expanded)
        for (target_id, merged_energy) in round_energy {
            if expanded.insert(target_id) {
                energy_map.insert(target_id, merged_energy);
                let trace = round_traces.remove(&target_id).unwrap_or_default();
                trace_map.insert(target_id, trace.clone());
                results.push((target_id, merged_energy, trace));
                next_frontier.push((target_id, merged_energy));
            }
        }

        frontier = next_frontier;
    }

    // Sort by energy descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Multi-path energy merge function (03 §4.3).
///
/// ```text
/// merge(existing, new) = max(existing, new)
///                      + merge_secondary_weight * min(existing, new)
/// ```
fn merge_energy(existing: f32, new: f32, params: &AlgoParams) -> f32 {
    let max_e = existing.max(new);
    let min_e = existing.min(new);
    max_e + params.merge_secondary_weight * min_e
}

// ── V9 backward-compatible wrapper ──
// Legacy &[Seed] signature: performs RRF fusion internally then delegates to the _fused version.
// Kept for tests and legacy callers.

/// Multi-hop spreading (V8-compatible signature): starts from &[Seed].
/// Performs RRF fusion (k=rrf_k) internally then delegates to [`spread_multi_hop_fused`].
pub fn spread_multi_hop(
    seeds: &[Seed],
    links_map: &HashMap<MemoryId, Vec<AssociationLink>>,
    params: &AlgoParams,
    importance_map: &HashMap<MemoryId, f32>,
) -> Vec<(MemoryId, f32, Vec<ActivationStep>)> {
    // V9: if seeds have rank_in_channel (production path) → RRF fusion; otherwise (legacy tests) → use score directly
    let has_ranks = seeds.iter().any(|s| s.rank_in_channel.is_some());
    let fused = if has_ranks {
        rrf_fuse(seeds, params)
    } else {
        seeds.iter().map(|s| (s.id, (s.score, s.channel))).collect()
    };
    spread_multi_hop_fused(&fused, links_map, params, importance_map)
}

/// Single-hop spreading (V8-compatible signature).
pub fn spread_one_hop(
    seeds: &[Seed],
    links_map: &HashMap<MemoryId, Vec<AssociationLink>>,
    params: &AlgoParams,
    importance_map: &HashMap<MemoryId, f32>,
) -> Vec<(MemoryId, f32, Vec<ActivationStep>)> {
    let has_ranks = seeds.iter().any(|s| s.rank_in_channel.is_some());
    let fused = if has_ranks {
        rrf_fuse(seeds, params)
    } else {
        seeds.iter().map(|s| (s.id, (s.score, s.channel))).collect()
    };
    spread_one_hop_fused(&fused, links_map, params, importance_map)
}
