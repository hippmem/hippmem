//! Background consolidation Worker: periodically runs Hebbian / decay / compaction (09 §3.3).
//!
//! Summary creation (03 §8) is owned by the Engine layer: it has access to AlgoParams
//! and performs cluster planning + persistence. The worker only mutates units/edges.

use crate::decay::{apply_decay_with_protection, DecayParams};
use crate::hebbian::{hebbian_reinforce, reject_weaken, HebbianParams};
use hippmem_core::ids::MemoryId;
use hippmem_core::model::unit::MemoryUnit;
use hippmem_core::time::Timestamp;

/// Consolidation cycle statistics.
#[derive(Debug, Clone, Default)]
pub struct CycleStats {
    pub edges_decayed: u64,
    pub edges_archived: u64,
    pub hebbian_applied: u64,
}

/// Simple consolidation Worker (synchronous version, for tests and single-threaded use).
#[derive(Debug, Default)]
pub struct ConsolidationWorker {
    cycle_count: u64,
}

impl ConsolidationWorker {
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Runs one consolidation cycle:
    /// 1. Hebbian reinforcement (based on feedback co-activation pairs)
    /// 2. Reverse Hebbian (B4: weaken edges of explicitly rejected memories)
    /// 3. Decay (non-protected edges)
    /// 4. Compaction (weak-edge archiving)
    ///
    /// (Summary creation is owned by the Engine layer, 03 §8)
    pub fn run_cycle(
        &mut self,
        units: &mut [MemoryUnit],
        co_activations: &[(MemoryId, MemoryId, f32)],
        rejected_ids: &[MemoryId],
        now: Timestamp,
    ) -> CycleStats {
        let mut stats = CycleStats::default();
        let heb_params = HebbianParams::default();
        let decay_params = DecayParams::default();
        let comp_params = crate::compaction::CompactionParams::default();

        // 1. Hebbian reinforcement: apply co-activation reinforcement to each unit's out-edges
        let mut hebbian_count: u64 = 0;
        for unit in units.iter_mut() {
            let pre_count = unit.links.len();
            hebbian_reinforce(&mut unit.links, co_activations, &heb_params, now);
            // Count edges whose activation_count changed
            let changed = unit.links.iter().filter(|l| l.activation_count > 0).count() as u64;
            hebbian_count += changed;
            // For co-activated pairs without an edge, create a new CoActivation edge
            let new_links = crate::hebbian::build_coactivation_links(
                co_activations,
                heb_params.coactivation_threshold as f32,
                now,
            );
            for (owner_id, link) in new_links {
                if owner_id == unit.id {
                    // Idempotency: the same co-activated pair is rebuilt on every
                    // cycle; skip if an identical (target, type) edge already exists,
                    // otherwise edge counts accumulate unboundedly across cycles.
                    let already = unit
                        .links
                        .iter()
                        .any(|l| l.target_id == link.target_id && l.link_type == link.link_type);
                    if !already {
                        unit.links.push(link);
                    }
                }
            }
            // Preserve the links-count invariant (deduplication)
            let after_count = unit.links.len();
            if after_count > pre_count {
                hebbian_count += (after_count - pre_count) as u64;
            }
        }
        stats.hebbian_applied = hebbian_count;

        // 2. Reverse Hebbian (B4, 0.4.0): explicitly rejected memories have
        //    their association edges weakened — the graph-level counterpart of
        //    the removed global usage penalty. Counted into hebbian_applied.
        if !rejected_ids.is_empty() {
            let mut weakened: u64 = 0;
            for unit in units.iter_mut() {
                weakened +=
                    reject_weaken(unit.id, &mut unit.links, rejected_ids, &heb_params, now) as u64;
            }
            stats.hebbian_applied += weakened;
        }

        // 3. Decay: decay non-protected edges
        let mut decayed: u64 = 0;
        for unit in units.iter_mut() {
            let pre_len = unit.links.len();
            apply_decay_with_protection(&mut unit.links, &decay_params, now);
            let post_len = unit.links.len();
            if post_len < pre_len {
                decayed += (pre_len - post_len) as u64;
            }
        }
        stats.edges_decayed = decayed;

        // 3. Compaction: archive weak edges
        let mut archived: u64 = 0;
        for unit in units.iter_mut() {
            let links = std::mem::take(&mut unit.links);
            let (kept, archived_links) = crate::compaction::compact_edges(links, &comp_params);
            archived += archived_links.len() as u64;
            unit.links = kept;
        }
        stats.edges_archived = archived;

        self.cycle_count += 1;
        stats
    }
}
