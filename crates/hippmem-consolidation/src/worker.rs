//! Background consolidation Worker: periodically runs Hebbian / decay / compaction / summary (09 §3.3).

use crate::decay::{apply_decay_with_protection, DecayParams};
use crate::hebbian::{hebbian_reinforce, HebbianParams};
use crate::summarize::{build_summary_unit, should_summarize};
use hippmem_core::ids::MemoryId;
use hippmem_core::model::unit::MemoryUnit;
use hippmem_core::time::Timestamp;
use hippmem_model::deterministic::summarize::DeterministicSummarizer;

/// Consolidation cycle statistics.
#[derive(Debug, Clone, Default)]
pub struct CycleStats {
    pub edges_decayed: u64,
    pub edges_archived: u64,
    pub hebbian_applied: u64,
    pub summaries_created: u64,
    /// Summary memory unit (if triggered and created this cycle); the Engine layer is responsible for persisting it.
    pub summary_unit: Option<MemoryUnit>,
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
    /// 2. Decay (non-protected edges)
    /// 3. Compaction (weak-edge archiving)
    /// 4. Summary check + creation (Summarizer integration, 03 §8)
    pub fn run_cycle(
        &mut self,
        units: &mut [MemoryUnit],
        co_activations: &[(MemoryId, MemoryId, u32)],
        now: Timestamp,
        summarizer: Option<&DeterministicSummarizer>,
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
                heb_params.coactivation_threshold,
                now,
            );
            for (owner_id, link) in new_links {
                if owner_id == unit.id {
                    unit.links.push(link);
                }
            }
            // Preserve the links-count invariant (deduplication)
            let after_count = unit.links.len();
            if after_count > pre_count {
                hebbian_count += (after_count - pre_count) as u64;
            }
        }
        stats.hebbian_applied = hebbian_count;

        // 2. Decay: decay non-protected edges
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

        // 4. Summary check + creation (Summarizer integration, 03 §8)
        if let Some(summarizer) = summarizer {
            let ids: Vec<_> = units.iter().map(|u| u.id).collect();
            if should_summarize(&ids, 12) {
                let summary_unit = build_summary_unit(units, summarizer);
                // Confidence gating: low confidence (<0.35) does not create a summary (Constitution C7)
                if summary_unit.understanding.confidence.value() >= 0.35 {
                    stats.summaries_created = 1;
                    stats.summary_unit = Some(summary_unit);
                }
            }
        }

        self.cycle_count += 1;
        stats
    }
}
