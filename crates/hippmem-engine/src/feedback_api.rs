//! Engine::feedback — usage feedback API.
//!
//! Corresponds to 05#feedback, 09 §4.6. Records usage signals to activation_log
//! (driving Hebbian reinforcement/decay, consumed by consolidate) and updates each
//! memory's `usage_score` (0.3.0: confirmations/references raise it, rejections
//! lower it — retrieval energy is weighted by usage_score).

use crate::signals::usage_delta;
use crate::{Engine, EngineError, EngineResult, FeedbackInput};
use hippmem_core::model::unit::MemoryUnit;
use hippmem_core::score::UnitScore;
use hippmem_core::time::{Clock, SystemClock};
use hippmem_store::activation_log::{ActivationLogger, ActivationRecord};
use hippmem_store::kv::KvStore;

impl Engine {
    /// Records a usage feedback signal and updates usage_score of the referenced memories.
    ///
    /// The signal is persisted to activation_log for consumption by the
    /// background Hebbian/decay worker; usage_score changes take effect on the
    /// next retrieval (energy formula, 03 §4).
    pub fn feedback(&self, input: FeedbackInput) -> EngineResult<()> {
        let clock = SystemClock;
        let now = clock.now();

        let logger = ActivationLogger::new(self.store.db_arc());
        let signal = signal_to_string(&input.signal);
        let rec = ActivationRecord {
            retrieval_id: input.retrieval_id,
            // P1 回归：保留完整 u128 id（截断会导致下游 Hebbian/RecentActivation 失配）
            used_memory_ids: input.used_memory_ids.iter().map(|id| id.0).collect(),
            signal: signal.clone(),
            recorded_at_ms: now.as_i64(),
        };
        logger
            .record(&rec)
            .map_err(|e| EngineError::Internal(format!("activation_log: {}", e)))?;

        // 0.3.0: usage_score 更新（确认 ↑ / 拒绝 ↓，clamp [0,1]）
        let delta = usage_delta(&signal);
        let apply_delta = |ids: &[u128], amount: f32| -> EngineResult<()> {
            let kv = KvStore::new(self.store.db_arc());
            for id in ids {
                if let Some(raw) = kv.get(id).map_err(|e| EngineError::Store(e.to_string()))? {
                    let (mut unit, _): (MemoryUnit, _) =
                        bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                            .map_err(|e| EngineError::Internal(e.to_string()))?;
                    let new_usage = (unit.activation.usage_score.value() + amount).clamp(0.0, 1.0);
                    unit.activation.usage_score = UnitScore::new(new_usage);
                    let encoded = bincode::serde::encode_to_vec(&unit, bincode::config::standard())
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    kv.put(*id, &encoded)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                }
            }
            Ok(())
        };
        if delta != 0.0 && !input.used_memory_ids.is_empty() {
            let ids: Vec<u128> = input.used_memory_ids.iter().map(|id| id.0).collect();
            apply_delta(&ids, delta)?;
        }

        // 0.4.3 memory-learning-mechanism: context-answer links. A positive
        // confirmation binds each used memory to the query's context
        // fingerprint (recovered via retrieval_id, written at retrieve time) —
        // the lift is scoped to similar queries instead of global heat. Old
        // retrievals (no fingerprint) degrade to no context links.
        if crate::signals::is_positive_signal(&signal) && !input.used_memory_ids.is_empty() {
            use hippmem_store::context_links::{
                read_query_context, read_retrieval_paths, strengthen_links,
            };
            if let Some(ctx) = read_query_context(self.store.db_arc(), input.retrieval_id) {
                for id in &input.used_memory_ids {
                    let _ = strengthen_links(
                        self.store.db_arc(),
                        &ctx,
                        id.0,
                        crate::CONTEXT_LINK_DELTA,
                        now.as_i64(),
                    );
                }
            }
            // Path reinforcement (memory-learning-mechanism 0.4.3): for a
            // non-seed answer (reached through the graph), strengthen the
            // guide→answer edge that carried it — the bridge that let this
            // query reach the answer. Seeds (no path) get only the context
            // links above.
            let paths = read_retrieval_paths(self.store.db_arc(), input.retrieval_id);
            if !paths.is_empty() {
                for path in &paths {
                    if input.used_memory_ids.iter().any(|id| id.0 == path.to) {
                        let _ = self.strengthen_edge(path.from, path.to, crate::CONTEXT_LINK_DELTA);
                    }
                }
            }
        }
        // 0.4.1: an empty used_memory_ids + UserRejected is a *retrieval-quality*
        // signal, not a *memory-quality* signal, and has no memory-side effects.
        // The 0.4.0 result-set reject (D-B) lowered every memory returned by the
        // retrieval and permanently removed them from the recent channel; trap
        // questions (no answer in the store) trigger it by construction, so it
        // permanently suppressed innocent memories that happened to be retrieved
        // (2026-08-12 test report O1). Removed; only the activation-log record
        // remains (audit trail).

        Ok(())
    }

    /// Strengthens the guide→answer edge (memory-learning-mechanism, 0.4.3):
    /// the edge that carried activation from the guide to the answer during
    /// retrieval. Both the authoritative MemoryUnit links and the graph
    /// overlay are updated (consolidate later reads edges from the unit, so
    /// both must move together). No-op when the edge does not exist (stale
    /// path record); never creates edges.
    fn strengthen_edge(&self, from: u128, to: u128, delta: f32) -> EngineResult<()> {
        use hippmem_core::ids::MemoryId;
        use hippmem_core::score::UnitScore;
        let now = SystemClock.now();
        let boost = |s: f32| UnitScore::new((s + delta).min(1.0));
        // Review bookkeeping (forgetting-curve model, 0.4.3): each confirmed
        // path is a review — it resets the decay clock (last_activated_at)
        // and doubles the edge's half-life (activation_count).
        let review = |link: &mut hippmem_core::model::links::AssociationLink| {
            link.strength = boost(link.strength.value());
            link.last_activated_at = Some(now);
            link.activation_count = link.activation_count.saturating_add(1);
        };

        // 1) Authoritative unit links (what consolidate reads).
        let kv = KvStore::new(self.store.db_arc());
        if let Some(raw) = kv
            .get(&from)
            .map_err(|e| EngineError::Store(e.to_string()))?
        {
            let (mut unit, _): (MemoryUnit, _) =
                bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
            let mut changed = false;
            for link in unit.links.iter_mut() {
                if link.target_id.0 == to {
                    review(link);
                    changed = true;
                }
            }
            if changed {
                let encoded = bincode::serde::encode_to_vec(&unit, bincode::config::standard())
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                kv.put(from, &encoded)
                    .map_err(|e| EngineError::Store(e.to_string()))?;
            }
        }

        // 2) Graph overlay (what retrieval reads).
        let graph = hippmem_store::graph::GraphStore::new(self.store.db_arc());
        let mut links = graph
            .get_outgoing(&MemoryId(from))
            .map_err(EngineError::Store)?;
        let mut changed = false;
        for link in links.iter_mut() {
            if link.target_id.0 == to {
                review(link);
                changed = true;
            }
        }
        if changed {
            graph
                .put_outgoing(MemoryId(from), &links)
                .map_err(EngineError::Store)?;
        }
        Ok(())
    }
}

fn signal_to_string(s: &crate::UsageSignal) -> String {
    match s {
        crate::UsageSignal::Referenced => "Referenced".into(),
        crate::UsageSignal::UserConfirmedCorrect => "UserConfirmedCorrect".into(),
        crate::UsageSignal::TaskSucceeded => "TaskSucceeded".into(),
        crate::UsageSignal::UserRejected => "UserRejected".into(),
    }
}
