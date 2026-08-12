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
        // D-B (0.4.0): result-set reject. An empty used_memory_ids with
        // UserRejected means "the whole result set was wrong" (trap questions,
        // noisy stores) — lower the usage score of every memory returned by
        // that retrieval. The result set is recovered from the retrieval's
        // own activation-log record (written by retrieve). A targeted reject
        // (non-empty used_memory_ids) keeps the stronger -0.10 per memory.
        if matches!(input.signal, crate::UsageSignal::UserRejected)
            && input.used_memory_ids.is_empty()
        {
            if let Ok(records) = logger.read_all() {
                if let Some(rec) = records
                    .iter()
                    .find(|r| r.retrieval_id == input.retrieval_id && r.signal == "retrieve")
                {
                    apply_delta(&rec.used_memory_ids, RESULT_SET_REJECT_DELTA)?;
                }
            }
        }

        Ok(())
    }
}

/// Usage-score adjustment per memory for a result-set reject (D-B, 0.4.0):
/// weaker than the targeted reject (-0.10) because no single memory was
/// singled out as wrong.
const RESULT_SET_REJECT_DELTA: f32 = -0.05;

fn signal_to_string(s: &crate::UsageSignal) -> String {
    match s {
        crate::UsageSignal::Referenced => "Referenced".into(),
        crate::UsageSignal::UserConfirmedCorrect => "UserConfirmedCorrect".into(),
        crate::UsageSignal::TaskSucceeded => "TaskSucceeded".into(),
        crate::UsageSignal::UserRejected => "UserRejected".into(),
    }
}
