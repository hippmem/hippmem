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
        if delta != 0.0 && !input.used_memory_ids.is_empty() {
            let kv = KvStore::new(self.store.db_arc());
            for id in &input.used_memory_ids {
                if let Some(raw) = kv
                    .get(&id.0)
                    .map_err(|e| EngineError::Store(e.to_string()))?
                {
                    let (mut unit, _): (MemoryUnit, _) =
                        bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                            .map_err(|e| EngineError::Internal(e.to_string()))?;
                    let new_usage = (unit.activation.usage_score.value() + delta).clamp(0.0, 1.0);
                    unit.activation.usage_score = UnitScore::new(new_usage);
                    let encoded = bincode::serde::encode_to_vec(&unit, bincode::config::standard())
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    kv.put(id.0, &encoded)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                }
            }
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
