//! Engine::feedback — usage feedback API.
//!
//! Corresponds to 05#feedback, 09 §4.6. Records usage signals to activation_log,
//! driving Hebbian reinforcement/decay (consumed by the consolidate worker).

use crate::{Engine, EngineError, EngineResult, FeedbackInput};
use hippmem_core::time::{Clock, SystemClock};
use hippmem_store::activation_log::{ActivationLogger, ActivationRecord};

impl Engine {
    /// Records a usage feedback signal.
    ///
    /// The signal is persisted to activation_log for consumption by the
    /// background Hebbian/decay worker.
    pub fn feedback(&self, input: FeedbackInput) -> EngineResult<()> {
        let clock = SystemClock;
        let now = clock.now();

        let logger = ActivationLogger::new(self.store.db_arc());
        let rec = ActivationRecord {
            retrieval_id: input.retrieval_id,
            used_memory_ids: input.used_memory_ids.iter().map(|id| id.0 as u64).collect(),
            signal: signal_to_string(&input.signal),
            recorded_at_ms: now.as_i64(),
        };
        logger
            .record(&rec)
            .map_err(|e| EngineError::Internal(format!("activation_log: {}", e)))?;
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
