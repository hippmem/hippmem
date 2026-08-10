//! Usage signal semantics: polarity filtering + usage_score deltas.
//!
//! activation_log records every signal verbatim (audit trail), but consumers
//! must distinguish positive signals from `UserRejected`:
//! - the RecentActivation channel and Hebbian co-activation pairs count positive
//!   signals only (05 §6: rejection must not strengthen memories);
//! - `usage_delta` drives `ActivationState.usage_score` (confirm ↑ / reject ↓).

/// Positive usage signals: internal retrieve records + feedback confirmations.
/// `UserRejected` (and unknown signals) are excluded.
pub(crate) fn is_positive_signal(signal: &str) -> bool {
    matches!(
        signal,
        "retrieve" | "Referenced" | "UserConfirmedCorrect" | "TaskSucceeded"
    )
}

/// usage_score delta per signal (03 §6 usage_signal values scaled by 0.1,
/// rejection is negative). Unknown signals → 0.0.
pub(crate) fn usage_delta(signal: &str) -> f32 {
    match signal {
        "UserConfirmedCorrect" => 0.10,
        "TaskSucceeded" => 0.08,
        "Referenced" => 0.05,
        "UserRejected" => -0.10,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_signals_whitelist() {
        for s in [
            "retrieve",
            "Referenced",
            "UserConfirmedCorrect",
            "TaskSucceeded",
        ] {
            assert!(is_positive_signal(s), "{s} 应为正信号");
        }
        assert!(!is_positive_signal("UserRejected"));
        assert!(!is_positive_signal("unknown_signal"));
    }

    #[test]
    fn usage_delta_values() {
        assert_eq!(usage_delta("UserConfirmedCorrect"), 0.10);
        assert_eq!(usage_delta("TaskSucceeded"), 0.08);
        assert_eq!(usage_delta("Referenced"), 0.05);
        assert_eq!(usage_delta("UserRejected"), -0.10);
        assert_eq!(usage_delta("retrieve"), 0.0);
        assert_eq!(usage_delta("bogus"), 0.0);
    }
}
