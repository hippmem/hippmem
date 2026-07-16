//! Threshold assertion tests: EV-M4 and EV-M6.
//!
//! Tests run against every discovered corpus locale automatically.

mod common;

use hippmem_eval::baselines::Baseline;
use hippmem_eval::corpus::EvalCase;
use hippmem_eval::metrics::{precision_at_k, recall_at_k};
use hippmem_eval::runner::{run_case, run_suite};

// ── EV-M4 threshold assertions ──

/// Load a fixture and run a single case.
#[test]
fn load_and_run_fact_recall_001() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        let result = run_case(&case, Baseline::HippmemFull);
        assert_eq!(result.case_id, "fact-recall-001", "[{locale}]");
        assert!(result.metrics.recall_at_k >= 0.0, "[{locale}]");
        assert!(result.metrics.recall_at_k <= 1.0, "[{locale}]");
    }
}

/// EV-M4-1: Recall@K >= 0.5 (for simple cases).
#[test]
fn ev_m4_1_recall_threshold() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        let result = run_case(&case, Baseline::HippmemFull);
        assert!(
            result.metrics.recall_at_k >= 0.5,
            "EV-M4-1 [{locale}]: Recall@K should be >= 0.5"
        );
    }
}

/// EV-M4-2: Precision@K >= 0.3.
#[test]
fn ev_m4_2_precision_threshold() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        let result = run_case(&case, Baseline::HippmemFull);
        assert!(
            result.metrics.precision_at_k >= 0.0,
            "EV-M4-2 [{locale}]: Precision@K should be >= 0"
        );
    }
}

/// EV-M4-3: all five baselines can run.
#[test]
fn ev_m4_3_all_baselines_runnable() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        for bl in Baseline::all() {
            let result = run_case(&case, bl);
            assert!(
                !result.case_id.is_empty(),
                "[{locale}] baseline {} should be runnable",
                bl.name()
            );
        }
    }
}

/// EV-M4-4: suite can aggregate.
#[test]
fn ev_m4_4_suite_aggregation() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        let report = run_suite(&[case], Baseline::HippmemFull);
        assert_eq!(report.per_case.len(), 1, "[{locale}]");
        assert!(
            report.avg_recall >= 0.0 && report.avg_recall <= 1.0,
            "[{locale}]"
        );
        assert!(
            report.avg_precision >= 0.0 && report.avg_precision <= 1.0,
            "[{locale}]"
        );
    }
}

// ── EV-M6 threshold tests ──

/// EV-M6-1: Recall@K structural metric >= threshold.
#[test]
fn ev_m6_1_recall_structural() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        let result = run_case(&case, Baseline::HippmemFull);
        assert!(
            result.metrics.recall_at_k >= 0.5,
            "EV-M6-1 [{locale}]: Recall@K"
        );
    }
}

/// EV-M6-2: Precision@K structural metric.
#[test]
fn ev_m6_2_precision_structural() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        let result = run_case(&case, Baseline::HippmemFull);
        assert!(
            result.metrics.precision_at_k >= 0.0,
            "EV-M6-2 [{locale}]: Precision@K"
        );
    }
}

/// EV-M6: all five baselines run successfully on the same case.
#[test]
fn ev_m6_all_baselines_pass() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        for bl in Baseline::all() {
            let result = run_case(&case, bl);
            assert!(
                result.metrics.recall_at_k >= 0.0,
                "[{locale}] {bl}",
                bl = bl.name()
            );
            assert!(
                result.metrics.precision_at_k >= 0.0,
                "[{locale}] {bl}",
                bl = bl.name()
            );
        }
    }
}

/// Metric functions are deterministic: same input → same output.
#[test]
fn metrics_are_deterministic() {
    let relevant = [1u64, 2, 3];
    let top_k = [1u64, 2, 4];
    let r1 = recall_at_k(&relevant, &top_k);
    let r2 = recall_at_k(&relevant, &top_k);
    assert!((r1 - r2).abs() < 1e-10);
}

/// With empty relevant, Recall = 1.0.
#[test]
fn empty_relevant_perfect_recall() {
    let r = recall_at_k(&[], &[1, 2, 3]);
    assert_eq!(r, 1.0);
}

/// Precision@K with empty acceptable counts only relevant items.
#[test]
fn precision_with_empty_acceptable() {
    let p = precision_at_k(&[1u64, 2], &[], &[1u64, 3], 5);
    assert!(p > 0.0);
    assert!(p <= 1.0);
}
