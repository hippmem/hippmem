//! Full corpus loading test: covers all ten task types.
//!
//! All cases live in `fixtures/corpus/<locale>/`. Tests discover locales
//! automatically — adding a new locale requires zero code changes.

mod common;

use hippmem_eval::corpus::EvalCase;

const FIXTURES: &[(&str, &str)] = &[
    ("fact-recall-001", "FactRecall"),
    ("causal-trace-001", "CausalTrace"),
    ("preference-recall-001", "PreferenceRecall"),
    ("project-continuity-001", "ProjectContinuity"),
    ("contradiction-detection-001", "ContradictionDetection"),
    ("state-change-001", "StateChange"),
    ("implicit-association-001", "ImplicitAssociation"),
    ("noise-resistance-001", "NoiseResistance"),
    ("long-tail-recall-001", "LongTailRecall"),
    ("explanation-quality-001", "ExplanationQuality"),
    // Additional corpus scenarios
    ("entity-network-001", "FactRecall"),
    ("causal-chain-001", "CausalTrace"),
    ("multi-dim-assoc-001", "ImplicitAssociation"),
];

/// All ten corpus types can be loaded. Runs for every discovered locale.
#[test]
fn all_ten_task_types_loadable() {
    for locale in common::discover_corpus_locales() {
        for (name, expected_type) in FIXTURES {
            let raw = common::load_corpus_case(&locale, name);
            let case: EvalCase = serde_json::from_str(&raw)
                .unwrap_or_else(|e| panic!("[{locale}] cannot parse {name}: {e}"));
            assert_eq!(
                case.task_type.to_string(),
                *expected_type,
                "[{locale}] {name}: task_type should be {expected_type}"
            );
        }
    }
}

/// Each task type has at least one case. Runs for every discovered locale.
#[test]
fn each_type_has_at_least_one_case() {
    for locale in common::discover_corpus_locales() {
        let mut types = std::collections::HashSet::new();
        for (name, _) in FIXTURES {
            let raw = common::load_corpus_case(&locale, name);
            let case: EvalCase = serde_json::from_str(&raw).unwrap();
            types.insert(case.task_type.to_string());
        }
        assert_eq!(types.len(), 10, "[{locale}] should have all 10 task types");
    }
}
