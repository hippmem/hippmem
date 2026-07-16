//! acceptance test: eval corpus format types and single-case loading
//!
//! All locale-specific data lives in `fixtures/corpus/<locale>/` and
//! `tests/fixtures/corpus/<locale>.json`. Adding a new locale requires
//! zero code changes — tests discover locales automatically.

mod common;

use hippmem_eval::corpus::EvalCase;

/// Can deserialize an EvalCase JSON and verify fields against the corpus fixture.
/// Runs for every discovered corpus locale so both zh and en are covered.
#[test]
fn deserialize_fact_recall_001() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase =
            serde_json::from_str(&raw).expect("should be able to deserialize EvalCase");

        let corpus = common::load_test_fixture("corpus", &locale);

        // Basic fields
        assert_eq!(case.case_id, "fact-recall-001", "[{locale}] case_id");
        assert_eq!(
            case.task_type.to_string(),
            "FactRecall",
            "[{locale}] task_type"
        );

        // writes
        assert_eq!(case.writes.len(), 2, "[{locale}] writes len");
        assert_eq!(case.writes[0].local_id, "m1", "[{locale}] local_id");
        assert_eq!(
            case.writes[0].content,
            corpus["writes"][0].as_str().unwrap(),
            "[{locale}] write content"
        );
        assert!(case.writes[0].context.session_id == Some(1));

        // query
        assert_eq!(
            case.query.text,
            corpus["queries"][0].as_str().unwrap(),
            "[{locale}] query text"
        );
        assert_eq!(case.query.top_k, 10);

        // ground_truth
        assert_eq!(
            case.ground_truth.relevant,
            vec!["m2"],
            "[{locale}] relevant"
        );
        assert_eq!(
            case.ground_truth.also_acceptable,
            vec!["m1"],
            "[{locale}] also_acceptable"
        );
        assert_eq!(case.ground_truth.expected_warnings.len(), 0);
        assert_eq!(case.ground_truth.expected_edges.len(), 1);
        assert_eq!(case.ground_truth.expected_edges[0].from, "m2");
        assert_eq!(case.ground_truth.expected_edges[0].to, "m1");
    }
}

/// Verify GroundTruth fields exist and are accessible.
#[test]
fn ground_truth_fields_accessible() {
    let json = r#"{
        "case_id": "test-01",
        "task_type": "CausalTrace",
        "writes": [],
        "query": { "text": "test", "mode": "Fast", "top_k": 5, "context": {} },
        "ground_truth": {
            "relevant": ["a"],
            "also_acceptable": ["b"],
            "expected_dimensions": ["Causal"],
            "expected_warnings": ["HasContradiction"],
            "expected_edges": [{"from": "x", "to": "y", "link_type": "Causal"}]
        }
    }"#;
    let case: EvalCase = serde_json::from_str(json).expect("should be able to deserialize");

    assert_eq!(case.case_id, "test-01");
    assert_eq!(case.ground_truth.expected_dimensions.len(), 1);
    assert_eq!(case.ground_truth.expected_warnings.len(), 1);
}

/// Verify all ten TaskType variants can be deserialized.
#[test]
fn task_type_all_variants_deserializable() {
    let types = [
        "FactRecall",
        "PreferenceRecall",
        "ProjectContinuity",
        "CausalTrace",
        "ContradictionDetection",
        "StateChange",
        "ImplicitAssociation",
        "NoiseResistance",
        "LongTailRecall",
        "ExplanationQuality",
    ];
    for t in &types {
        let json = format!(
            r#"{{"case_id":"t","task_type":"{}","writes":[],"query":{{"text":"x","mode":"Fast","top_k":1,"context":{{}}}},"ground_truth":{{"relevant":[],"also_acceptable":[],"expected_dimensions":[],"expected_warnings":[],"expected_edges":[]}}}}"#,
            t
        );
        let case: EvalCase = serde_json::from_str(&json).unwrap_or_else(|e| panic!("{t}: {e}"));
        assert_eq!(case.task_type.to_string(), *t);
    }
}

/// EvalCase can round-trip through JSON. Runs for every discovered corpus locale.
#[test]
fn evalcase_roundtrip_json() {
    for locale in common::discover_corpus_locales() {
        let raw = common::load_corpus_case(&locale, "fact-recall-001");
        let case: EvalCase = serde_json::from_str(&raw).unwrap();
        let re_serialized = serde_json::to_string_pretty(&case).unwrap();
        let case2: EvalCase = serde_json::from_str(&re_serialized).unwrap();
        assert_eq!(case.case_id, case2.case_id, "[{locale}]");
        assert_eq!(case.writes.len(), case2.writes.len(), "[{locale}]");
    }
}
