//! acceptance test: DeterministicReranker

use hippmem_model::deterministic::rerank::DeterministicReranker;
use hippmem_model::traits::Reranker;

/// A candidate matching the query's terms scores higher.
#[test]
fn matching_candidate_scores_higher() {
    let r = DeterministicReranker;
    let scores = r
        .rerank_sync(
            "rust programming",
            &["hello world".into(), "rust is great".into()],
        )
        .unwrap();
    assert!(
        scores[1] > scores[0],
        "the candidate containing 'rust' should score higher"
    );
}

/// Scores are deterministic: running the same input repeatedly yields identical scores.
#[test]
fn scores_are_deterministic() {
    let r = DeterministicReranker;
    let cands: Vec<String> = vec!["hello world".into(), "rust lang".into()];
    let s1 = r.rerank_sync("rust programming", &cands).unwrap();
    let s2 = r.rerank_sync("rust programming", &cands).unwrap();
    assert_eq!(s1.len(), s2.len());
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

/// backend_id is correct.
#[test]
fn backend_id_correct() {
    let r = DeterministicReranker;
    assert_eq!(r.backend_id(), "deterministic-bm25-overlap");
}
