//! Eval metric computation: Recall@K, Precision@K, etc. (06 §5.1).

use std::collections::HashSet;

/// Eval metrics.
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub explanation_accuracy: f64,
    pub contradiction_awareness: f64,
}

/// Compute Recall@K = |relevant ∩ topK| / |relevant|.
pub fn recall_at_k(relevant: &[u64], top_k: &[u64]) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    let rel_set: HashSet<_> = relevant.iter().copied().collect();
    let hit = top_k.iter().filter(|id| rel_set.contains(id)).count();
    hit as f64 / relevant.len() as f64
}

/// Compute Precision@K = |(relevant ∪ acceptable) ∩ topK| / K.
pub fn precision_at_k(relevant: &[u64], also_acceptable: &[u64], top_k: &[u64], k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut allowed: HashSet<u64> = relevant.iter().copied().collect();
    allowed.extend(also_acceptable);
    let hit = top_k.iter().filter(|id| allowed.contains(id)).count();
    hit as f64 / k.min(top_k.len().max(1)) as f64
}

/// Compute explanation accuracy: fraction of dimensions hit.
pub fn explanation_accuracy(expected: &[String], actual: &[String]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let act_set: HashSet<_> = actual.iter().collect();
    let hit = expected.iter().filter(|dim| act_set.contains(*dim)).count();
    hit as f64 / expected.len() as f64
}

/// Compute contradiction awareness.
pub fn contradiction_awareness(
    expected_warnings: usize,
    _actual_warnings: usize,
    has_contradiction: bool,
) -> f64 {
    if expected_warnings == 0 {
        return 1.0;
    }
    if has_contradiction {
        1.0
    } else {
        0.0
    }
}
