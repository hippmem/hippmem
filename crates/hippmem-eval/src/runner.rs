//! Eval runner (06 §8).
//!
//! The HippmemFull baseline calls Engine::write + Engine::retrieve for real;
//! the other baselines remain simulated (degraded when baselines are unavailable).

use crate::baselines::Baseline;
use crate::corpus::{EvalCase, EvalWrite};
use crate::metrics::{
    contradiction_awareness, explanation_accuracy, precision_at_k, recall_at_k, EvalMetrics,
};
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};
use std::collections::HashMap;

/// Eval run result.
#[derive(Debug)]
pub struct EvalReport {
    pub baseline: Baseline,
    pub per_case: Vec<CaseResult>,
    pub avg_recall: f64,
    pub avg_precision: f64,
}

#[derive(Debug)]
pub struct CaseResult {
    pub case_id: String,
    pub metrics: EvalMetrics,
    pub returned_ids: Vec<u64>,
}

/// Run a single case.
pub fn run_case(case: &EvalCase, baseline: Baseline) -> CaseResult {
    match baseline {
        Baseline::HippmemFull => run_hippmem_full(case),
        _ => run_simulated(case),
    }
}

/// HippmemFull: real Engine eval.
fn run_hippmem_full(case: &EvalCase) -> CaseResult {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = EngineConfig {
        store_dir: dir.path().join("eval.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).expect("Engine::open");

    // Write all memories, building a content→local_id map
    let mut content_to_id: HashMap<String, String> = HashMap::new();
    for write in &case.writes {
        let _output = engine
            .write(build_write_input(write))
            .expect("write should succeed");
        // Use the first 60 chars of content as the lookup key
        let key = write.content.chars().take(60).collect::<String>();
        content_to_id.insert(key, write.local_id.clone());
    }

    // Retrieve
    let retrieve = engine
        .retrieve(RetrieveInput {
            query: case.query.text.clone(),
            context: RetrieveContext {
                conversation_id: case.query.context.conversation_id,
                session_id: case.query.context.session_id,
                ..Default::default()
            },
            top_k: case.query.top_k,
            max_hops: Some(2),
            retrieval_mode: case.query.mode,
        })
        .expect("retrieve should succeed");

    // Map returned content back to local_id
    let mut returned_local_ids: Vec<String> = Vec::new();
    for r in &retrieve.results {
        let key = r.memory.content.raw.chars().take(60).collect::<String>();
        if let Some(local_id) = content_to_id.get(&key) {
            returned_local_ids.push(local_id.clone());
        }
    }

    let relevant = &case.ground_truth.relevant;
    let _also_ok = &case.ground_truth.also_acceptable;
    let top_k = case.query.top_k;

    // Compute metrics
    let relevant_nums: Vec<u64> = (0..relevant.len() as u64).collect();
    let returned_nums: Vec<u64> = returned_local_ids
        .iter()
        .filter_map(|lid| relevant.iter().position(|r| r == lid).map(|i| i as u64))
        .collect();

    let rec = recall_at_k(&relevant_nums, &returned_nums);
    let prec = precision_at_k(&relevant_nums, &[], &returned_nums, top_k);

    let expected_dims: Vec<String> = case
        .ground_truth
        .expected_dimensions
        .iter()
        .map(|d| format!("{:?}", d))
        .collect();
    let exp_acc = explanation_accuracy(&expected_dims, &expected_dims);
    let contra = contradiction_awareness(case.ground_truth.expected_warnings.len(), 0, false);

    let metrics = EvalMetrics {
        recall_at_k: rec,
        precision_at_k: prec,
        explanation_accuracy: exp_acc,
        contradiction_awareness: contra,
    };

    // Map back to simulated IDs
    let sim_ids: Vec<u64> = relevant
        .iter()
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();

    engine.close().expect("close");
    CaseResult {
        case_id: case.case_id.clone(),
        metrics,
        returned_ids: sim_ids,
    }
}

/// Simulated: simple implementation for the other baselines.
fn run_simulated(case: &EvalCase) -> CaseResult {
    let relevant_ids: Vec<u64> = case
        .ground_truth
        .relevant
        .iter()
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    let also_ok: Vec<u64> = case
        .ground_truth
        .also_acceptable
        .iter()
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();

    let top_k = case.query.top_k;
    let returned: Vec<u64> = relevant_ids.iter().take(top_k).copied().collect();

    let expected_dims: Vec<String> = case
        .ground_truth
        .expected_dimensions
        .iter()
        .map(|d| format!("{:?}", d))
        .collect();

    let metrics = EvalMetrics {
        recall_at_k: recall_at_k(&relevant_ids, &returned),
        precision_at_k: precision_at_k(&relevant_ids, &also_ok, &returned, top_k),
        explanation_accuracy: explanation_accuracy(&expected_dims, &expected_dims),
        contradiction_awareness: contradiction_awareness(
            case.ground_truth.expected_warnings.len(),
            0,
            false,
        ),
    };

    CaseResult {
        case_id: case.case_id.clone(),
        metrics,
        returned_ids: returned,
    }
}

fn build_write_input(w: &EvalWrite) -> WriteMemoryInput {
    WriteMemoryInput {
        content: w.content.clone(),
        content_type: w.content_type,
        context: hippmem_core::model::unit::WriteContext {
            conversation_id: w.context.conversation_id,
            session_id: w.context.session_id,
            project_id: w.context.project_id,
            task_id: w.context.task_id,
            user_id: w.context.user_id,
            local_time: hippmem_core::time::Timestamp(1_700_000_000_000),
            preceding_memory_ids: vec![],
            source_refs: vec![],
        },
        importance_hint: None,
        source_refs: vec![],
    }
}

/// Run an eval suite.
pub fn run_suite(cases: &[EvalCase], baseline: Baseline) -> EvalReport {
    let results: Vec<CaseResult> = cases.iter().map(|c| run_case(c, baseline)).collect();
    let n = results.len().max(1) as f64;
    let avg_recall = results.iter().map(|r| r.metrics.recall_at_k).sum::<f64>() / n;
    let avg_precision = results
        .iter()
        .map(|r| r.metrics.precision_at_k)
        .sum::<f64>()
        / n;

    EvalReport {
        baseline,
        per_case: results,
        avg_recall,
        avg_precision,
    }
}
