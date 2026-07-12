//! Explain + Inspect API (M6-002): retrieval-result diagnostics and
//! explanation.

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{ActivationStep, MatchDimension, MemoryWarning};
use hippmem_core::model::unit::MemoryUnit;

/// Explanation output: answers "why was this memory recalled".
#[derive(Debug, Clone)]
pub struct Explanation {
    pub memory_id: MemoryId,
    /// Activation trace.
    pub trace: Vec<ActivationStep>,
    /// Dimensions that hit.
    pub dimensions: Vec<MatchDimension>,
    /// Risk warnings.
    pub warnings: Vec<MemoryWarning>,
    /// Text explanation.
    pub summary: String,
}

/// Generate an explanation for a retrieval result.
pub fn explain_result(
    unit: &MemoryUnit,
    trace: &[ActivationStep],
    dimensions: &[MatchDimension],
    warnings: &[MemoryWarning],
) -> Explanation {
    let dim_strs: Vec<String> = dimensions.iter().map(|d| format!("{:?}", d)).collect();
    let warn_strs: Vec<String> = warnings.iter().map(|w| format!("{:?}", w)).collect();

    let summary = format!(
        "Memory {} was recalled via {:?} dimension(s) ({}), hops={}. Warnings: {}.",
        unit.id.as_u128(),
        dimensions.len(),
        dim_strs.join(", "),
        trace.iter().map(|s| s.hop).max().unwrap_or(0),
        if warn_strs.is_empty() {
            "none".to_string()
        } else {
            warn_strs.join(", ")
        },
    );

    Explanation {
        memory_id: unit.id,
        trace: trace.to_vec(),
        dimensions: dimensions.to_vec(),
        warnings: warnings.to_vec(),
        summary,
    }
}

/// Inspect query: diagnose store state.
#[derive(Debug)]
pub struct InspectReport {
    pub memory_count: usize,
    pub link_count: usize,
    pub stage_distribution: Vec<(String, usize)>,
}

/// Generate an inspect report.
pub fn inspect_units(units: &[MemoryUnit]) -> InspectReport {
    let memory_count = units.len();
    let link_count: usize = units.iter().map(|u| u.links.len()).sum();
    let mut stage_dist: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for unit in units {
        let key = format!("{:?}", unit.stage);
        *stage_dist.entry(key).or_insert(0) += 1;
    }
    let stage_distribution: Vec<(String, usize)> = stage_dist.into_iter().collect();

    InspectReport {
        memory_count,
        link_count,
        stage_distribution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explanation_generates_summary() {
        let unit = MemoryUnit {
            schema_version: 1,
            id: MemoryId(42),
            created_at: hippmem_core::time::Timestamp(0),
            updated_at: hippmem_core::time::Timestamp(0),
            content: hippmem_core::model::unit::MemoryContent {
                raw: "test".into(),
                summary: None,
                normalized: None,
                language: hippmem_core::model::unit::Language::Zh,
                content_type: hippmem_core::model::unit::ContentType::UserStatement,
            },
            context: hippmem_core::model::unit::WriteContext {
                conversation_id: None,
                session_id: None,
                project_id: None,
                task_id: None,
                user_id: None,
                local_time: hippmem_core::time::Timestamp(0),
                preceding_memory_ids: vec![],
                source_refs: vec![],
            },
            understanding: hippmem_core::model::understanding::MemoryUnderstanding {
                entities: vec![],
                events: vec![],
                goals: vec![],
                decisions: vec![],
                preferences: vec![],
                emotions: vec![],
                causal_claims: vec![],
                contradictions: vec![],
                topics: vec![],
                importance: hippmem_core::score::UnitScore::new(0.5),
                confidence: hippmem_core::score::UnitScore::new(0.5),
            },
            association_keys: hippmem_core::model::links::AssociationKeys {
                entity_keys: vec![],
                temporal_keys: vec![],
                lexical_signature: hippmem_core::model::links::LexicalSignature { simhash: [0; 4] },
                semantic_signature: hippmem_core::model::links::SemanticSignature {
                    lexical_simhash: [0; 4],
                    dense_embedding_ref: None,
                    binary_code: [0, 0],
                    topic_minhash: [0u32; 16],
                },
                topic_keys: vec![],
                emotion_keys: vec![],
                goal_keys: vec![],
                event_keys: vec![],
                causal_keys: vec![],
            },
            links: vec![],
            activation: hippmem_core::model::links::ActivationState {
                last_retrieved_at: None,
                retrieval_count: 0,
                co_activations: vec![],
                usage_score: hippmem_core::score::UnitScore::new(0.5),
            },
            lifecycle: hippmem_core::model::unit::MemoryLifecycle::Active,
            provenance: hippmem_core::model::unit::Provenance {
                origin: hippmem_core::model::unit::SourceKind::Conversation,
                generated_by: hippmem_core::model::unit::GeneratedBy::UserDirect,
                reliability: hippmem_core::score::UnitScore::new(0.5),
                evidence_refs: vec![],
                revision_history: vec![],
            },
            stage: hippmem_core::model::unit::MemoryStage::Indexed,
        };
        let exp = explain_result(&unit, &[], &[], &[]);
        assert!(exp.summary.contains("42"));
    }

    #[test]
    fn inspect_counts_units() {
        let report = inspect_units(&[]);
        assert_eq!(report.memory_count, 0);
    }
}
