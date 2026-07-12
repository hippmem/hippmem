//! Staged write path: raw -> indexed (03/05).

use crate::candidates::{discover_candidates, simhash_similarity};
use crate::edges::{build_edges, EdgeBuildParams};
use crate::keys::generate_keys;
use hippmem_core::config::AlgoParams;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{ActivationState, AssociationLink, SemanticSignature};
use hippmem_core::model::understanding::MemoryUnderstanding;
use hippmem_core::model::unit::{
    MemoryContent, MemoryLifecycle, MemoryStage, MemoryUnit, WriteContext,
};
use hippmem_core::score::UnitScore;

pub struct StagedWriteInput {
    pub id: MemoryId,
    pub content: MemoryContent,
    pub understanding: MemoryUnderstanding,
    pub context: WriteContext,
    pub semantic: SemanticSignature,
}

pub struct StagedWriteOutput {
    pub unit: MemoryUnit,
    pub created_links: Vec<AssociationLink>,
}

/// raw -> indexed: generate keys and build initial edges.
pub fn raw_to_indexed(
    input: StagedWriteInput,
    existing_units: &[MemoryUnit],
    edge_params: &EdgeBuildParams,
    algo_params: &AlgoParams,
) -> Result<StagedWriteOutput, String> {
    let now = input.context.local_time;
    let keys = generate_keys(
        &input.content,
        &input.understanding,
        &input.context,
        &input.semantic,
    )?;

    // Candidate pre-filter: sort by SimHash similarity and cap the candidate
    // count to control O(n^2) cost
    let mut candidates: Vec<(&MemoryUnit, f32)> = existing_units
        .iter()
        .map(|unit| {
            // Fast SimHash similarity (cheap, no full Jaccard needed)
            let sim = simhash_similarity(
                &keys.lexical_signature.simhash,
                &unit.association_keys.lexical_signature.simhash,
            );
            (unit, sim)
        })
        .collect();

    // Sort by similarity descending, take top max_candidates (0 = no limit)
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if edge_params.max_candidates > 0 {
        candidates.truncate(edge_params.max_candidates);
    }

    let total_memories = (existing_units.len() + 1) as u32;
    let mut all_links = Vec::new();
    for (existing, _sim) in &candidates {
        let mut cand = discover_candidates(&keys, &existing.association_keys);
        // Fill in dimensions that discover_candidates cannot derive:
        // importance: importance of the existing memory (03 §2.1)
        cand.importance_value = existing.understanding.importance.value();
        // co_context: shared context ratio (03 §2.1)
        cand.co_context_score = context_shared_ratio(&input.context, &existing.context);
        let result = build_edges(
            input.id,
            existing.id,
            &cand,
            cand.matched_dimensions.len(),
            edge_params,
            algo_params,
            &existing.links,
            now,
            total_memories,
        );
        all_links.extend(result.created_links);
    }

    let unit = MemoryUnit {
        schema_version: 1,
        id: input.id,
        created_at: now,
        updated_at: now,
        content: input.content,
        context: input.context,
        understanding: input.understanding,
        association_keys: keys,
        links: all_links.clone(),
        activation: ActivationState {
            last_retrieved_at: None,
            retrieval_count: 0,
            co_activations: vec![],
            usage_score: UnitScore::new(0.5),
        },
        lifecycle: MemoryLifecycle::Active,
        provenance: hippmem_core::model::unit::Provenance {
            origin: hippmem_core::model::unit::SourceKind::Conversation,
            generated_by: hippmem_core::model::unit::GeneratedBy::UserDirect,
            reliability: UnitScore::new(0.5),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: MemoryStage::Indexed,
    };

    Ok(StagedWriteOutput {
        unit,
        created_links: all_links,
    })
}

/// Compute the shared ratio of two WriteContexts: shared field count / total
/// non-None field count.
/// Result is in [0, 1], used for co_context_score (03 §2.1).
fn context_shared_ratio(a: &WriteContext, b: &WriteContext) -> f32 {
    let fields = [
        (a.conversation_id, b.conversation_id),
        (a.session_id, b.session_id),
        (a.project_id, b.project_id),
        (a.task_id, b.task_id),
    ];
    let total: usize = fields
        .iter()
        .filter(|(fa, fb)| fa.is_some() || fb.is_some())
        .count();
    if total == 0 {
        return 0.0;
    }
    let shared = fields
        .iter()
        .filter(|(fa, fb)| fa.is_some() && fa == fb)
        .count();
    shared as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::model::understanding::{EntityMention, EntityType};
    use hippmem_core::model::unit::{ContentType, Language};
    use hippmem_core::time::Timestamp;

    fn make_input(id: u128, text: &str, entity: &str) -> StagedWriteInput {
        StagedWriteInput {
            id: MemoryId(id),
            content: MemoryContent {
                raw: text.into(),
                summary: None,
                normalized: None,
                language: Language::Zh,
                content_type: ContentType::UserStatement,
            },
            understanding: MemoryUnderstanding {
                entities: vec![EntityMention {
                    text: entity.into(),
                    canonical: entity.to_lowercase(),
                    entity_type: EntityType::Other,
                    span: None,
                    confidence: UnitScore::new(0.8),
                }],
                events: vec![],
                goals: vec![],
                decisions: vec![],
                preferences: vec![],
                emotions: vec![],
                causal_claims: vec![],
                contradictions: vec![],
                topics: vec![],
                importance: UnitScore::new(0.5),
                confidence: UnitScore::new(0.5),
            },
            context: WriteContext {
                conversation_id: Some(1),
                session_id: Some(1),
                project_id: None,
                task_id: None,
                user_id: None,
                local_time: Timestamp(1_700_000_000_000),
                preceding_memory_ids: vec![],
                source_refs: vec![],
            },
            semantic: SemanticSignature {
                lexical_simhash: [1, 2, 3, 4],
                dense_embedding_ref: None,
                binary_code: [0, 0],
                topic_minhash: [0u32; 16],
            },
        }
    }

    #[test]
    fn raw_to_indexed_succeeds() {
        let input = make_input(1, "Rust", "Rust");
        let output = raw_to_indexed(
            input,
            &[],
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
        )
        .unwrap();
        assert_eq!(output.unit.stage, MemoryStage::Indexed);
    }

    #[test]
    fn shared_entity_produces_links() {
        let first = raw_to_indexed(
            make_input(1, "Rust", "Rust"),
            &[],
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
        )
        .unwrap();
        let second = raw_to_indexed(
            make_input(2, "I also use Rust for systems programming", "Rust"),
            &[first.unit],
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
        )
        .unwrap();
        assert!(!second.created_links.is_empty());
    }
}
