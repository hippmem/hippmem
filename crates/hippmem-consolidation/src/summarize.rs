//! Compaction and merging: low-level memories → summary memory + covers chain (03 §8).

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{AssociationLink, LinkDirection, LinkType, ObservationState};
use hippmem_core::model::unit::{GeneratedBy, MemoryContent, MemoryStage, MemoryUnit};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_model::deterministic::summarize::DeterministicSummarizer;
use hippmem_model::traits::SummarizeInput;

/// Checks whether summarization should be triggered: the number of similar memories exceeds the threshold.
pub fn should_summarize(similar_ids: &[MemoryId], threshold: usize) -> bool {
    similar_ids.len() >= threshold
}

/// Builds a summary MemoryUnit: uses the Summarizer to generate summary text,
/// covering all original memories in `sources` (covers chain).
///
/// If the Summarizer returns confidence < 0.35, the caller should skip summary creation
/// (confidence gating, Constitution C7).
/// The returned MemoryUnit.understanding.confidence reflects the actual confidence.
pub fn build_summary_unit(
    sources: &[MemoryUnit],
    summarizer: &DeterministicSummarizer,
) -> MemoryUnit {
    // Use the Summarizer to generate the summary (the degraded backend does extractive summarization)
    let summarize_inputs: Vec<SummarizeInput> = sources
        .iter()
        .map(|u| SummarizeInput {
            id: u.id,
            text: u.content.raw.clone(),
        })
        .collect();

    let summary_output = match summarizer.summarize_sync(&summarize_inputs) {
        Ok(out) => out,
        Err(_) => {
            // Fallback summary when the Summarizer fails (simple concatenation)
            let fallback_text: String = sources
                .iter()
                .take(3)
                .map(|u| u.content.raw.chars().take(80).collect::<String>())
                .collect::<Vec<_>>()
                .join("; ");
            hippmem_model::traits::SummaryOutput {
                summary: fallback_text,
                covers: sources.iter().map(|u| u.id).collect(),
                confidence: UnitScore::new(0.1),
            }
        }
    };

    let summary_text = summary_output.summary;
    let summary_confidence = summary_output.confidence;
    let covers: Vec<MemoryId> = sources.iter().map(|u| u.id).collect();

    // Build an Elaboration edge for each original memory (summary → original)
    let links: Vec<AssociationLink> = covers
        .iter()
        .map(|target_id| AssociationLink {
            target_id: *target_id,
            link_type: LinkType::Elaboration,
            direction: LinkDirection::Forward,
            strength: UnitScore::new(0.5),
            confidence: UnitScore::new(0.6),
            evidence: hippmem_core::model::links::LinkEvidence {
                contributing_dimensions: vec![],
                score_breakdown: vec![],
                text_spans: vec![],
                note: Some("summary covers".into()),
            },
            formed_at: Timestamp(0),
            last_activated_at: None,
            activation_count: 0,
            observation: ObservationState::Confirmed,
        })
        .collect();

    MemoryUnit {
        schema_version: 1,
        id: MemoryId::generate(),
        created_at: Timestamp(0),
        updated_at: Timestamp(0),
        content: MemoryContent {
            raw: summary_text.clone(),
            summary: Some(summary_text),
            normalized: None,
            language: hippmem_core::model::unit::Language::Zh,
            content_type: hippmem_core::model::unit::ContentType::Reflection,
        },
        context: hippmem_core::model::unit::WriteContext {
            conversation_id: None,
            session_id: None,
            project_id: None,
            task_id: None,
            user_id: None,
            local_time: Timestamp(0),
            preceding_memory_ids: covers.clone(),
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
            importance: UnitScore::new(0.5),
            confidence: summary_confidence,
        },
        association_keys: hippmem_core::model::links::AssociationKeys {
            entity_keys: vec![],
            temporal_keys: vec![],
            lexical_signature: hippmem_core::model::links::LexicalSignature { simhash: [0; 4] },
            semantic_signature: hippmem_core::model::links::SemanticSignature {
                lexical_simhash: [0; 4],
                dense_embedding_ref: None,
                binary_code: [0; 2],
                topic_minhash: [0u32; 16],
            },
            topic_keys: vec![],
            emotion_keys: vec![],
            goal_keys: vec![],
            event_keys: vec![],
            causal_keys: vec![],
        },
        links,
        activation: hippmem_core::model::links::ActivationState {
            last_retrieved_at: None,
            retrieval_count: 0,
            co_activations: vec![],
            usage_score: UnitScore::new(0.5),
        },
        lifecycle: hippmem_core::model::unit::MemoryLifecycle::Active,
        provenance: hippmem_core::model::unit::Provenance {
            origin: hippmem_core::model::unit::SourceKind::Conversation,
            generated_by: GeneratedBy::Consolidation,
            reliability: UnitScore::new(0.6),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: MemoryStage::Consolidated,
    }
}
