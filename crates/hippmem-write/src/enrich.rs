//! Enrich stage: complete strong semantic dimensions in the background
//! (06 §M6-001).

use hippmem_core::model::understanding::{
    DecisionFrame, EmotionFrame, GoalFrame, GoalStatus, Polarity, PreferenceFrame,
};
use hippmem_core::model::unit::MemoryUnit;
use hippmem_core::score::UnitScore;
use hippmem_model::lang::active_locales;

/// Enrich input.
pub struct EnrichInput {
    pub unit: MemoryUnit,
}

/// Enrich output: a MemoryUnit with strong semantic dimensions completed.
pub struct EnrichOutput {
    pub unit: MemoryUnit,
}

/// Background enrich: complete goals/preferences/emotions/decisions using
/// deterministic rules.
pub fn enrich_unit(input: EnrichInput) -> EnrichOutput {
    let mut unit = input.unit;
    let text = &unit.content.raw.clone();
    let mut u = unit.understanding.clone();

    // Goals
    if u.goals.is_empty() {
        u.goals = extract_goals_enrich(text);
    }
    // Preferences
    if u.preferences.is_empty() {
        u.preferences = extract_preferences(text);
    }
    // Emotions
    if u.emotions.is_empty() {
        u.emotions = extract_emotions_enrich(text);
    }
    // Decisions
    if u.decisions.is_empty() {
        u.decisions = extract_decisions_enrich(text);
    }

    unit.understanding = u;
    unit.stage = hippmem_core::model::unit::MemoryStage::Enriched;
    unit.updated_at = unit.context.local_time;

    EnrichOutput { unit }
}

fn extract_goals_enrich(text: &str) -> Vec<GoalFrame> {
    for lang in active_locales() {
        for m in lang.goal_markers {
            if text.contains(m) {
                return vec![GoalFrame {
                    description: format!("marker:{m}"),
                    status: GoalStatus::Active,
                    constraints: vec![],
                    confidence: UnitScore::new(0.45),
                }];
            }
        }
    }
    vec![]
}

fn extract_preferences(text: &str) -> Vec<PreferenceFrame> {
    for lang in active_locales() {
        for w in lang.preference_pos {
            if text.contains(w) {
                return vec![PreferenceFrame {
                    object: format!("preference:{w}"),
                    polarity: Polarity::Like,
                    strength: UnitScore::new(0.5),
                    still_valid: true,
                    confidence: UnitScore::new(0.5),
                }];
            }
        }
    }
    for lang in active_locales() {
        for w in lang.preference_neg {
            if text.contains(w) {
                return vec![PreferenceFrame {
                    object: format!("preference:{w}"),
                    polarity: Polarity::Dislike,
                    strength: UnitScore::new(0.5),
                    still_valid: true,
                    confidence: UnitScore::new(0.5),
                }];
            }
        }
    }
    vec![]
}

fn extract_emotions_enrich(text: &str) -> Vec<EmotionFrame> {
    for lang in active_locales() {
        for (w, ek) in lang.emotion_keywords {
            if text.contains(w) {
                return vec![EmotionFrame {
                    emotion: *ek,
                    intensity: UnitScore::new(0.6),
                    trigger: Some(format!("emotion:{w}")),
                    confidence: UnitScore::new(0.5),
                }];
            }
        }
    }
    vec![]
}

fn extract_decisions_enrich(text: &str) -> Vec<DecisionFrame> {
    for lang in active_locales() {
        for m in lang.decision_markers {
            if text.contains(m) {
                return vec![DecisionFrame {
                    decision: format!("decision:{m}"),
                    rationale: Some(text.chars().take(100).collect()),
                    decided_at: None,
                    reverted: false,
                    confidence: UnitScore::new(0.45),
                }];
            }
        }
    }
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::ids::MemoryId;
    use hippmem_core::model::understanding::MemoryUnderstanding;
    use hippmem_core::model::unit::{
        ContentType, Language, MemoryContent, MemoryStage, WriteContext,
    };
    use hippmem_core::time::Timestamp;

    fn make_unit(text: &str) -> MemoryUnit {
        MemoryUnit {
            schema_version: 1,
            id: MemoryId(1),
            created_at: Timestamp(0),
            updated_at: Timestamp(0),
            content: MemoryContent {
                raw: text.into(),
                summary: None,
                normalized: None,
                language: Language::Zh,
                content_type: ContentType::UserStatement,
            },
            context: WriteContext {
                conversation_id: None,
                session_id: None,
                project_id: None,
                task_id: None,
                user_id: None,
                local_time: Timestamp(1_700_000_000_000),
                preceding_memory_ids: vec![],
                source_refs: vec![],
            },
            understanding: MemoryUnderstanding {
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
                confidence: UnitScore::new(0.5),
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
                usage_score: UnitScore::new(0.5),
            },
            lifecycle: hippmem_core::model::unit::MemoryLifecycle::Active,
            provenance: hippmem_core::model::unit::Provenance {
                origin: hippmem_core::model::unit::SourceKind::Conversation,
                generated_by: hippmem_core::model::unit::GeneratedBy::UserDirect,
                reliability: UnitScore::new(0.5),
                evidence_refs: vec![],
                revision_history: vec![],
            },
            stage: MemoryStage::Indexed,
        }
    }

    #[test]
    fn enrich_adds_preferences() {
        let unit = make_unit("I like programming in Rust");
        let output = enrich_unit(EnrichInput { unit });
        assert!(!output.unit.understanding.preferences.is_empty());
        assert_eq!(output.unit.stage, MemoryStage::Enriched);
    }

    #[test]
    fn enrich_adds_goals() {
        let unit = make_unit("I plan to learn Rust programming");
        let output = enrich_unit(EnrichInput { unit });
        assert!(!output.unit.understanding.goals.is_empty());
    }

    #[test]
    fn enrich_noop_on_empty_text() {
        let unit = make_unit("The weather is nice today");
        let output = enrich_unit(EnrichInput { unit });
        assert_eq!(output.unit.stage, MemoryStage::Enriched);
    }
}
