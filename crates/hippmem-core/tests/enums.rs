//! acceptance test: all enums defined in 02 and their serialization

use hippmem_core::ids::MemoryId;
use hippmem_core::model::enums::*;
use hippmem_core::model::links::{
    ActivationStep, LinkType, MatchDimension, ObservationState, RecallChannel, RetrievalMode,
};
use hippmem_core::model::understanding::{
    CausalKind, EmotionKind, EntityType, GoalStatus, Polarity,
};
use hippmem_core::model::unit::{
    ContentType, GeneratedBy, Language, MemoryLifecycle, MemoryStage, SourceKind,
};
use hippmem_core::model::MemoryUnit;
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;

// ── Serialization round-trip helpers ──

fn serde_roundtrip_bincode<
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
>(
    value: &T,
) {
    let encoded = bincode::serde::encode_to_vec(value, bincode::config::standard())
        .expect("bincode encoding failed");
    let decoded: T = bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
        .expect("bincode decoding failed")
        .0;
    assert_eq!(*value, decoded, "bincode round-trip not equal");
}

fn serde_roundtrip_json<
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
>(
    value: &T,
) {
    let json = serde_json::to_string(value).expect("JSON serialization failed");
    let decoded: T = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(*value, decoded, "JSON round-trip not equal");
}

// ── ContentType ──

#[test]
fn content_type_all_variants() {
    let variants = [
        ContentType::UserStatement,
        ContentType::AssistantObservation,
        ContentType::ToolResult,
        ContentType::Decision,
        ContentType::Preference,
        ContentType::Event,
        ContentType::TaskState,
        ContentType::ProjectKnowledge,
        ContentType::Reflection,
        ContentType::Correction,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── Language ──

#[test]
fn language_all_variants() {
    let variants = [
        Language::Zh,
        Language::En,
        Language::Code,
        Language::Mixed,
        Language::Other(42),
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── SourceKind ──

#[test]
fn source_kind_all_variants() {
    let variants = [
        SourceKind::Conversation,
        SourceKind::File,
        SourceKind::Tool,
        SourceKind::ExternalSystem,
        SourceKind::MemoryRef,
        SourceKind::Other,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── MemoryStage ──

#[test]
fn memory_stage_all_variants() {
    let variants = [
        MemoryStage::Raw,
        MemoryStage::Indexed,
        MemoryStage::Enriched,
        MemoryStage::Consolidated,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── MemoryLifecycle ──

#[test]
fn memory_lifecycle_all_variants() {
    let id = MemoryId(1);
    let variants = [
        MemoryLifecycle::Active,
        MemoryLifecycle::Compressed { into: id },
        MemoryLifecycle::Archived,
        MemoryLifecycle::Superseded { by: id },
        MemoryLifecycle::Deprecated,
        MemoryLifecycle::Negated { by: id },
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── GeneratedBy ──

#[test]
fn generated_by_all_variants() {
    let variants = [
        GeneratedBy::UserDirect,
        GeneratedBy::Extractor {
            backend: "deterministic".into(),
        },
        GeneratedBy::Consolidation,
        GeneratedBy::Rule,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── EntityType ──

#[test]
fn entity_type_all_variants() {
    let variants = [
        EntityType::Person,
        EntityType::Project,
        EntityType::Library,
        EntityType::File,
        EntityType::Org,
        EntityType::Concept,
        EntityType::Other,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── GoalStatus ──

#[test]
fn goal_status_all_variants() {
    let variants = [
        GoalStatus::Active,
        GoalStatus::Achieved,
        GoalStatus::Abandoned,
        GoalStatus::Blocked,
        GoalStatus::Unknown,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── Polarity ──

#[test]
fn polarity_all_variants() {
    let variants = [Polarity::Like, Polarity::Dislike, Polarity::Neutral];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── EmotionKind ──

#[test]
fn emotion_kind_all_variants() {
    let variants = [
        EmotionKind::Joy,
        EmotionKind::Sadness,
        EmotionKind::Anger,
        EmotionKind::Fear,
        EmotionKind::Surprise,
        EmotionKind::Disgust,
        EmotionKind::Frustration,
        EmotionKind::Anxiety,
        EmotionKind::Satisfaction,
        EmotionKind::Neutral,
        EmotionKind::Other,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── CausalKind ──

#[test]
fn causal_kind_all_variants() {
    let variants = [CausalKind::Explicit, CausalKind::Implicit];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── LinkType (including Supersedes/Deprecated) ──

#[test]
fn link_type_all_variants() {
    let variants = [
        LinkType::EntityOverlap,
        LinkType::TemporalAdjacent,
        LinkType::SemanticSimilar,
        LinkType::TopicRelated,
        LinkType::SameGoal,
        LinkType::SameEvent,
        LinkType::Causal,
        LinkType::EmotionalResonance,
        LinkType::Contradiction,
        LinkType::Correction,
        LinkType::Elaboration,
        LinkType::CoActivation,
        LinkType::Supersedes,
        LinkType::Deprecated,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── LinkDirection ──

#[test]
fn link_direction_all_variants() {
    let variants = [
        LinkDirection::Undirected,
        LinkDirection::Forward,
        LinkDirection::Backward,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── MatchDimension ──

#[test]
fn match_dimension_all_variants() {
    let variants = [
        MatchDimension::Entity,
        MatchDimension::Semantic,
        MatchDimension::Temporal,
        MatchDimension::Topic,
        MatchDimension::Goal,
        MatchDimension::Event,
        MatchDimension::Emotion,
        MatchDimension::Causal,
        MatchDimension::CoContext,
        MatchDimension::Importance,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── ObservationState ──

#[test]
fn observation_state_all_variants() {
    let ts = Timestamp::from_millis(1_700_000_000_000);
    let variants = [
        ObservationState::Confirmed,
        ObservationState::Observing { since: ts },
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── RecallChannel ──

#[test]
fn recall_channel_all_variants() {
    let variants = [
        RecallChannel::Bm25,
        RecallChannel::EntityInverted,
        RecallChannel::SemanticDense,
        RecallChannel::SemanticBinary,
        RecallChannel::Temporal,
        RecallChannel::TopicCluster,
        RecallChannel::Goal,
        RecallChannel::Event,
        RecallChannel::Causal,
        RecallChannel::RecentActivation,
        RecallChannel::GraphSpreading,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── RetrievalMode ──

#[test]
fn retrieval_mode_all_variants() {
    let variants = [
        RetrievalMode::Fast,
        RetrievalMode::Balanced,
        RetrievalMode::Deep,
        RetrievalMode::Diagnostic,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── MemoryWarning ──

#[test]
fn memory_warning_all_variants() {
    let id = MemoryId(1);
    let variants = [
        MemoryWarning::HasCorrection { by: id },
        MemoryWarning::HasContradiction { with: id },
        MemoryWarning::Superseded { by: id },
        MemoryWarning::Deprecated,
        MemoryWarning::LowConfidence,
        MemoryWarning::StaleFreshness,
    ];
    for v in &variants {
        serde_roundtrip_bincode(v);
        serde_roundtrip_json(v);
    }
}

// ── MemoryUnit in activation_trace is serializable ──

#[test]
fn activation_trace_serializable() {
    // Build an activation trace with an empty ActivationState
    let trace = ActivationStep {
        from: None,
        to: MemoryId(1),
        via_link: None,
        channel: Some(RecallChannel::Bm25),
        hop: 0,
        energy_in: 1.0,
        energy_out: 0.8,
    };
    serde_roundtrip_bincode(&trace);
    serde_roundtrip_json(&trace);
}

#[test]
fn simplest_memory_unit_serializable() {
    // A minimal MemoryUnit verifying that all enum types are serializable
    let unit = MemoryUnit {
        schema_version: 1,
        id: MemoryId(1),
        created_at: Timestamp::from_millis(0),
        updated_at: Timestamp::from_millis(0),
        content: hippmem_core::model::unit::MemoryContent {
            raw: String::new(),
            summary: None,
            normalized: None,
            language: Language::Zh,
            content_type: ContentType::UserStatement,
        },
        context: hippmem_core::model::unit::WriteContext {
            conversation_id: None,
            session_id: None,
            project_id: None,
            task_id: None,
            user_id: None,
            local_time: Timestamp::from_millis(0),
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
                binary_code: [0; 2],
                topic_minhash: [0; 16],
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
        lifecycle: MemoryLifecycle::Active,
        provenance: hippmem_core::model::unit::Provenance {
            origin: SourceKind::Other,
            generated_by: GeneratedBy::UserDirect,
            reliability: UnitScore::new(0.5),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: MemoryStage::Raw,
    };
    serde_roundtrip_bincode(&unit);
    serde_roundtrip_json(&unit);
}
