//! acceptance test: MemoryUnit and all sub-types serialization round-trip
//!
//! Locale-specific test content lives in `tests/fixtures/model_roundtrip/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{
    ActivationState, ActivationStep, AssociationKeys, AssociationLink, CoActivationCount,
    LexicalSignature, LinkDirection, LinkEvidence, LinkType, MatchDimension, MemoryWarning,
    ObservationState, RecallChannel, RetrievalResult, SemanticSignature,
};
use hippmem_core::model::understanding::{
    CausalClaim, CausalKind, DecisionFrame, EmotionFrame, EmotionKind, EntityMention, EntityType,
    EventFrame, GoalFrame, GoalStatus, MemoryUnderstanding, Polarity, PreferenceFrame, TopicTag,
};
use hippmem_core::model::unit::{
    ContentType, GeneratedBy, Language, MemoryContent, MemoryLifecycle, MemoryStage, MemoryUnit,
    Provenance, RevisionMark, SourceKind, SourceRef, TextSpan, WriteContext,
};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use std::fs;

/// Discover available locale fixtures.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!(
        "{}/tests/fixtures/model_roundtrip",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut locales = vec![];
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".json") {
                locales.push(name.trim_end_matches(".json").to_string());
            }
        }
    }
    locales.sort();
    if locales.is_empty() {
        panic!("no locale fixtures found in model_roundtrip/");
    }
    locales
}

/// Load model roundtrip fixture for a specific locale.
fn load_fixture(locale: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/model_roundtrip/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = fs::read_to_string(&path).expect("failed to read fixture");
    serde_json::from_str(&data).expect("invalid fixture")
}

fn fixture_str(fixture: &serde_json::Value, key: &str) -> String {
    fixture[key].as_str().unwrap().to_string()
}

fn locale_to_lang(locale: &str) -> Language {
    match locale {
        "zh" => Language::Zh,
        _ => Language::En,
    }
}

// ── Helper function to construct a complete MemoryUnit ──

fn make_sample_memory_unit(id: u128, locale: &str) -> MemoryUnit {
    let f = load_fixture(locale);
    let lang = locale_to_lang(locale);
    MemoryUnit {
        schema_version: 1,
        id: MemoryId(id),
        created_at: Timestamp::from_millis(1_700_000_000_000),
        updated_at: Timestamp::from_millis(1_700_000_001_000),
        content: MemoryContent {
            raw: fixture_str(&f, "content_raw"),
            summary: None,
            normalized: Some("hippmem is a native associative memory engine".into()),
            language: lang,
            content_type: ContentType::ProjectKnowledge,
        },
        context: WriteContext {
            conversation_id: Some(42),
            session_id: Some(1),
            project_id: Some(100),
            task_id: None,
            user_id: None,
            local_time: Timestamp::from_millis(1_700_000_000_000),
            preceding_memory_ids: vec![MemoryId(99)],
            source_refs: vec![SourceRef {
                kind: SourceKind::Conversation,
                locator: "urn:session:1:turn:5".into(),
                span: Some(TextSpan { start: 0, end: 100 }),
            }],
        },
        understanding: MemoryUnderstanding {
            entities: vec![EntityMention {
                text: "HIPPMEM".into(),
                canonical: "hippmem".into(),
                entity_type: EntityType::Project,
                span: Some(TextSpan { start: 0, end: 7 }),
                confidence: UnitScore::new(0.95),
            }],
            events: vec![EventFrame {
                action: fixture_str(&f, "event_action"),
                participants: vec!["hippmem".into()],
                occurred_at: Some(Timestamp::from_millis(1_700_000_000_000)),
                outcome: None,
                confidence: UnitScore::new(0.8),
            }],
            goals: vec![GoalFrame {
                description: fixture_str(&f, "goal_description"),
                status: GoalStatus::Active,
                constraints: vec![fixture_str(&f, "goal_constraint")],
                confidence: UnitScore::new(0.9),
            }],
            decisions: vec![DecisionFrame {
                decision: fixture_str(&f, "decision_choice"),
                rationale: Some(fixture_str(&f, "goal_constraint")),
                decided_at: Some(Timestamp::from_millis(1_699_000_000_000)),
                reverted: false,
                confidence: UnitScore::new(0.85),
            }],
            preferences: vec![PreferenceFrame {
                object: fixture_str(&f, "preference_object"),
                polarity: Polarity::Like,
                strength: UnitScore::new(0.7),
                still_valid: true,
                confidence: UnitScore::new(0.8),
            }],
            emotions: vec![EmotionFrame {
                emotion: EmotionKind::Satisfaction,
                intensity: UnitScore::new(0.6),
                trigger: Some(fixture_str(&f, "emotion_trigger")),
                confidence: UnitScore::new(0.7),
            }],
            causal_claims: vec![CausalClaim {
                cause: fixture_str(&f, "causal_cause"),
                effect: fixture_str(&f, "causal_effect"),
                kind: CausalKind::Explicit,
                evidence_span: Some(TextSpan { start: 0, end: 50 }),
                confidence: UnitScore::new(0.9),
            }],
            contradictions: vec![],
            topics: vec![TopicTag {
                label: fixture_str(&f, "topic_label"),
                confidence: UnitScore::new(0.85),
            }],
            importance: UnitScore::new(0.75),
            confidence: UnitScore::new(0.8),
        },
        association_keys: AssociationKeys {
            entity_keys: vec![42],
            temporal_keys: vec![1_700_000],
            lexical_signature: LexicalSignature {
                simhash: [1, 2, 3, 4],
            },
            semantic_signature: SemanticSignature {
                lexical_simhash: [1, 2, 3, 4],
                dense_embedding_ref: None,
                binary_code: [0xDEAD, 0xBEEF],
                topic_minhash: [0; 16],
            },
            topic_keys: vec![100],
            emotion_keys: vec![7],
            goal_keys: vec![200],
            event_keys: vec![300],
            causal_keys: vec![400],
        },
        links: vec![AssociationLink {
            target_id: MemoryId(99),
            link_type: LinkType::Causal,
            direction: LinkDirection::Forward,
            strength: UnitScore::new(0.6),
            confidence: UnitScore::new(0.8),
            evidence: LinkEvidence {
                contributing_dimensions: vec![MatchDimension::Entity, MatchDimension::Causal],
                score_breakdown: vec![(MatchDimension::Entity, 0.3), (MatchDimension::Causal, 0.3)],
                text_spans: vec![TextSpan { start: 10, end: 60 }],
                note: Some(fixture_str(&f, "link_note")),
            },
            formed_at: Timestamp::from_millis(1_700_000_001_000),
            last_activated_at: None,
            activation_count: 0,
            observation: ObservationState::Confirmed,
        }],
        activation: ActivationState {
            last_retrieved_at: None,
            retrieval_count: 0,
            co_activations: vec![CoActivationCount {
                with: MemoryId(99),
                count: 3,
                last_at: Timestamp::from_millis(1_700_000_002_000),
            }],
            usage_score: UnitScore::new(0.5),
        },
        lifecycle: MemoryLifecycle::Active,
        provenance: Provenance {
            origin: SourceKind::Conversation,
            generated_by: GeneratedBy::Extractor {
                backend: "deterministic".into(),
            },
            reliability: UnitScore::new(0.9),
            evidence_refs: vec![],
            revision_history: vec![RevisionMark {
                at: Timestamp::from_millis(1_700_000_001_000),
                reason: fixture_str(&f, "revision_reason"),
                by: GeneratedBy::Rule,
            }],
        },
        stage: MemoryStage::Indexed,
    }
}

// ── Serialization round-trip tests ──

#[test]
fn memory_unit_roundtrip_bincode() {
    for locale in discover_fixture_locales() {
        let unit = make_sample_memory_unit(1, &locale);
        let encoded = bincode::serde::encode_to_vec(&unit, bincode::config::standard())
            .expect("bincode encoding failed");
        let decoded: MemoryUnit =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
                .expect("bincode decoding failed")
                .0;
        assert_eq!(
            unit, decoded,
            "[{locale}] MemoryUnit bincode round-trip not equal"
        );
    }
}

#[test]
fn memory_unit_roundtrip_json() {
    for locale in discover_fixture_locales() {
        let unit = make_sample_memory_unit(2, &locale);
        let json = serde_json::to_string(&unit).expect("JSON serialization failed");
        let decoded: MemoryUnit = serde_json::from_str(&json).expect("JSON deserialization failed");
        assert_eq!(
            unit, decoded,
            "[{locale}] MemoryUnit JSON round-trip not equal"
        );
    }
}

// ── Invariant tests ──

#[test]
fn invariant_updated_at_gte_created_at() {
    for locale in discover_fixture_locales() {
        let unit = make_sample_memory_unit(3, &locale);
        assert!(
            unit.updated_at.as_i64() >= unit.created_at.as_i64(),
            "[{locale}] updated_at ({}) should be >= created_at ({})",
            unit.updated_at.as_i64(),
            unit.created_at.as_i64()
        );
    }
}

#[test]
fn invariant_no_self_loop() {
    for locale in discover_fixture_locales() {
        let unit = make_sample_memory_unit(4, &locale);
        let self_id = unit.id;
        let has_self_loop = unit.links.iter().any(|link| link.target_id == self_id);
        assert!(
            !has_self_loop,
            "[{locale}] links should not contain self-loops"
        );
    }
}

#[test]
fn invariant_no_duplicate_target_linktype() {
    for locale in discover_fixture_locales() {
        let unit = make_sample_memory_unit(5, &locale);
        let mut seen = std::collections::HashSet::new();
        for link in &unit.links {
            let key = (link.target_id, link.link_type);
            assert!(
                seen.insert(key),
                "[{locale}] links contains a duplicate (target_id={:?}, link_type={:?}) pair",
                link.target_id,
                link.link_type
            );
        }
    }
}

// ── All enums serialization round-trip tests ──

#[test]
fn all_enums_roundtrip_bincode() {
    test_enum_roundtrip(&MemoryLifecycle::Active);
    test_enum_roundtrip(&MemoryLifecycle::Archived);
    test_enum_roundtrip(&MemoryLifecycle::Deprecated);
    test_enum_roundtrip(&MemoryStage::Raw);
    test_enum_roundtrip(&MemoryStage::Consolidated);
    test_enum_roundtrip(&Language::Zh);
    test_enum_roundtrip(&Language::En);
    test_enum_roundtrip(&Language::Mixed);
    test_enum_roundtrip(&Language::Code);
    test_enum_roundtrip(&LinkType::EntityOverlap);
    test_enum_roundtrip(&LinkType::Causal);
    test_enum_roundtrip(&LinkType::Supersedes);
    test_enum_roundtrip(&LinkType::Deprecated);
}

fn test_enum_roundtrip<
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
>(
    value: &T,
) {
    let encoded = bincode::serde::encode_to_vec(value, bincode::config::standard())
        .expect("enum encoding failed");
    let decoded: T = bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
        .expect("enum decoding failed")
        .0;
    assert_eq!(*value, decoded, "enum bincode round-trip not equal");
}

#[test]
fn retrieval_result_roundtrip_bincode() {
    for locale in discover_fixture_locales() {
        let unit = make_sample_memory_unit(6, &locale);
        let result = RetrievalResult {
            memory: unit,
            final_score: 0.85,
            activation_trace: vec![ActivationStep {
                from: None,
                to: MemoryId(6),
                via_link: None,
                channel: Some(RecallChannel::Bm25),
                hop: 0,
                energy_in: 1.0,
                energy_out: 0.8,
            }],
            matched_dimensions: vec![MatchDimension::Entity, MatchDimension::Semantic],
            warnings: vec![MemoryWarning::HasCorrection { by: MemoryId(99) }],
        };
        let encoded = bincode::serde::encode_to_vec(&result, bincode::config::standard())
            .expect("bincode encoding failed");
        let decoded: RetrievalResult =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
                .expect("bincode decoding failed")
                .0;
        assert_eq!(
            result, decoded,
            "[{locale}] RetrievalResult bincode round-trip not equal"
        );
    }
}
