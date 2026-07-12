//! acceptance tests: full write pipeline integration
//! (keys -> candidates -> scoring -> edges).
//!
//! Uses `raw_to_indexed` to verify the complete write chain, covering 4 scenarios.
//! Locale-specific test data lives in `tests/fixtures/pipeline/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_core::config::AlgoParams;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{MatchDimension, ObservationState, SemanticSignature};
use hippmem_core::model::understanding::{
    EntityMention, EntityType, MemoryUnderstanding, TopicTag,
};
use hippmem_core::model::unit::{ContentType, Language, MemoryContent, WriteContext};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_write::edges::EdgeBuildParams;
use hippmem_write::staged::{raw_to_indexed, StagedWriteInput};
use std::fs;

/// Discover available locale fixtures.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/pipeline", env!("CARGO_MANIFEST_DIR"));
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
        panic!("no locale fixtures found in pipeline/");
    }
    locales
}

/// Load pipeline fixture for a specific locale.
fn load_fixture(locale: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/pipeline/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = fs::read_to_string(&path).expect("failed to read fixture");
    serde_json::from_str(&data).expect("invalid fixture")
}

fn now() -> Timestamp {
    Timestamp(1_700_000_000_000)
}

fn make_semantic() -> SemanticSignature {
    SemanticSignature {
        lexical_simhash: [1, 2, 3, 4],
        dense_embedding_ref: None,
        binary_code: [0xABCD, 0x1234],
        topic_minhash: [0u32; 16],
    }
}

fn make_input(id: u128, text: &str, entity: &str, extra_topics: Vec<&str>) -> StagedWriteInput {
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
            topics: extra_topics
                .into_iter()
                .map(|t| TopicTag {
                    label: t.to_string(),
                    confidence: UnitScore::new(0.5),
                })
                .collect(),
            events: vec![],
            goals: vec![],
            decisions: vec![],
            preferences: vec![],
            emotions: vec![],
            causal_claims: vec![],
            contradictions: vec![],
            importance: UnitScore::new(0.5),
            confidence: UnitScore::new(0.5),
        },
        context: WriteContext {
            conversation_id: Some(1),
            session_id: Some(1),
            project_id: None,
            task_id: None,
            user_id: None,
            local_time: now(),
            preceding_memory_ids: vec![],
            source_refs: vec![],
        },
        semantic: make_semantic(),
    }
}

/// Scenario 1: two memories sharing an entity produce an EntityOverlap edge.
#[test]
fn shared_entity_builds_entity_overlap_edge() {
    let first = raw_to_indexed(
        make_input(1, "Rust backend development", "Rust", vec![]),
        &[],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();
    let first_id = first.unit.id;
    let first_unit = first.unit;

    let second = raw_to_indexed(
        make_input(2, "Rust dev tools", "Rust", vec![]),
        &[first_unit],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();

    assert!(
        !second.created_links.is_empty(),
        "shared entity Rust should create an edge"
    );
    let edge = &second.created_links[0];
    assert_eq!(
        edge.target_id, first_id,
        "edge should point to the existing memory"
    );
    assert_eq!(
        edge.link_type,
        hippmem_core::model::links::LinkType::EntityOverlap,
        "should be of type EntityOverlap"
    );
    assert!(
        !edge.evidence.contributing_dimensions.is_empty(),
        "evidence should have hit dimensions"
    );
    assert!(
        !edge.evidence.score_breakdown.is_empty(),
        "evidence should have a score breakdown"
    );
}

/// Scenario 2: unrelated memories (different entities) do not create an edge.
/// Test data loaded from locale-tagged fixture per P7. All locales tested.
#[test]
fn unrelated_entities_produce_no_edge() {
    for locale in discover_fixture_locales() {
        let fixture = load_fixture(&locale);
        let svc1 = fixture["services"][0].as_object().unwrap();
        let svc2 = fixture["services"][1].as_object().unwrap();
        let text1 = svc1["name"].as_str().unwrap();
        let text2 = svc2["name"].as_str().unwrap();
        let kw1 = svc1["keyword"].as_str().unwrap();
        let kw2 = svc2["keyword"].as_str().unwrap();

        let first = raw_to_indexed(
            make_input(svc1["id"].as_u64().unwrap() as u128, text1, kw1, vec![]),
            &[],
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
        )
        .unwrap();

        let second = raw_to_indexed(
            make_input(svc2["id"].as_u64().unwrap() as u128, text2, kw2, vec![]),
            &[first.unit],
            &EdgeBuildParams::default(),
            &AlgoParams::default(),
        )
        .unwrap();

        assert!(
            second.created_links.is_empty(),
            "[{locale}] different entities should not create an edge"
        );
    } // end locale loop
}

/// Scenario 3: multi-dim sharing (entity+topic+time) yields a higher dimension
/// hit count.
#[test]
fn multi_dim_shared_has_more_dimensions_in_evidence() {
    // M1: contains entity Rust + topic tags (database / engine)
    let first = raw_to_indexed(
        make_input(
            1,
            "Rust database engine",
            "Rust",
            vec!["database", "engine"],
        ),
        &[],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();

    // M2: same entity + same topic + same time bucket -> multi-dim hit
    let second = raw_to_indexed(
        make_input(
            2,
            "Rust database engine optimization",
            "Rust",
            vec!["database", "engine"],
        ),
        &[first.unit],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();

    assert!(
        !second.created_links.is_empty(),
        "multi-dim sharing should create an edge"
    );

    let edge = &second.created_links[0];
    let dims = &edge.evidence.contributing_dimensions;
    // Should have at least Entity + Topic + Temporal (same time) dimensions
    assert!(
        dims.len() >= 2,
        "should have at least 2 hit dimensions, got {:?}",
        dims
    );

    // Verify specific dimensions are present
    let has_entity = dims.contains(&MatchDimension::Entity);
    let has_topic = dims.contains(&MatchDimension::Topic);
    assert!(has_entity, "should include the Entity dimension");
    assert!(has_topic, "should include the Topic dimension");
}

/// Scenario 4: edge strength is in the valid range [0, 1], Confirmed state.
#[test]
fn edge_strength_in_valid_range() {
    let first = raw_to_indexed(
        make_input(1, "Rust database engine", "Rust", vec!["database"]),
        &[],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();

    let second = raw_to_indexed(
        make_input(2, "Rust database query", "Rust", vec!["database"]),
        &[first.unit],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();

    assert!(!second.created_links.is_empty());
    let edge = &second.created_links[0];

    // Strength in [0, 1]
    let strength = edge.strength.value();
    assert!(
        (0.0..=1.0).contains(&strength),
        "strength {strength} should be in [0,1]"
    );
    // Confidence in [0, 1]
    let conf = edge.confidence.value();
    assert!(
        (0.0..=1.0).contains(&conf),
        "confidence {conf} should be in [0,1]"
    );
    // Should have an activated_at timestamp
    assert!(
        edge.last_activated_at.is_some(),
        "should have an activation time"
    );
    // Multi-dim sharing (entity+topic+temporal>=3) -> should trigger the
    // multi-dim bonus -> score should be high enough to reach Confirmed
    let is_confirmed = matches!(edge.observation, ObservationState::Confirmed);
    assert!(
        is_confirmed,
        "multi-dim sharing should create a Confirmed edge"
    );
}
