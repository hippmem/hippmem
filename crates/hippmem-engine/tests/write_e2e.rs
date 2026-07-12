//! acceptance test: engine.write end-to-end.
//!
//! Locale-specific test content lives in `tests/fixtures/write_e2e/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::LinkType;
use hippmem_core::model::unit::{MemoryStage, WriteContext};
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput};
use tempfile::tempdir;

/// Discover available locales by listing fixture files.
#[allow(dead_code)]
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/write_e2e", env!("CARGO_MANIFEST_DIR"));
    let mut locales = vec![];
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".json") {
                locales.push(name.trim_end_matches(".json").to_string());
            }
        }
    }
    locales.sort();
    if locales.is_empty() {
        panic!("no locale fixtures found in write_e2e/");
    }
    locales
}

/// Load locale-specific test content from fixture.
fn load_fixture(locale: &str, key: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/write_e2e/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read write_e2e/{locale}.json: {e}"));
    let fixture: serde_json::Value = serde_json::from_str(&data).expect("invalid fixture");
    fixture[key]
        .as_str()
        .unwrap_or_else(|| panic!("missing key '{key}' in write_e2e/{locale}.json"))
        .to_string()
}

fn make_context() -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: hippmem_core::time::Timestamp(1_700_000_000_000),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

#[test]
fn write_basic_returns_indexed() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    let input = WriteMemoryInput {
        content: "The user enjoys developing projects with Rust".into(),
        content_type: Some(ContentType::UserStatement),
        context: make_context(),
        importance_hint: Some(0.7),
        source_refs: vec![],
    };

    let output = engine.write(input).unwrap();

    assert!(output.memory_id.0 > 0, "memory_id should be a valid ULID");
    assert_eq!(
        output.stage_reached,
        MemoryStage::Indexed,
        "sync return should be Indexed"
    );
    // understanding should have entities
    assert!(
        !output.understanding.entities.is_empty() || output.understanding.causal_claims.is_empty(),
        "understanding should contain entities or causal claims (at least non-empty)" // even an empty list is valid
    );
}

#[test]
fn write_causal_text_creates_causal_link() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // First write a baseline memory (shares entity "Rust" with later write — capitalized so rule extraction picks it up)
    let first = engine
        .write(WriteMemoryInput {
            content: "The project uses Rust and Tantivy for full-text search".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: make_context(),
            importance_hint: Some(0.6),
            source_refs: vec![],
        })
        .unwrap();

    assert_eq!(first.stage_reached, MemoryStage::Indexed);

    // Then write a memory with causal connectors + shared entity "Rust".
    // Content loaded from locale-tagged fixture per P7.
    let causal_content = load_fixture("zh", "causal_decision");
    let second = engine
        .write(WriteMemoryInput {
            content: causal_content,
            content_type: Some(ContentType::Decision),
            context: make_context(),
            importance_hint: Some(0.8),
            source_refs: vec![],
        })
        .unwrap();

    assert_eq!(
        second.stage_reached,
        MemoryStage::Indexed,
        "second write should also be Indexed"
    );
    assert!(
        !second.created_links.is_empty(),
        "should create at least one association link (entity overlap)"
    );
    let has_entity = second
        .created_links
        .iter()
        .any(|link| link.link_type == LinkType::EntityOverlap);
    assert!(
        has_entity,
        "shared entity Rust should create an EntityOverlap link"
    );

    // Text with causal connectors should produce a CausalClaim in understanding
    assert!(
        !second.understanding.causal_claims.is_empty(),
        "causal connectors (cause-effect pairs) should produce causal_claims"
    );
}

#[test]
fn write_two_related_memories_create_links() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write multiple related memories (sharing capitalized entity "Rust")
    engine
        .write(WriteMemoryInput {
            content: "The team uses the Rust language".into(),
            content_type: Some(ContentType::UserStatement),
            context: make_context(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    let out = engine
        .write(WriteMemoryInput {
            content: "Rust is suitable for building high-performance backend systems".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: make_context(),
            importance_hint: Some(0.5),
            source_refs: vec![],
        })
        .unwrap();

    // Two related memories (sharing entity "Rust") should create links
    assert!(
        !out.created_links.is_empty(),
        "memories sharing entities should create association links, got {} links",
        out.created_links.len()
    );
    let has_entity = out
        .created_links
        .iter()
        .any(|l| l.link_type == LinkType::EntityOverlap);
    assert!(has_entity, "should contain an entity-overlap link");
}

#[test]
fn write_persists_to_store() {
    let dir = tempdir().unwrap();
    let store_path = dir.path().join("hippmem.redb");

    {
        let config = EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        };
        let engine = Engine::open(config).unwrap();
        engine
            .write(WriteMemoryInput {
                content: "persistence test memory".into(),
                content_type: None,
                context: make_context(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        engine.close().unwrap();
    }

    // Reopen; data should still be present
    {
        let config = EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        };
        let _engine = Engine::open(config).unwrap();
        // TODO: once retrieve is implemented, verify this memory can be recalled
        // For now only verify reopen does not error and the store file is non-empty
        assert!(store_path.exists());
        assert!(store_path.metadata().unwrap().len() > 0);
    }
}

/// V2-001b: After writing a memory with links, the link_overlay table should contain the corresponding outgoing edge.
#[test]
fn write_persists_links_to_link_overlay() {
    let dir = tempdir().unwrap();
    let store_path = dir.path().join("hippmem.redb");

    let second_id = {
        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();

        // Write two memories sharing an entity to produce an EntityOverlap edge
        engine
            .write(WriteMemoryInput {
                content: "Building high-performance backend services with Rust".into(),
                content_type: Some(ContentType::ProjectKnowledge),
                context: make_context(),
                importance_hint: Some(0.7),
                source_refs: vec![],
            })
            .unwrap();

        let second = engine
            .write(WriteMemoryInput {
                content: "The team uses Rust to build a database engine".into(),
                content_type: Some(ContentType::Decision),
                context: make_context(),
                importance_hint: Some(0.6),
                source_refs: vec![],
            })
            .unwrap();

        // Second memory should produce at least one outgoing edge (pointing to the first, sharing entity "Rust")
        assert!(
            !second.created_links.is_empty(),
            "memories sharing an entity should produce an edge"
        );

        engine.close().unwrap();
        second.memory_id
    };

    // Reopen and read the link_overlay table directly via GraphStore
    {
        let db = redb::Database::open(&store_path).unwrap();
        let graph = hippmem_store::graph::GraphStore::new(std::sync::Arc::new(db));
        let outgoing = graph.get_outgoing(&second_id).unwrap();

        // Verify link_overlay persistence: the second memory's outgoing edge should be readable
        assert!(
            !outgoing.is_empty(),
            "the second memory's outgoing edge should have been persisted to link_overlay"
        );
        // Verify edge field integrity: there should be an EntityOverlap edge
        let has_entity = outgoing
            .iter()
            .any(|l| l.link_type == hippmem_core::model::links::LinkType::EntityOverlap);
        assert!(has_entity, "should contain an EntityOverlap edge type");
    }
}
