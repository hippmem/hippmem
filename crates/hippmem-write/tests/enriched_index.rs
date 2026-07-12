//! acceptance tests: write back goal/event/causal inverted indexes
//! at the Enriched stage.
//!
//! Verifies:
//! - after enriched: goal_keys -> goal_index
//! - after enriched: event_keys -> event_index
//! - after enriched: causal_keys -> causal_index (or Causal edges in
//!   link_overlay)
//! - existing memories can rebuild their indexes after Reindex

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{AssociationKeys, LexicalSignature, SemanticSignature};
use hippmem_core::model::understanding::{
    CausalClaim, CausalKind, EntityMention, EntityType, EventFrame, GoalFrame, GoalStatus,
    MemoryUnderstanding,
};
use hippmem_core::model::unit::{
    ContentType, Language, MemoryContent, MemoryLifecycle, MemoryStage, MemoryUnit, WriteContext,
};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_store::kv::InvertedIndex;
use hippmem_store::store::{RedbStore, Store};
use hippmem_write::understanding::index_enriched_keys;
use tempfile::TempDir;

/// Build a MemoryUnit containing goal/event/causal understanding results
/// (simulating the post-enriched state).
fn make_enriched_unit(
    id: u128,
    goal_desc: &str,
    event_action: &str,
    cause: &str,
    effect: &str,
) -> MemoryUnit {
    let goals = if goal_desc.is_empty() {
        vec![]
    } else {
        vec![GoalFrame {
            description: goal_desc.into(),
            status: GoalStatus::Active,
            constraints: vec![],
            confidence: UnitScore::new(0.8),
        }]
    };

    let events = if event_action.is_empty() {
        vec![]
    } else {
        vec![EventFrame {
            action: event_action.into(),
            participants: vec![],
            occurred_at: None,
            outcome: None,
            confidence: UnitScore::new(0.7),
        }]
    };

    let causal_claims = if cause.is_empty() || effect.is_empty() {
        vec![]
    } else {
        vec![CausalClaim {
            cause: cause.into(),
            effect: effect.into(),
            kind: CausalKind::Explicit,
            confidence: UnitScore::new(0.6),
            evidence_span: None,
        }]
    };

    let understanding = MemoryUnderstanding {
        entities: vec![EntityMention {
            text: "test".into(),
            canonical: "test".into(),
            entity_type: EntityType::Other,
            span: None,
            confidence: UnitScore::new(0.8),
        }],
        events,
        goals,
        decisions: vec![],
        preferences: vec![],
        emotions: vec![],
        causal_claims,
        contradictions: vec![],
        topics: vec![],
        importance: UnitScore::new(0.5),
        confidence: UnitScore::new(0.7),
    };

    MemoryUnit {
        schema_version: 1,
        id: MemoryId(id),
        created_at: Timestamp(1_700_000_000_000),
        updated_at: Timestamp(1_700_000_000_000),
        content: MemoryContent {
            raw: "test content".into(),
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
        understanding,
        association_keys: AssociationKeys {
            entity_keys: vec![1],
            temporal_keys: vec![1],
            lexical_signature: LexicalSignature { simhash: [0; 4] },
            semantic_signature: SemanticSignature {
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
        lifecycle: MemoryLifecycle::Active,
        provenance: hippmem_core::model::unit::Provenance {
            origin: hippmem_core::model::unit::SourceKind::Conversation,
            generated_by: hippmem_core::model::unit::GeneratedBy::UserDirect,
            reliability: UnitScore::new(0.5),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: MemoryStage::Enriched,
    }
}

/// Open a temporary database and return an InvertedIndex.
fn open_temp_inverted(dir: &TempDir) -> (RedbStore, InvertedIndex) {
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("should be able to create a temporary database");
    let inverted = InvertedIndex::new(store.db_arc());
    (store, inverted)
}

// ═══════════════════════════════════════════════════════════════════
// Acceptance test 1: goal_keys -> goal_index
// ═══════════════════════════════════════════════════════════════════

#[test]
fn enriched_writes_goal_keys_to_goal_index() {
    let tmp = TempDir::new().expect("tempdir");
    let (_store, inverted) = open_temp_inverted(&tmp);

    let unit = make_enriched_unit(100, "Learn Rust programming", "", "", "");

    // Call the enriched -> index writeback
    index_enriched_keys(&unit, &inverted, 100).expect("index_enriched_keys should succeed");

    // Verify: goal_index can read the corresponding key -> id mapping
    let goal_key = xxhash_rust::xxh3::xxh3_64("Learn Rust programming".as_bytes());
    let goal_ids = inverted
        .get_goal(&goal_key)
        .expect("get_goal should succeed");
    assert!(
        goal_ids.contains(&100),
        "goal_index should contain memory_id=100, got: {:?}",
        goal_ids
    );
}

// ═══════════════════════════════════════════════════════════════════
// Acceptance test 2: event_keys -> event_index
// ═══════════════════════════════════════════════════════════════════

#[test]
fn enriched_writes_event_keys_to_event_index() {
    let tmp = TempDir::new().expect("tempdir");
    let (_store, inverted) = open_temp_inverted(&tmp);

    let unit = make_enriched_unit(200, "", "Completed deployment", "", "");

    index_enriched_keys(&unit, &inverted, 200).expect("index_enriched_keys should succeed");

    let event_key = xxhash_rust::xxh3::xxh3_64("Completed deployment".as_bytes());
    let event_ids = inverted
        .get_event(&event_key)
        .expect("get_event should succeed");
    assert!(
        event_ids.contains(&200),
        "event_index should contain memory_id=200, got: {:?}",
        event_ids
    );
}

// ═══════════════════════════════════════════════════════════════════
// Acceptance test 3: causal_keys -> causal_index
// ═══════════════════════════════════════════════════════════════════

#[test]
fn enriched_writes_causal_keys_to_causal_index() {
    let tmp = TempDir::new().expect("tempdir");
    let (_store, inverted) = open_temp_inverted(&tmp);

    let unit = make_enriched_unit(300, "", "", "Learn Rust", "Improve efficiency");

    index_enriched_keys(&unit, &inverted, 300).expect("index_enriched_keys should succeed");

    // causal_key = hash(cause ++ " -> " ++ effect)
    let causal_input = "Learn Rust -> Improve efficiency";
    let causal_key = xxhash_rust::xxh3::xxh3_64(causal_input.as_bytes());
    let causal_ids = inverted
        .get_causal(&causal_key)
        .expect("get_causal should succeed");
    assert!(
        causal_ids.contains(&300),
        "causal_index should contain memory_id=300, got: {:?}",
        causal_ids
    );
}

// ═══════════════════════════════════════════════════════════════════
// Acceptance test 4: write back multiple dimensions simultaneously
// ═══════════════════════════════════════════════════════════════════

#[test]
fn enriched_writes_all_three_dimensions_simultaneously() {
    let tmp = TempDir::new().expect("tempdir");
    let (_store, inverted) = open_temp_inverted(&tmp);

    let unit = make_enriched_unit(
        400,
        "Master Rust",
        "Project launch",
        "Learn Rust",
        "Project launch",
    );

    index_enriched_keys(&unit, &inverted, 400).expect("index_enriched_keys should succeed");

    // goal
    let gk = xxhash_rust::xxh3::xxh3_64("Master Rust".as_bytes());
    assert!(inverted.get_goal(&gk).unwrap().contains(&400));

    // event
    let ek = xxhash_rust::xxh3::xxh3_64("Project launch".as_bytes());
    assert!(inverted.get_event(&ek).unwrap().contains(&400));

    // causal
    let ck = xxhash_rust::xxh3::xxh3_64("Learn Rust -> Project launch".as_bytes());
    assert!(inverted.get_causal(&ck).unwrap().contains(&400));
}

// ═══════════════════════════════════════════════════════════════════
// Acceptance test 5: no write when there are no goal/event/causal
// (idempotent / no side effect)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn enriched_noop_when_no_goals_events_causals() {
    let tmp = TempDir::new().expect("tempdir");
    let (_store, inverted) = open_temp_inverted(&tmp);

    // Empty goal/event/causal
    let unit = make_enriched_unit(500, "", "", "", "");

    let result = index_enriched_keys(&unit, &inverted, 500);
    assert!(
        result.is_ok(),
        "empty keys should also succeed (no side effect)"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Acceptance test 6: existing-memory Reindex — rebuild indexes from the
// new understanding
// ═══════════════════════════════════════════════════════════════════

#[test]
fn reindex_rebuilds_goal_event_causal_from_existing_unit() {
    let tmp = TempDir::new().expect("tempdir");
    let (_store, inverted) = open_temp_inverted(&tmp);

    // Simulate an existing memory: already enriched but indexes never written
    let unit = make_enriched_unit(
        600,
        "Learn English",
        "Meeting completed",
        "Requirements clear",
        "Plan confirmed",
    );

    // Reindex: rebuild indexes from the existing MemoryUnit
    index_enriched_keys(&unit, &inverted, 600).expect("reindex should succeed");

    // Verify all three dimensions are queryable
    let gk = xxhash_rust::xxh3::xxh3_64("Learn English".as_bytes());
    assert!(
        inverted.get_goal(&gk).unwrap().contains(&600),
        "after Reindex, goal_index should contain the existing memory"
    );

    let ek = xxhash_rust::xxh3::xxh3_64("Meeting completed".as_bytes());
    assert!(
        inverted.get_event(&ek).unwrap().contains(&600),
        "after Reindex, event_index should contain the existing memory"
    );

    let ck = xxhash_rust::xxh3::xxh3_64("Requirements clear -> Plan confirmed".as_bytes());
    assert!(
        inverted.get_causal(&ck).unwrap().contains(&600),
        "after Reindex, causal_index should contain the existing memory"
    );
}
