//! acceptance tests: AssociationKeys generation

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::SemanticSignature;
use hippmem_core::model::understanding::{
    CausalClaim, CausalKind, EntityMention, EntityType, GoalFrame, GoalStatus, TopicTag,
};
use hippmem_core::model::unit::{ContentType, Language, MemoryContent, WriteContext};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_write::keys::generate_keys;

fn make_sig() -> SemanticSignature {
    SemanticSignature {
        lexical_simhash: [1, 2, 3, 4],
        dense_embedding_ref: None,
        binary_code: [0, 0],
        topic_minhash: [0u32; 16],
    }
}

fn make_understanding() -> hippmem_core::model::understanding::MemoryUnderstanding {
    hippmem_core::model::understanding::MemoryUnderstanding {
        entities: vec![EntityMention {
            text: "Rust".into(),
            canonical: "rust".into(),
            entity_type: EntityType::Other,
            span: None,
            confidence: UnitScore::new(0.8),
        }],
        events: vec![],
        goals: vec![GoalFrame {
            description: "Learn Rust".into(),
            status: GoalStatus::Active,
            constraints: vec![],
            confidence: UnitScore::new(0.5),
        }],
        decisions: vec![],
        preferences: vec![],
        emotions: vec![],
        causal_claims: vec![CausalClaim {
            cause: "compilation too slow".into(),
            effect: "switch to redb".into(),
            kind: CausalKind::Explicit,
            evidence_span: None,
            confidence: UnitScore::new(0.7),
        }],
        contradictions: vec![],
        topics: vec![TopicTag {
            label: "storage".into(),
            confidence: UnitScore::new(0.6),
        }],
        importance: UnitScore::new(0.5),
        confidence: UnitScore::new(0.5),
    }
}

fn make_content() -> MemoryContent {
    MemoryContent {
        raw: "Because RocksDB compilation was too slow, we switched to redb".into(),
        summary: None,
        normalized: None,
        language: Language::Zh,
        content_type: ContentType::Decision,
    }
}

fn make_context() -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(42),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: Timestamp(1_700_000_000_000), // fixed time
        preceding_memory_ids: vec![MemoryId(100)],
        source_refs: vec![],
    }
}

/// Deterministic: same input produces same output.
#[test]
fn keys_are_deterministic() {
    let content = make_content();
    let understanding = make_understanding();
    let context = make_context();
    let sig = make_sig();

    let k1 = generate_keys(&content, &understanding, &context, &sig).unwrap();
    let k2 = generate_keys(&content, &understanding, &context, &sig).unwrap();

    assert_eq!(k1.entity_keys, k2.entity_keys);
    assert_eq!(k1.temporal_keys, k2.temporal_keys);
    assert_eq!(k1.topic_keys, k2.topic_keys);
    assert_eq!(k1.goal_keys, k2.goal_keys);
    assert_eq!(k1.causal_keys, k2.causal_keys);
    assert_eq!(k1.lexical_signature.simhash, k2.lexical_signature.simhash);
}

/// entity_keys is non-empty (when entities are extracted).
#[test]
fn entity_keys_from_understanding() {
    let content = make_content();
    let understanding = make_understanding();
    let context = make_context();
    let sig = make_sig();

    let keys = generate_keys(&content, &understanding, &context, &sig).unwrap();
    assert!(
        !keys.entity_keys.is_empty(),
        "entity_keys should be produced when entities are present"
    );
}

/// temporal_keys is non-empty.
#[test]
fn temporal_keys_from_context() {
    let content = make_content();
    let understanding = make_understanding();
    let context = make_context();
    let sig = make_sig();

    let keys = generate_keys(&content, &understanding, &context, &sig).unwrap();
    assert!(
        !keys.temporal_keys.is_empty(),
        "temporal_keys should be produced when a time context is present"
    );
}

/// goal_keys are generated from goal frames.
#[test]
fn goal_keys_from_understanding() {
    let content = make_content();
    let understanding = make_understanding();
    let context = make_context();
    let sig = make_sig();

    let keys = generate_keys(&content, &understanding, &context, &sig).unwrap();
    assert!(
        !keys.goal_keys.is_empty(),
        "goal_keys should be produced when a GoalFrame is present"
    );
}

/// causal_keys are generated from causal claims.
#[test]
fn causal_keys_from_claims() {
    let content = make_content();
    let understanding = make_understanding();
    let context = make_context();
    let sig = make_sig();

    let keys = generate_keys(&content, &understanding, &context, &sig).unwrap();
    assert!(
        !keys.causal_keys.is_empty(),
        "causal_keys should be produced when a CausalClaim is present"
    );
}

/// Lexical signature is non-zero (for non-empty text).
#[test]
fn lexical_signature_non_zero() {
    let content = make_content();
    let understanding = make_understanding();
    let context = make_context();
    let sig = make_sig();

    let keys = generate_keys(&content, &understanding, &context, &sig).unwrap();
    let simhash = keys.lexical_signature.simhash;
    assert!(
        simhash.iter().any(|&x| x != 0),
        "non-empty text should produce a non-zero SimHash"
    );
}
