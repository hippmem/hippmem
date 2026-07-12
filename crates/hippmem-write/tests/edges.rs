//! Edge construction + staged write acceptance tests.

use hippmem_core::config::AlgoParams;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::MatchDimension;
use hippmem_core::model::links::SemanticSignature;
use hippmem_core::model::understanding::{EntityMention, EntityType, MemoryUnderstanding};
use hippmem_core::model::unit::{ContentType, Language, MemoryContent, MemoryStage, WriteContext};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_write::candidates::CandidateResult;
use hippmem_write::edges::{build_edges, EdgeBuildParams};
use hippmem_write::staged::{raw_to_indexed, StagedWriteInput};

fn now() -> Timestamp {
    Timestamp(1_700_000_000_000)
}

/// Edge construction: dedup, no self-loop.
#[test]
fn no_self_loop() {
    let c = make_cand(vec![MatchDimension::Entity], 1);
    let r = build_edges(
        MemoryId(1),
        MemoryId(1),
        &c,
        2,
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
        &[],
        now(),
        1000,
    );
    assert!(r.created_links.is_empty());
}

/// Edge construction: multi-dim hit produces a strong edge.
#[test]
fn multi_dim_produces_edge() {
    let c = make_cand(
        vec![
            MatchDimension::Entity,
            MatchDimension::Topic,
            MatchDimension::Temporal,
        ],
        3,
    );
    let r = build_edges(
        MemoryId(1),
        MemoryId(2),
        &c,
        3,
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
        &[],
        now(),
        1000,
    );
    assert!(!r.created_links.is_empty());
}

/// Staged write produces an indexed MemoryUnit.
#[test]
fn staged_write_produces_indexed() {
    let input = StagedWriteInput {
        id: MemoryId(42),
        content: MemoryContent {
            raw: "Because compilation was too slow, we switched to redb".into(),
            summary: None,
            normalized: None,
            language: Language::Zh,
            content_type: ContentType::Decision,
        },
        understanding: MemoryUnderstanding {
            entities: vec![EntityMention {
                text: "redb".into(),
                canonical: "redb".into(),
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
            local_time: now(),
            preceding_memory_ids: vec![],
            source_refs: vec![],
        },
        semantic: SemanticSignature {
            lexical_simhash: [1, 2, 3, 4],
            dense_embedding_ref: None,
            binary_code: [0xABCD, 0x1234],
            topic_minhash: [0u32; 16],
        },
    };
    let output = raw_to_indexed(
        input,
        &[],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();
    assert_eq!(output.unit.stage, MemoryStage::Indexed);
    assert_eq!(output.unit.id, MemoryId(42));
}

/// Shared entity produces an edge (end-to-end).
#[test]
fn shared_entity_e2e_link() {
    let make = |id: u128, text: &str| -> StagedWriteInput {
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
                    text: "Rust".into(),
                    canonical: "rust".into(),
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
                local_time: now(),
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
    };

    let first = raw_to_indexed(
        make(1, "Rust programming"),
        &[],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();
    let second = raw_to_indexed(
        make(2, "I love Rust"),
        &[first.unit],
        &EdgeBuildParams::default(),
        &AlgoParams::default(),
    )
    .unwrap();
    assert!(
        !second.created_links.is_empty(),
        "shared entity Rust should build an edge"
    );
}

fn make_cand(dims: Vec<MatchDimension>, n: usize) -> CandidateResult {
    CandidateResult {
        matched_dimensions: dims,
        entity_jaccard: if n >= 1 { 0.6 } else { 0.0 },
        topic_jaccard: if n >= 2 { 0.5 } else { 0.0 },
        temporal_overlap: if n >= 3 { 1 } else { 0 },
        goal_jaccard: 0.0,
        event_jaccard: 0.0,
        causal_overlap: 0,
        emotion_overlap: 0,
        importance_value: 0.0,
        co_context_score: 0.0,
        lexical_similarity: 0.8,
        semantic_binary_similarity: 0.0,
    }
}
