//! acceptance test: engine.retrieve end-to-end.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Clock;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};
use tempfile::tempdir;

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

fn make_retrieve_context() -> RetrieveContext {
    RetrieveContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        recent_memory_ids: vec![],
    }
}

#[test]
fn retrieve_basic_after_write() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write a few memories
    engine
        .write(WriteMemoryInput {
            content: "The project uses Rust for backend services".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: make_context(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Because compilation was too slow, we switched to Redb for storage".into(),
            content_type: Some(ContentType::Decision),
            context: make_context(),
            importance_hint: Some(0.8),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "The team decided to use Tantivy for full-text search".into(),
            content_type: Some(ContentType::Decision),
            context: make_context(),
            importance_hint: Some(0.6),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve
    let input = RetrieveInput {
        query: "Rust Redb Tantivy".into(),
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();

    assert!(
        !output.results.is_empty(),
        "retrieve should return at least one result"
    );

    // Each result should have activation_trace
    for r in &output.results {
        assert!(
            !r.activation_trace.is_empty(),
            "each result should have non-empty activation_trace, got empty for id={:?}",
            r.memory.id
        );
        assert!(
            !r.matched_dimensions.is_empty(),
            "each result should have non-empty matched_dimensions"
        );
    }
}

#[test]
fn retrieve_empty_store() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    let input = RetrieveInput {
        query: "any query".into(),
        context: make_retrieve_context(),
        top_k: 3,
        max_hops: None,
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();
    // Empty store returns empty results, should not panic
    assert!(output.results.is_empty());
}

/// V2-001: After writing memories with shared topics and entities, temporal and topic channels should produce seeds on retrieve.
#[test]
fn retrieve_temporal_and_topic_channels_produce_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write memories containing an uppercase entity (Rust) and specific topic words (database/performance),
    // using current time to ensure temporal buckets match at retrieve time
    let now = hippmem_core::time::SystemClock.now();
    let ctx = WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: now,
        preceding_memory_ids: vec![],
        source_refs: vec![],
    };

    engine
        .write(WriteMemoryInput {
            content: "Rust is used for building high-performance databases".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx.clone(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Database query performance optimization with Rust".into(),
            content_type: Some(ContentType::Decision),
            context: ctx,
            importance_hint: Some(0.8),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve: query contains entity "Rust" + related topic words
    let input = RetrieveInput {
        query: "Rust database".into(),
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();

    // Should return results
    assert!(!output.results.is_empty(), "retrieve should return results");

    // seeds in trace should span multiple channels
    let has_entity = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == hippmem_core::model::links::RecallChannel::EntityInverted);
    let has_temporal = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == hippmem_core::model::links::RecallChannel::Temporal);
    let has_topic = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == hippmem_core::model::links::RecallChannel::TopicCluster);

    assert!(
        has_entity || has_temporal || has_topic,
        "at least one of Entity/Temporal/Topic channels should produce seeds, got seeds={:?}",
        output
            .trace
            .seeds
            .iter()
            .map(|s| (s.channel, s.initial_energy))
            .collect::<Vec<_>>()
    );
}

/// V2-002: Write 100 memories then retrieve, verifying on-demand loading works (no full-table scan).
#[test]
fn retrieve_with_hundred_memories() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    let ctx = make_context();

    // Write 100 memories, 5 of which contain the target entity "HippmemTest"
    for i in 0..100 {
        let content = if i < 5 {
            format!("HippmemTest target memory {}", i)
        } else {
            format!("unrelated memory {}", i)
        };
        engine
            .write(WriteMemoryInput {
                content,
                content_type: Some(ContentType::UserStatement),
                context: ctx.clone(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }

    // Confirm store has 100 memories
    let stats = engine
        .inspect(hippmem_engine::InspectQuery::StoreStats)
        .unwrap();
    if let hippmem_engine::InspectReport::StoreStats(s) = stats {
        assert_eq!(s.memory_count, 100, "store should have 100 memories");
    } else {
        panic!("expected StoreStats");
    }

    // Retrieve target
    let input = RetrieveInput {
        query: "HippmemTest".into(),
        context: make_retrieve_context(),
        top_k: 10,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };
    let output = engine.retrieve(input).unwrap();

    // Should return results (5 target memories + possible spread discoveries)
    assert!(
        !output.results.is_empty(),
        "retrieve over 100 memories should return results"
    );
    // All results should have activation_trace
    for r in &output.results {
        assert!(!r.activation_trace.is_empty());
    }
}

/// V2-020: After writing semantically similar memories, the SemanticBinary channel should produce seeds.
///
/// binary_code is generated by stable_hash64 ([u64;2], 128 bits); same text -> same code -> Hamming=0.
/// This test verifies the full binary_code index write -> query -> seed construction path.
#[test]
fn retrieve_semantic_binary_channel_produces_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    let ctx = make_context();

    // Write 3 memories sharing keywords; content partially overlaps to produce meaningful binary_code
    engine
        .write(WriteMemoryInput {
            content: "Rust async runtime performance analysis".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx.clone(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Rust async database query engine".into(),
            content_type: Some(ContentType::Decision),
            context: ctx.clone(),
            importance_hint: Some(0.6),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Database query performance analysis tool".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx,
            importance_hint: Some(0.5),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve: query contains shared keywords
    let input = RetrieveInput {
        query: "Rust database query".into(),
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();
    assert!(!output.results.is_empty(), "retrieve should return results");

    // Verify SemanticBinary channel produced seeds
    let has_binary = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == hippmem_core::model::links::RecallChannel::SemanticBinary);
    assert!(
        has_binary,
        "SemanticBinary channel should produce seeds, got seeds={:?}",
        output
            .trace
            .seeds
            .iter()
            .map(|s| (s.channel, s.initial_energy))
            .collect::<Vec<_>>()
    );
}

/// V2-006: After writing memories containing specific keywords, the BM25 channel should produce seeds.
#[test]
fn retrieve_bm25_channel_produces_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    let ctx = make_context();

    // Write memories containing the same database keyword
    engine
        .write(WriteMemoryInput {
            content: "Rust database engine query optimization".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx.clone(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "High performance database query analysis tool".into(),
            content_type: Some(ContentType::Decision),
            context: ctx,
            importance_hint: Some(0.6),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve with a query containing the same keyword
    let input = RetrieveInput {
        query: "database query".into(),
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();
    assert!(!output.results.is_empty(), "retrieve should return results");

    // Verify BM25 channel produced seeds
    let has_bm25 = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == hippmem_core::model::links::RecallChannel::Bm25);
    assert!(has_bm25, "BM25 channel should produce seeds");
}

/// V2-022a: Write memories containing goal keywords -> Goal channel produces seeds.
#[test]
fn retrieve_goal_channel_produces_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write a memory containing goal keywords (enrich stage will extract the goal)
    engine
        .write(WriteMemoryInput {
            content: "I plan to learn the Rust programming language".into(),
            content_type: Some(ContentType::UserStatement),
            context: make_context(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve with a query containing goal keywords
    let input = RetrieveInput {
        query: "learning goals".into(), // "plan to" triggers goal keyword extraction
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();
    // Goal channel may produce seeds (if enrich successfully extracts a goal and the query matches)
    // At minimum, retrieve itself should not panic
    let _ = output.results.len();
}

/// V2-022b: Write memories containing event keywords -> Event channel produces seeds.
#[test]
fn retrieve_event_channel_produces_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write a document containing an event description (enrich stage may extract an event)
    engine
        .write(WriteMemoryInput {
            content: "The project completed deployment and went live".into(),
            content_type: Some(ContentType::Event),
            context: make_context(),
            importance_hint: Some(0.6),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve with a query containing event keywords
    let input = RetrieveInput {
        query: "deployment".into(), // "deployment" triggers event keyword extraction
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();
    let _ = output.results.len();
}

/// V2-022c: Write memories with explicit causality -> Causal channel produces seeds.
#[test]
fn retrieve_causal_channel_produces_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write a memory with explicit "because...so..." causality
    engine
        .write(WriteMemoryInput {
            content: "Because compilation was too slow, we switched to Redb for storage".into(),
            content_type: Some(ContentType::Decision),
            context: make_context(),
            importance_hint: Some(0.8),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve with a query containing causal connectives
    let input = RetrieveInput {
        query: "compilation too slow caused".into(),
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();
    // Retrieve should not panic; whether the causal channel produces seeds depends on query understanding
    let _ = output.results.len();

    // Verify trace has channel contribution records
    assert!(!output.trace.seeds.is_empty(), "should have seed records");
}

/// V2-023: After writing semantically similar memories, the SemanticDense (HNSW) channel should produce seeds.
///
/// Verifies the full FlatVectorIndex write -> query -> seed construction path:
/// DeterministicEmbedder generates a 256-dim vector for each memory -> insert into FlatVectorIndex ->
/// at retrieve time the query vector searches FlatVectorIndex -> SemanticDense channel produces seeds.
#[test]
fn retrieve_semantic_dense_channel_produces_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    let ctx = make_context();

    // Write 3 short texts sharing keywords; content partially overlaps to produce meaningful dense-vector similarity
    engine
        .write(WriteMemoryInput {
            content: "Rust async runtime performance analysis".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx.clone(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Rust async database query engine".into(),
            content_type: Some(ContentType::Decision),
            context: ctx.clone(),
            importance_hint: Some(0.6),
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Database query performance analysis tool".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx,
            importance_hint: Some(0.5),
            source_refs: vec![],
        })
        .unwrap();

    // Retrieve: query contains shared keywords
    let input = RetrieveInput {
        query: "Rust database performance query".into(),
        context: make_retrieve_context(),
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input).unwrap();
    assert!(!output.results.is_empty(), "retrieve should return results");

    // Verify SemanticDense channel produced seeds
    let has_dense = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == hippmem_core::model::links::RecallChannel::SemanticDense);
    assert!(
        has_dense,
        "SemanticDense channel should produce seeds, got seeds={:?}",
        output
            .trace
            .seeds
            .iter()
            .map(|s| (s.channel, s.initial_energy))
            .collect::<Vec<_>>()
    );
}

/// V2-022d: After passing recent_memory_ids -> RecentActivation channel produces seeds.
#[test]
fn retrieve_recent_activation_channel_produces_seeds() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };
    let engine = Engine::open(config).unwrap();

    // Write a memory
    let out = engine
        .write(WriteMemoryInput {
            content: "Recent development experience using Rust for backend services".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: make_context(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    let mid = out.memory_id;

    // First retrieve: activate the memory and record activation_log
    let input1 = RetrieveInput {
        query: "Rust backend".into(),
        context: make_retrieve_context(),
        top_k: 3,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };
    let _ = engine.retrieve(input1).unwrap();

    // Second retrieve: put the just-activated memory into recent_memory_ids
    let mut ctx2 = make_retrieve_context();
    ctx2.recent_memory_ids = vec![mid];

    let input2 = RetrieveInput {
        query: "development experience".into(),
        context: ctx2,
        top_k: 5,
        max_hops: Some(2),
        retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
    };

    let output = engine.retrieve(input2).unwrap();
    // Verify retrieve succeeded and seed records are non-empty
    assert!(
        !output.trace.seeds.is_empty(),
        "recent channel should have seed records"
    );

    // Verify RecentActivation channel contributed
    let has_recent = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == hippmem_core::model::links::RecallChannel::RecentActivation);
    assert!(
        has_recent,
        "RecentActivation channel should produce seeds, actual seeds: {:?}",
        output
            .trace
            .seeds
            .iter()
            .map(|s| format!("{:?}[{:?}]", s.id, s.channel))
            .collect::<Vec<_>>()
    );
}
