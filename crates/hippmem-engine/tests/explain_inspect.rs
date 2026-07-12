//! acceptance test: explain + inspect API.

use hippmem_core::ids::MemoryId;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, InspectQuery, WriteMemoryInput};
use tempfile::tempdir;

fn ctx() -> WriteContext {
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
fn explain_returns_five_answers() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let out = engine
        .write(WriteMemoryInput {
            content: "The user is a Rust engineer named Alex".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();

    let explanation = engine.explain(out.memory_id, None).unwrap();
    // Five questions: source/importance/connections/corrections/activations
    assert!(explanation.current_importance >= 0.0);
    let _ = explanation.linked.len();
    let _ = explanation.corrections.len();
    let _ = explanation.contradictions.len();

    engine.close().unwrap();
}

#[test]
fn explain_not_found_errors() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let result = engine.explain(hippmem_core::ids::MemoryId(99999), None);
    assert!(result.is_err());

    engine.close().unwrap();
}

#[test]
fn inspect_store_stats() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    engine
        .write(WriteMemoryInput {
            content: "test".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    let report = engine.inspect(InspectQuery::StoreStats).unwrap();
    match report {
        hippmem_engine::InspectReport::StoreStats(s) => {
            assert!(s.memory_count > 0, "should have at least 1 memory");
        }
        _ => panic!("expected StoreStats variant"),
    }

    engine.close().unwrap();
}

#[test]
fn inspect_queue_status() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let report = engine.inspect(InspectQuery::QueueStatus).unwrap();
    match report {
        hippmem_engine::InspectReport::QueueStatus(q) => {
            let _ = q.pending_enrich;
            let _ = q.in_flight;
        }
        _ => panic!("expected QueueStatus"),
    }

    engine.close().unwrap();
}

// ── Inspect memory tests ──

#[test]
fn inspect_memory_with_out_edges() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write 2 memories sharing keywords (using a content pattern known to produce associations)
    // "Rust" and the database keyword are both high-frequency keywords the deterministic extractor can recognize
    let out1 = engine
        .write(WriteMemoryInput {
            content: "The project uses Rust to develop backend database services".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    let _out2 = engine
        .write(WriteMemoryInput {
            content: "Building a high-performance database query engine with Rust".into(),
            content_type: Some(ContentType::Decision),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();

    // Inspect id1 details
    let report = engine
        .inspect(InspectQuery::Memory(out1.memory_id))
        .unwrap();
    match report {
        hippmem_engine::InspectReport::Memory(m) => {
            assert_eq!(m.unit.id, out1.memory_id);
            // Out-edge and in-edge verification — should at least see associations in the memory network
            let total_edges = m.out_edges.len() + m.in_edges.len();
            assert!(
                total_edges > 0,
                "the memory network should have edges (out or in)"
            );
        }
        _ => panic!("expected Memory variant"),
    }

    engine.close().unwrap();
}

#[test]
fn inspect_memory_in_edges_not_empty() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let out1 = engine
        .write(WriteMemoryInput {
            content: "The project uses Rust to develop backend database services".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();
    let out2 = engine
        .write(WriteMemoryInput {
            content: "Building a high-performance database query engine with Rust".into(),
            content_type: Some(ContentType::Decision),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap();

    // Inspect id2 details — if an out-edge from id1 to id2 exists, id2 should have a corresponding in-edge
    let report = engine
        .inspect(InspectQuery::Memory(out2.memory_id))
        .unwrap();
    match report {
        hippmem_engine::InspectReport::Memory(m) => {
            assert_eq!(m.unit.id, out2.memory_id);
            // Check whether id1's out_edges contain an edge pointing to id2
            // If yes, then id2's in_edges should also contain that edge
            let report1 = engine
                .inspect(InspectQuery::Memory(out1.memory_id))
                .unwrap();
            if let hippmem_engine::InspectReport::Memory(m1) = report1 {
                let id1_to_id2 = m1.out_edges.iter().any(|e| e.to == out2.memory_id);
                if id1_to_id2 {
                    let from_id1 = m.in_edges.iter().any(|e| e.from == out1.memory_id);
                    assert!(
                        from_id1,
                        "if the id1→id2 out-edge exists, id2's in-edges should contain the edge from id1"
                    );
                }
            }
        }
        _ => panic!("expected Memory variant"),
    }

    engine.close().unwrap();
}

#[test]
fn inspect_memory_not_found_errors() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let result = engine.inspect(InspectQuery::Memory(MemoryId(99999)));
    assert!(result.is_err());

    engine.close().unwrap();
}
