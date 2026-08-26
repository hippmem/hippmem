//! acceptance test: delete API (M3, memory-management proposal).
//!
//! Tests cascade deletion: kv + inverted indexes (retrieval no longer sees
//! the memory), graph in-edges held by other units, idempotency for unknown
//! ids, and counting of deleted memories / removed edges.

use hippmem_core::ids::MemoryId;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{DeleteInput, Engine, EngineConfig, RetrieveInput, WriteMemoryInput};
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

fn retrieve_ctx() -> hippmem_engine::RetrieveContext {
    hippmem_engine::RetrieveContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        recent_memory_ids: vec![],
    }
}

fn open_engine(dir: &tempfile::TempDir) -> Engine {
    Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap()
}

fn write(engine: &Engine, content: &str) -> u128 {
    engine
        .write(WriteMemoryInput {
            content: content.into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap()
        .memory_id
        .0
}

fn retrieve_ids(engine: &Engine, query: &str) -> Vec<u128> {
    engine
        .retrieve(RetrieveInput {
            query: query.into(),
            context: retrieve_ctx(),
            top_k: 10,
            max_hops: Some(2),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap()
        .results
        .iter()
        .map(|r| r.memory.id.0)
        .collect()
}

/// Extracts the digit run of a top-level field, e.g. `"id":21612...` → u128.
/// Done at string level: memory ids exceed u64, so parsing the JSON into a
/// `serde_json::Value` would degrade them to f64 and lose precision.
fn parse_u128_field(line: &str, field: &str) -> u128 {
    let marker = format!("\"{}\":", field);
    let start = line.find(&marker).expect("field present") + marker.len();
    let digits: String = line[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().expect("digit run parses")
}

/// Every dump line is one memory unit; returns (id, links) pairs with the
/// target ids of each outgoing link.
fn dump_links(engine: &Engine) -> Vec<(u128, Vec<u128>)> {
    let json = engine
        .dump(hippmem_engine::DumpInput::default())
        .unwrap()
        .json
        .unwrap();
    json.trim()
        .lines()
        .map(|line| {
            let id = parse_u128_field(line, "id");
            let targets = line
                .split("\"target_id\":")
                .skip(1)
                .map(|rest| {
                    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    digits.parse().expect("target id digits")
                })
                .collect();
            (id, targets)
        })
        .collect()
}

#[test]
fn delete_removes_memory_and_indexes() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    // Two memories sharing the entity "小明" — both must surface for the
    // entity query (shared posting lists in the inverted indexes).
    let a = write(&engine, "小明住在北京海淀区。");
    let b = write(&engine, "小明和李华是高中同学。");
    let before = retrieve_ids(&engine, "小明");
    assert!(before.contains(&a), "A should be retrievable before delete");
    assert!(before.contains(&b), "B should be retrievable before delete");

    // Delete A: kv + inverted indexes must be cleaned.
    let out = engine
        .delete(DeleteInput {
            memory_ids: vec![MemoryId(a)],
        })
        .unwrap();
    assert_eq!(out.deleted, 1, "exactly one memory deleted");

    let after = retrieve_ids(&engine, "小明");
    assert!(
        !after.contains(&a),
        "deleted memory must not be retrievable anymore"
    );
    assert!(
        after.contains(&b),
        "the other memory sharing the entity must survive"
    );

    let listed = engine
        .list(hippmem_engine::ListInput {
            limit: 20,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(listed.total, 1, "only B remains in the database");

    engine.close().unwrap();
}

#[test]
fn delete_cascades_incoming_edges() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    // Similar memories get an EntityOverlap edge at write time; A must hold
    // at least one incoming edge from another unit.
    let a = write(&engine, "小明住在北京海淀区。");
    let b = write(&engine, "小明住在北京市海淀区。");
    let pairs = dump_links(&engine);
    let incoming_before: Vec<u128> = pairs
        .iter()
        .filter(|(id, links)| *id != a && links.contains(&a))
        .map(|(id, _)| *id)
        .collect();
    assert!(
        !incoming_before.is_empty(),
        "precondition: some unit links to A (auto-built edge)"
    );
    let _ = b;

    let out = engine
        .delete(DeleteInput {
            memory_ids: vec![MemoryId(a)],
        })
        .unwrap();
    assert_eq!(out.deleted, 1);
    assert!(
        out.edges_removed >= 1,
        "incoming edges should be counted (got {})",
        out.edges_removed
    );

    // No dangling references: A is gone and no surviving unit links to it.
    let pairs = dump_links(&engine);
    assert!(
        pairs.iter().all(|(id, _)| *id != a),
        "deleted memory must not appear in the dump"
    );
    assert!(
        pairs.iter().all(|(_, links)| !links.contains(&a)),
        "no surviving unit may keep an edge to the deleted memory"
    );
    assert!(
        pairs.iter().any(|(id, _)| *id != a),
        "the other memory must survive the cascade"
    );

    engine.close().unwrap();
}

#[test]
fn delete_idempotent_for_unknown_ids() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    // Empty database: unknown id is a no-op, not an error.
    let out = engine
        .delete(DeleteInput {
            memory_ids: vec![MemoryId(12345)],
        })
        .unwrap();
    assert_eq!(out.deleted, 0);

    // Deleted twice: second time the id is unknown → 0.
    let a = write(&engine, "小明住在北京海淀区。");
    engine
        .delete(DeleteInput {
            memory_ids: vec![MemoryId(a)],
        })
        .unwrap();
    let again = engine
        .delete(DeleteInput {
            memory_ids: vec![MemoryId(a)],
        })
        .unwrap();
    assert_eq!(
        again.deleted, 0,
        "deleting an already-deleted id is a no-op"
    );

    engine.close().unwrap();
}
