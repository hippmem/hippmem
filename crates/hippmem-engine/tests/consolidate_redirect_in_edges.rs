//! B5 tests (0.4.0): in-edges pointing at compressed sources are redirected
//! to their summary.
//!
//! Before B5, a memory that associated with a cluster source kept a "ghost
//! edge" to a compressed unit: the edge existed in the graph but the target
//! could neither expand (F3) nor rank (7c) — and the association itself was
//! lost. After B5 the edge points at the summary, so the graph stays
//! connected and the summary becomes reachable as an upward view through
//! graph edges (the channel B1 reserved).

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::{GeneratedBy, MemoryLifecycle, MemoryUnit, WriteContext};
use hippmem_engine::{
    ConsolidationScope, Engine, EngineConfig, InspectQuery, InspectReport, RetrieveContext,
    RetrieveInput, WriteMemoryInput,
};
use redb::ReadableDatabase;
use redb::ReadableTable;
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

fn retrieve_ctx() -> RetrieveContext {
    RetrieveContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        recent_memory_ids: vec![],
    }
}

fn all_units(db_path: &std::path::Path) -> Vec<MemoryUnit> {
    let db = redb::Database::create(db_path).unwrap();
    let read_txn = db.begin_read().unwrap();
    let table = read_txn
        .open_table(hippmem_store::store::MEMORY_KV)
        .unwrap();
    let mut out = Vec::new();
    for entry in table.iter().unwrap().flatten() {
        if let Ok((unit, _)) = bincode::serde::decode_from_slice::<MemoryUnit, _>(
            entry.1.value(),
            bincode::config::standard(),
        ) {
            out.push(unit);
        }
    }
    out
}

fn out_targets(
    engine: &Engine,
    id: hippmem_core::ids::MemoryId,
) -> Vec<hippmem_core::ids::MemoryId> {
    match engine.inspect(InspectQuery::Memory(id)) {
        Ok(InspectReport::Memory(m)) => m.out_edges.iter().map(|e| e.to).collect(),
        _ => panic!("inspect failed for {id:?}"),
    }
}

#[test]
fn in_edges_are_redirected_to_summary_and_summary_is_reachable() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("hippmem.redb");
    let engine = Engine::open(EngineConfig {
        store_dir: db_path.clone(),
        ..Default::default()
    })
    .unwrap();

    // 13 near-duplicate template memories → one cluster → one summary.
    for i in 0..13u32 {
        engine
            .write(WriteMemoryInput {
                content: format!("项目{i}采用了 Rust 编写核心引擎，重点优化内存检索性能。"),
                content_type: Some(ContentType::ProjectKnowledge),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }
    // One OUTSIDE memory sharing an entity with the cluster ("Rust"): the
    // write path builds an EntityOverlap edge outside→template.
    let outside = engine
        .write(WriteMemoryInput {
            content: "Rust 核心引擎的检索性能非常重要。".to_string(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: Some(0.7),
            source_refs: vec![],
        })
        .unwrap()
        .memory_id;
    let outside_targets_before = out_targets(&engine, outside);
    assert!(
        outside_targets_before.len() >= 13,
        "outside memory must associate with the cluster templates, got {}",
        outside_targets_before.len()
    );

    // Consolidate → summary created, sources compressed.
    let report = engine.consolidate(ConsolidationScope::Incremental).unwrap();
    assert!(
        report.summaries_created >= 1,
        "cluster must trigger a summary"
    );

    // Contract 1: no ghost edges — nothing points at a compressed source.
    engine.close().unwrap();
    let units = all_units(&db_path);
    let compressed: Vec<hippmem_core::ids::MemoryId> = units
        .iter()
        .filter(|u| matches!(u.lifecycle, MemoryLifecycle::Compressed { .. }))
        .map(|u| u.id)
        .collect();
    assert!(!compressed.is_empty(), "sources must be compressed");
    let ghost = units
        .iter()
        .flat_map(|u| u.links.iter().map(move |l| (u.id, l.target_id)))
        .filter(|(_, t)| compressed.contains(t))
        .count();
    assert_eq!(ghost, 0, "no edge may point at a compressed source (B5)");

    // Contract 2: the outside memory's edges now point at the summary.
    let summary: hippmem_core::ids::MemoryId = units
        .iter()
        .find(|u| u.provenance.generated_by == GeneratedBy::Consolidation)
        .map(|u| u.id)
        .expect("summary must exist");
    let engine = Engine::open(EngineConfig {
        store_dir: db_path.clone(),
        ..Default::default()
    })
    .unwrap();
    let outside_targets_after = out_targets(&engine, outside);
    assert!(
        outside_targets_after.contains(&summary),
        "outside memory must now point at the summary (B5 redirect)"
    );

    // Contract 3: the summary is reachable through the graph — a query that
    // seeds the outside memory also surfaces the summary (upward view).
    let out = engine
        .retrieve(RetrieveInput {
            query: "Rust 核心引擎的检索性能".to_string(),
            context: retrieve_ctx(),
            top_k: 10,
            max_hops: Some(2),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    assert!(
        out.results.iter().any(|r| r.memory.id == summary),
        "summary must be reachable via graph edges (upward view, B5)"
    );
    engine.close().unwrap();
}
