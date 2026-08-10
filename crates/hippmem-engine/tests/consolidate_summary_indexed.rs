//! 回归测试（P3）：consolidate 创建的摘要必须进入全部检索索引，可被检索到。
//!
//! 背景：摘要曾只写 memory_kv + link_overlay，不进 fulltext/向量/倒排索引，
//! 也不写 memory_log——检索永远看不到摘要，reindex 还会丢摘要。

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::{GeneratedBy, WriteContext};
use hippmem_engine::{
    ConsolidationScope, Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput,
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

/// 从 memory_kv 解码全部单元，返回 consolidate 生成的摘要（按生成来源识别）。
fn find_summary_units(db_path: &std::path::Path) -> Vec<hippmem_core::model::unit::MemoryUnit> {
    let db = redb::Database::create(db_path).unwrap();
    let read_txn = db.begin_read().unwrap();
    let table = read_txn
        .open_table(hippmem_store::store::MEMORY_KV)
        .unwrap();
    let mut out = Vec::new();
    for entry in table.iter().unwrap().flatten() {
        if let Ok((unit, _)) = bincode::serde::decode_from_slice::<
            hippmem_core::model::unit::MemoryUnit,
            _,
        >(entry.1.value(), bincode::config::standard())
        {
            if unit.provenance.generated_by == GeneratedBy::Consolidation {
                out.push(unit);
            }
        }
    }
    out
}

#[test]
fn consolidate_summary_becomes_retrievable() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("hippmem.redb");
    let engine = Engine::open(EngineConfig {
        store_dir: db_path.clone(),
        ..Default::default()
    })
    .unwrap();

    // 13 条记忆（≥ should_summarize 触发阈值 12）
    let texts: Vec<String> = (0..13)
        .map(|i| format!("项目{i}采用了 Rust 编写核心引擎，重点优化内存检索性能。"))
        .collect();
    for t in &texts {
        engine
            .write(WriteMemoryInput {
                content: t.clone(),
                content_type: Some(ContentType::ProjectKnowledge),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }

    let report = engine.consolidate(ConsolidationScope::Incremental).unwrap();
    assert!(
        report.summaries_created >= 1,
        "13 条记忆应触发摘要创建, 实际 summaries_created={}",
        report.summaries_created
    );
    engine.close().unwrap();

    // 找到摘要单元及其内容
    let summaries = find_summary_units(&db_path);
    assert_eq!(summaries.len(), 1, "应恰好创建一条摘要");
    let summary_text = summaries[0].content.raw.clone();
    let summary_id = summaries[0].id;

    // 重新打开引擎，用摘要原文作为查询——必须能检索到摘要本身
    let engine = Engine::open(EngineConfig {
        store_dir: db_path,
        ..Default::default()
    })
    .unwrap();
    let out = engine
        .retrieve(RetrieveInput {
            query: summary_text,
            context: RetrieveContext::default(),
            top_k: 5,
            max_hops: None,
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap();
    engine.close().unwrap();

    assert!(
        out.results.iter().any(|r| r.memory.id == summary_id),
        "consolidate 创建的摘要必须可被检索到（需进入 fulltext/向量/倒排索引），\
         实际 top-{}: {:?}",
        out.results.len(),
        out.results
            .iter()
            .map(|r| &r.memory.content.raw)
            .collect::<Vec<_>>()
    );
}
