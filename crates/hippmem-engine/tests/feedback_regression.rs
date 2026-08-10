//! 回归测试（P1）：feedback 必须真实作用于检索。
//!
//! 背景：activation_log 曾把 MemoryId(u128 ULID) 截断成 u64 落库、读回时零扩展，
//! 导致 RecentActivation 通道与 Hebbian 永远指向"幽灵 id"——feedback 完全无效。
//! 这两个测试钉死该接缝：id 往返逐位不变 + feedback 确认后检索结果必须变化。

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{
    Engine, EngineConfig, FeedbackInput, RetrieveContext, RetrieveInput, UsageSignal,
    WriteMemoryInput,
};
use hippmem_store::activation_log::ActivationLogger;
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

fn retrieve(engine: &Engine, query: &str, top_k: usize) -> hippmem_engine::RetrieveOutput {
    engine
        .retrieve(RetrieveInput {
            query: query.into(),
            context: RetrieveContext::default(),
            top_k,
            max_hops: None,
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap()
}

fn write(engine: &Engine, text: &str) -> hippmem_core::ids::MemoryId {
    engine
        .write(WriteMemoryInput {
            content: text.into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap()
        .memory_id
}

/// 真实 ULID id 经 retrieve → activation_log → feedback 往返后必须逐位不变。
#[test]
fn activation_log_preserves_generated_memory_ids() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let mid = write(&engine, "小明和李华是高中同学。");
    let out = retrieve(&engine, "小明和李华之间有什么关系？", 5);
    std::thread::sleep(std::time::Duration::from_millis(2)); // 避免 retrieval_id 毫秒级碰撞
    engine
        .feedback(FeedbackInput {
            retrieval_id: out.retrieval_id,
            used_memory_ids: vec![mid],
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();
    engine.close().unwrap();

    // 引擎关闭后直接读库（Engine.store 私有）
    let db = redb::Database::create(dir.path().join("hippmem.redb")).unwrap();
    let records = ActivationLogger::new(std::sync::Arc::new(db))
        .read_all()
        .unwrap();
    assert!(
        !records.is_empty(),
        "retrieve + feedback 应产生 activation_log 记录"
    );

    for rec in &records {
        for used in &rec.used_memory_ids {
            assert_eq!(
                *used, mid.0,
                "activation_log 中的 memory id 必须等于写入的 ULID（禁止 u64 截断），\
                 record: retrieval_id={} signal={}",
                rec.retrieval_id, rec.signal
            );
        }
    }
}

/// 行为契约：feedback 确认后，后续检索结果必须发生变化（被确认记忆获得强化）。
#[test]
fn feedback_confirm_changes_subsequent_retrieval() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    write(&engine, "小明和李华是高中同学。");
    write(&engine, "李华和王芳在同一家公司工作。");
    write(&engine, "王芳和张伟是一对夫妻。");
    let confirmed = write(&engine, "张伟是北京大学的教授。");

    let out1 = retrieve(&engine, "张伟在哪里工作？", 5);
    assert!(out1.results.len() >= 2, "测试前提：至少两条结果");

    // 确认一个非 top-1 的候选
    let target = out1
        .results
        .iter()
        .find(|r| r.memory.id.0 == confirmed.0)
        .expect("被确认记忆应在结果中");
    assert!(
        target.final_score < out1.results[0].final_score,
        "测试前提：被确认记忆不是 top-1"
    );

    let before: Vec<(u128, f32)> = out1
        .results
        .iter()
        .map(|r| (r.memory.id.0, r.final_score))
        .collect();

    std::thread::sleep(std::time::Duration::from_millis(2));
    engine
        .feedback(FeedbackInput {
            retrieval_id: out1.retrieval_id,
            used_memory_ids: vec![confirmed],
            signal: UsageSignal::UserConfirmedCorrect,
        })
        .unwrap();

    let out2 = retrieve(&engine, "张伟在哪里工作？", 5);
    let after: Vec<(u128, f32)> = out2
        .results
        .iter()
        .map(|r| (r.memory.id.0, r.final_score))
        .collect();

    assert_ne!(
        before, after,
        "feedback 确认后检索结果必须变化：被确认记忆经 RecentActivation 通道获得强化"
    );

    let after_target = out2
        .results
        .iter()
        .find(|r| r.memory.id.0 == confirmed.0)
        .expect("被确认记忆仍应在结果中");
    assert!(
        after_target.final_score > target.final_score,
        "被确认记忆的分数必须提升（{} → {}）",
        target.final_score,
        after_target.final_score
    );

    engine.close().unwrap();
}
