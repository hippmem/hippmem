//! 0.3.0 回归测试：usage_score 降权机制 + UserRejected 极性过滤。
//!
//! 覆盖：确认/引用/任务成功/拒绝更新 usage_score（clamp）；拒绝不得强化
//! （RecentActivation 不计数 + usage 降权）；拒绝不产生 Hebbian 共现对。

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::LinkType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{
    Engine, EngineConfig, FeedbackInput, InspectQuery, InspectReport, RetrieveContext,
    RetrieveInput, UsageSignal, WriteMemoryInput,
};
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

fn retrieve(engine: &Engine, query: &str) -> hippmem_engine::RetrieveOutput {
    engine
        .retrieve(RetrieveInput {
            query: query.into(),
            context: RetrieveContext::default(),
            top_k: 5,
            max_hops: None,
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap()
}

fn usage_of(engine: &Engine, id: hippmem_core::ids::MemoryId) -> f32 {
    match engine.inspect(InspectQuery::Memory(id)).unwrap() {
        InspectReport::Memory(m) => m.unit.activation.usage_score.value(),
        _ => panic!("expected Memory inspect report"),
    }
}

fn assert_usage_close(engine: &Engine, id: hippmem_core::ids::MemoryId, expected: f32) {
    let actual = usage_of(engine, id);
    assert!(
        (actual - expected).abs() < 1e-5,
        "usage_score 应为 {expected}, 实际 {actual}"
    );
}

/// 反馈信号序列正确更新 usage_score（±Δ，clamp [0,1]）。
#[test]
fn usage_score_updates_on_feedback_signals() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();
    let mid = write(&engine, "小明和李华是高中同学。");
    let out = retrieve(&engine, "小明和李华之间有什么关系？");
    assert_eq!(
        usage_of(&engine, mid),
        0.5,
        "初始 usage_score 应为 0.5（中性）"
    );

    let feedback = |signal: UsageSignal| {
        engine
            .feedback(FeedbackInput {
                retrieval_id: out.retrieval_id,
                used_memory_ids: vec![mid],
                signal,
            })
            .unwrap();
    };

    feedback(UsageSignal::Referenced); // +0.05
    assert_usage_close(&engine, mid, 0.55);
    feedback(UsageSignal::UserConfirmedCorrect); // +0.10
    assert_usage_close(&engine, mid, 0.65);
    feedback(UsageSignal::TaskSucceeded); // +0.08
    assert_usage_close(&engine, mid, 0.73);
    feedback(UsageSignal::UserRejected); // -0.10
    assert_usage_close(&engine, mid, 0.63);

    // clamp 上限
    for _ in 0..10 {
        feedback(UsageSignal::UserConfirmedCorrect);
    }
    assert_eq!(usage_of(&engine, mid), 1.0);
    // clamp 下限
    for _ in 0..10 {
        feedback(UsageSignal::UserRejected);
    }
    assert_eq!(usage_of(&engine, mid), 0.0);

    engine.close().unwrap();
}

/// 拒绝必须降权而非强化：被拒记忆分数下降（B4 反向 Hebbian：拒绝削弱其关联边，
/// consolidate 后经图传播生效），其它记忆不得上升。
///
/// 注：本测试关闭 recent 通道（rrf_w_recent=0）以隔离边削弱效应——recent 通道的
/// 并列排名依赖 HashMap 布局（随累积记录变化），会引入非确定性；拒绝记录的极性过滤
/// 由 signals 单测 + user_rejected_does_not_create_coactivation_edges 覆盖。
/// 0.4.0 起 usage_score 不再参与检索能量（见 proposals/usage-score-semantics-redesign.md），
/// 定向 reject 的检索效果 = 反向 Hebbian（需要一次 consolidate 生效）。
#[test]
fn user_rejected_demotes_instead_of_boosting() {
    use hippmem_core::config::AlgoParams;
    use hippmem_engine::ConsolidationScope;
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        algo: AlgoParams {
            rrf_w_recent: 0.0,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    write(&engine, "小明和李华是高中同学。");
    write(&engine, "李华和王芳在同一家公司工作。");
    write(&engine, "王芳和张伟是一对夫妻。");
    let rejected = write(&engine, "张伟是北京大学的教授。");

    let out1 = retrieve(&engine, "张伟在哪里工作？");
    assert!(
        out1.results.iter().any(|r| r.memory.id.0 == rejected.0),
        "被拒记忆应在结果中"
    );

    // B4: the retrieval-level effect is the edge weakening — capture the
    // rejected memory's out-edge strengths before the reject.
    let edges_before: Vec<f32> = match engine.inspect(InspectQuery::Memory(rejected)) {
        Ok(InspectReport::Memory(m)) => m.out_edges.iter().map(|e| e.strength).collect(),
        _ => panic!("inspect failed"),
    };
    assert!(
        !edges_before.is_empty(),
        "rejected memory should have edges (shared entities)"
    );

    std::thread::sleep(std::time::Duration::from_millis(2));
    engine
        .feedback(FeedbackInput {
            retrieval_id: out1.retrieval_id,
            used_memory_ids: vec![rejected],
            signal: UsageSignal::UserRejected,
        })
        .unwrap();
    // B4: the rejection weakens the rejected memory's edges; run one
    // consolidation cycle so the reverse-Hebbian step takes effect.
    engine
        .consolidate(ConsolidationScope::Incremental)
        .expect("consolidate should succeed");

    let edges_after: Vec<f32> = match engine.inspect(InspectQuery::Memory(rejected)) {
        Ok(InspectReport::Memory(m)) => m.out_edges.iter().map(|e| e.strength).collect(),
        _ => panic!("inspect failed"),
    };
    assert_eq!(edges_before.len(), edges_after.len());
    for (b, a) in edges_before.iter().zip(edges_after.iter()) {
        assert!(
            a < b,
            "reject must weaken the rejected memory's out-edges ({b} → {a})"
        );
    }

    let out2 = retrieve(&engine, "张伟在哪里工作？");
    // 其它记忆不得因拒绝而上升（经被拒记忆传播的邻居可能随之下降——合法效应）
    for r in &out2.results {
        if r.memory.id.0 == rejected.0 {
            continue;
        }
        let before = out1
            .results
            .iter()
            .find(|b| b.memory.id == r.memory.id)
            .map(|b| b.final_score)
            .expect("其它记忆应在两轮结果中");
        assert!(
            r.final_score <= before + 1e-6,
            "拒绝不得提升其它记忆的分数，{:?} {} → {}",
            r.memory.id,
            before,
            r.final_score
        );
    }

    engine.close().unwrap();
}

/// 拒绝记录不得形成 Hebbian 共现对（count 阈值 3 时，仅拒绝贡献的配对不得建边/强化）。
#[test]
fn user_rejected_does_not_create_coactivation_edges() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();
    let a = write(&engine, "小明和李华是高中同学。");
    let b = write(&engine, "李华和王芳在同一家公司工作。");

    let out = retrieve(&engine, "小明和李华之间有什么关系？");
    // 两次拒绝同一对 → 若不过滤信号，共现 count = 1(检索) + 2(拒绝) = 3 ≥ 阈值，会建边/强化
    for _ in 0..2 {
        std::thread::sleep(std::time::Duration::from_millis(2));
        engine
            .feedback(FeedbackInput {
                retrieval_id: out.retrieval_id,
                used_memory_ids: vec![a, b],
                signal: UsageSignal::UserRejected,
            })
            .unwrap();
    }

    engine
        .consolidate(hippmem_engine::ConsolidationScope::Incremental)
        .unwrap();

    // 两记忆之间不得出现被强化的边（activation_count > 0 或 Causal CoActivation 边）
    let report = match engine.inspect(InspectQuery::Memory(a)).unwrap() {
        InspectReport::Memory(m) => m,
        _ => panic!("expected Memory inspect report"),
    };
    for edge in &report.out_edges {
        if edge.to == b {
            assert_eq!(
                edge.activation_count, 0,
                "拒绝信号不得强化 a→b 边（activation_count={}）",
                edge.activation_count
            );
            assert_ne!(
                edge.link_type,
                LinkType::Causal,
                "拒绝信号不得创建 CoActivation 边（Causal）"
            );
        }
    }

    engine.close().unwrap();
}
