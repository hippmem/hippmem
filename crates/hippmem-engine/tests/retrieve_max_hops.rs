//! 回归测试（P2）：max_hops 必须真实传入扩散步骤，hops_used 必须报告实际跳数。
//!
//! 背景：input.max_hops 曾被丢弃（扩散固定用 max_hops_default=2），
//! hops_used 硬编码 0——调用方无法得知图遍历深度，也无法控制它。

use hippmem_core::config::AlgoParams;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};
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

/// 关闭语义通道与 recent 通道（binary/dense 对所有记忆打分 > 0 会把链上节点全变成种子；
/// recent 通道会把上次检索结果变成种子、阻塞二次检索的图扩散）——
/// 只保留 entity/bm25 通道 → 只有查询实体命中者为种子。
/// 另降低传播能量阈值：默认 decay_factor=0.55 且能量按 hop 指数衰减（×0.55^hop），
/// 两跳能量 ≈ 0.47×0.62²×0.45² ≈ 0.006 < 0.05，两跳传播在默认参数下不可达；
/// 本测试聚焦跳数控制语义而非能量衰减真实性。
fn chain_engine(store_dir: &std::path::Path) -> Engine {
    Engine::open(EngineConfig {
        store_dir: store_dir.into(),
        algo: AlgoParams {
            rrf_w_semantic_binary: 0.0,
            rrf_w_semantic_dense: 0.0,
            rrf_w_recent: 0.0,
            min_propagation_energy: 0.005,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap()
}

fn write(engine: &Engine, text: &str) {
    engine
        .write(WriteMemoryInput {
            content: text.into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
}

fn retrieve_with_hops(
    engine: &Engine,
    query: &str,
    max_hops: Option<usize>,
) -> hippmem_engine::RetrieveOutput {
    engine
        .retrieve(RetrieveInput {
            query: query.into(),
            context: RetrieveContext::default(),
            top_k: 10,
            max_hops,
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap()
}

fn max_hop_in_trace(out: &hippmem_engine::RetrieveOutput) -> u8 {
    out.trace.steps.iter().map(|s| s.hop).max().unwrap_or(0)
}

/// 链 C→B→A：query 只命中 C（实体 赵磊）；A 与查询零词面重叠，只能经图到达。
/// 写路径只建 新→旧 单向边（边存在新单元上），故按 A→B→C 顺序写入：
/// C 的出边 → B（王芳 + 计划(goal) + 开会/讨论(event)，多维强边），B 的出边 → A（张伟 实体边），
/// C↔A 无共享维度 → 无直接边（不短路两跳）。
/// max_hops=1 时 A 不可达（hops_used=1）；max_hops=3 时 A 可达（hops_used=2）。
#[test]
fn max_hops_controls_traversal_depth() {
    let dir = tempdir().unwrap();
    let engine = chain_engine(&dir.path().join("hippmem.redb"));

    write(&engine, "张伟喜欢打篮球，经常去球场训练。"); // A（最老：无候选，无出边）
    write(
        &engine,
        "王芳和张伟计划优化数据库性能，上周开会讨论了方案。",
    ); // B：出边 → A
    write(
        &engine,
        "赵磊和王芳计划优化数据库性能，上周开会讨论了方案。",
    ); // C：种子，出边 → B

    // 单跳：只有种子 + 一跳邻居
    let out1 = retrieve_with_hops(&engine, "赵磊在哪里上学？", Some(1));
    assert_eq!(
        out1.trace.hops_used, 1,
        "max_hops=1 时 hops_used 必须为 1（实际 0 = 报告硬编码 bug 未修）"
    );
    assert_eq!(max_hop_in_trace(&out1), 1, "max_hops=1 时不得出现 2 跳节点");

    // 多跳：两跳邻居可达
    let out3 = retrieve_with_hops(&engine, "赵磊在哪里上学？", Some(3));
    assert!(
        out3.trace.hops_used >= 2,
        "max_hops=3 时两跳链必须被执行, 实际 hops_used={}",
        out3.trace.hops_used
    );
    assert!(
        max_hop_in_trace(&out3) >= 2,
        "max_hops=3 时 trace 中必须包含 hop=2 的扩散步"
    );

    // 两跳邻居（张伟喜欢打篮球）应出现在结果中
    assert!(
        out3.results
            .iter()
            .any(|r| r.memory.content.raw.contains("张伟喜欢打篮球")),
        "两跳邻居必须出现在 top-10 结果中"
    );

    engine.close().unwrap();
}

/// 无扩散发生时 hops_used 必须为 0（而非硬编码占位）。
#[test]
fn hops_used_zero_when_no_propagation() {
    let dir = tempdir().unwrap();
    let engine = chain_engine(&dir.path().join("hippmem.redb"));
    write(&engine, "一个人独自在房间里看书。");

    let out = retrieve_with_hops(&engine, "完全无关的查询词，没有匹配。", None);
    // 无匹配时走 fallback 种子；单节点无出边 → 无扩散
    assert_eq!(out.trace.hops_used, 0);
    engine.close().unwrap();
}
