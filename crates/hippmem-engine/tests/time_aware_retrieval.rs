//! acceptance test: query time-aware retrieval (proposal
//! `query-time-aware-retrieval`, confirmed 2026-08-27).
//!
//! Writes memories with explicit `local_time`, then verifies that temporal
//! queries ("2026年3月5日", "昨天", "3月到5月") hit the target memories via
//! the Temporal channel, and that time-less queries are unchanged.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::{days_from_civil, Clock, SystemClock, Timestamp};
use hippmem_engine::{Engine, EngineConfig, RetrieveInput, WriteMemoryInput};
use tempfile::tempdir;

const MS_PER_DAY: i64 = 86_400_000;

fn ctx_at(ms: i64) -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: Timestamp(ms),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

fn write_at(engine: &Engine, content: &str, ms: i64) -> u128 {
    engine
        .write(WriteMemoryInput {
            content: content.into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx_at(ms),
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
            context: hippmem_engine::RetrieveContext {
                conversation_id: Some(1),
                session_id: Some(1),
                project_id: None,
                task_id: None,
                user_id: None,
                recent_memory_ids: vec![],
            },
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

fn open_engine(dir: &tempfile::TempDir) -> Engine {
    Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap()
}

/// An absolute-date query hits the memory written on that UTC day
/// (01:00 UTC — inside the UTC day bucket; the ±1 neighbour buckets absorb
/// any local-time offset).
#[test]
fn absolute_date_backtracks_to_target_day() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    let target = write_at(
        &engine,
        "部署了新的日志采集系统。",
        days_from_civil(2026, 3, 5) * MS_PER_DAY + 3_600_000,
    );
    let other = write_at(
        &engine,
        "完成了数据库分库迁移。",
        days_from_civil(2026, 2, 10) * MS_PER_DAY + 3_600_000,
    );
    let _ = other;

    let ids = retrieve_ids(&engine, "2026年3月5日做了什么");
    assert!(
        ids.contains(&target),
        "target-day memory must be retrieved (got ranks {:?})",
        ids
    );
    engine.close().unwrap();
}

/// A yearless date ("3月5日") resolves to the current year when not in the
/// future, and hits the matching memory.
#[test]
fn yearless_date_hits_current_year_day() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    let today = SystemClock.now().0 / MS_PER_DAY * MS_PER_DAY;
    let (y, m, d) = hippmem_core::time::civil_from_days(today / MS_PER_DAY);
    let target = write_at(
        &engine,
        "确定了明年的产品路线图。",
        days_from_civil(y, m, d) * MS_PER_DAY + 3_600_000,
    );
    let query = format!("{m}月{d}日做了什么");
    let ids = retrieve_ids(&engine, &query);
    assert!(
        ids.contains(&target),
        "current-year date memory must be retrieved (got {:?})",
        ids
    );
    engine.close().unwrap();
}

/// A relative "昨天" query hits a memory written one day before now.
#[test]
fn relative_yesterday_hits_previous_day() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    let yesterday = SystemClock.now().0 - MS_PER_DAY;
    let target = write_at(&engine, "处理了客户投诉工单。", yesterday);
    let ids = retrieve_ids(&engine, "昨天做了什么");
    assert!(
        ids.contains(&target),
        "yesterday memory must be retrieved (got {:?})",
        ids
    );
    engine.close().unwrap();
}

/// A month-range query hits memories in the range and keeps out-of-range
/// memories out of the temporal contribution (they may still surface via
/// semantic channels — assert on the in-range ones only).
#[test]
fn month_range_hits_all_in_range_days() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    let in_march = write_at(
        &engine,
        "上线了新的支付渠道。",
        days_from_civil(2026, 3, 15) * MS_PER_DAY + 3_600_000,
    );
    let in_may = write_at(
        &engine,
        "发布了季度运营报告。",
        days_from_civil(2026, 5, 20) * MS_PER_DAY + 3_600_000,
    );
    let before = write_at(
        &engine,
        "完成年度审计准备。",
        days_from_civil(2026, 1, 8) * MS_PER_DAY + 3_600_000,
    );
    let _ = before;

    let ids = retrieve_ids(&engine, "3月到5月做了哪些事");
    assert!(ids.contains(&in_march), "March memory must be retrieved");
    assert!(ids.contains(&in_may), "May memory must be retrieved");
    engine.close().unwrap();
}

/// A query without temporal expression keeps current-time behavior: the
/// retrieval still returns results through the other channels.
#[test]
fn no_temporal_expression_regression() {
    let dir = tempdir().unwrap();
    let engine = open_engine(&dir);

    write_at(
        &engine,
        "小明住在北京海淀区。",
        SystemClock.now().0 - 60_000,
    );
    let ids = retrieve_ids(&engine, "小明住在哪里");
    assert!(
        !ids.is_empty(),
        "time-less query must still retrieve through semantic channels"
    );
    engine.close().unwrap();
}
