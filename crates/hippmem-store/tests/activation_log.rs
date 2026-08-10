//! acceptance test: activation_log u128 id roundtrip.
//!
//! 回归测试（P1）：MemoryId 是 128 位 ULID，activation_log 记录必须完整保留
//! used_memory_ids（不得截断为 u64 后再零扩展，否则下游 RecentActivation /
//! Hebbian 全部指向幽灵 id）。

use hippmem_store::activation_log::{ActivationLogger, ActivationRecord};
use hippmem_store::store::{RedbStore, Store};
use tempfile::tempdir;

/// u128 高位置位（模拟 ULID：高 48 位是毫秒时间戳，必然非零）时，
/// 记录后读回必须逐位一致。
#[test]
fn activation_log_roundtrips_high_bit_u128_ids() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let logger = ActivationLogger::new(store.db_arc());

    // ULID 形态的 id：高 48 位时间戳 + 随机低位
    let ulid_like: u128 = (1_786_289_882_389u64 as u128) << 80 | 0xDEAD_BEEF_1234_5678;
    let ids: Vec<u128> = vec![ulid_like, 1, u128::MAX - 7];

    logger
        .record(&ActivationRecord {
            retrieval_id: 42,
            used_memory_ids: ids.clone(),
            signal: "retrieve".into(),
            recorded_at_ms: 1_000,
        })
        .expect("record should succeed");

    let recs = logger.read_all().expect("read_all should succeed");
    assert_eq!(recs.len(), 1);
    assert_eq!(
        recs[0].used_memory_ids, ids,
        "used_memory_ids must survive the storage roundtrip bit-exactly \
         (u64 truncation corrupts ULID memory ids)"
    );
}

/// 多条记录（含同 retrieval_id 不同时间戳）都能独立读回。
#[test]
fn activation_log_multiple_records_roundtrip() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let logger = ActivationLogger::new(store.db_arc());

    logger
        .record(&ActivationRecord {
            retrieval_id: 100,
            used_memory_ids: vec![11, 22],
            signal: "retrieve".into(),
            recorded_at_ms: 1_000,
        })
        .expect("record 1");
    logger
        .record(&ActivationRecord {
            retrieval_id: 100, // 同 retrieval_id，不同时间戳 → 不同 key
            used_memory_ids: vec![33],
            signal: "UserConfirmedCorrect".into(),
            recorded_at_ms: 1_005,
        })
        .expect("record 2");

    let recs = logger.read_all().expect("read_all should succeed");
    assert_eq!(
        recs.len(),
        2,
        "same retrieval_id with different timestamps must not overwrite"
    );
    let mut signals: Vec<String> = recs.iter().map(|r| r.signal.clone()).collect();
    signals.sort();
    assert_eq!(
        signals,
        vec![
            String::from("UserConfirmedCorrect"),
            String::from("retrieve")
        ]
    );
}
