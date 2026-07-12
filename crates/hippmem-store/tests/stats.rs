//! acceptance test: StoreStats diagnostics

use hippmem_store::kv::{InvertedIndex, KvStore};
use hippmem_store::memory_log::MemoryLog;
use hippmem_store::store::{RedbStore, Store, StoreStats};
use tempfile::tempdir;

/// An empty store returns all-zero stats.
#[test]
fn empty_store_returns_zero_counts() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");

    let stats = store.stats().expect("get stats");
    assert_eq!(stats.memory_log_count, 0);
    assert_eq!(stats.memory_kv_count, 0);
    assert_eq!(stats.entity_index_size, 0);
    assert_eq!(stats.topic_index_size, 0);
    assert_eq!(stats.goal_index_size, 0);
    assert_eq!(stats.event_index_size, 0);
    assert_eq!(stats.temporal_index_size, 0);
}

/// Stats update after writing memories.
#[test]
fn stats_reflect_writes() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");

    // Write memory_log and memory_kv
    let log = MemoryLog::new(store.db_arc());
    let kv = KvStore::new(store.db_arc());

    log.append(1u128, b"record-1").expect("append");
    log.append(2u128, b"record-2").expect("append");
    log.append(3u128, b"record-3").expect("append");

    kv.put(10u128, b"data").expect("put");
    kv.put(20u128, b"data").expect("put");

    // Write inverted indexes
    let idx = InvertedIndex::new(store.db_arc());
    idx.add_entity(100, 1).expect("write entity");
    idx.add_entity(100, 2).expect("append entity");
    idx.add_entity(200, 3).expect("write another entity");
    idx.add_topic(42, 1).expect("write topic");
    idx.add_goal(7, 1).expect("write goal");
    idx.add_event(5, 99).expect("write event");
    idx.add_temporal(202401, 50).expect("write temporal");

    let stats = store.stats().expect("get stats");
    assert_eq!(stats.memory_log_count, 3);
    assert_eq!(stats.memory_kv_count, 2);
    // entity_index: key 100 has 2 entries, key 200 has 1 entry -> 2 keys total
    assert_eq!(stats.entity_index_size, 2);
    assert_eq!(stats.topic_index_size, 1);
    assert_eq!(stats.goal_index_size, 1);
    assert_eq!(stats.event_index_size, 1);
    assert_eq!(stats.temporal_index_size, 1);
}

/// StoreStats supports Clone + Debug.
#[test]
fn store_stats_debug_and_clone() {
    let stats = StoreStats {
        memory_log_count: 5,
        memory_kv_count: 3,
        entity_index_size: 2,
        topic_index_size: 1,
        goal_index_size: 0,
        event_index_size: 0,
        temporal_index_size: 1,
        causal_index_size: 0,
    };

    let cloned = stats.clone();
    assert_eq!(stats.memory_log_count, cloned.memory_log_count);

    let debug_str = format!("{stats:?}");
    assert!(debug_str.contains("memory_log_count"));
}
