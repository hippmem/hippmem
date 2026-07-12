//! acceptance test: attribute inverted index writes and queries

use hippmem_store::kv::InvertedIndex;
use hippmem_store::store::{RedbStore, Store};
use tempfile::tempdir;

/// After writing to entity_index, memory_ids can be queried back by key.
#[test]
fn entity_index_write_and_query() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let idx = InvertedIndex::new(store.db_arc());

    idx.add_entity(100u64, 1u128).expect("write entity");
    idx.add_entity(100u64, 2u128).expect("append entity");
    idx.add_entity(200u64, 3u128).expect("write another key");

    let ids = idx.get_entity(&100u64).expect("query");
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&1u128));
    assert!(ids.contains(&2u128));

    let ids = idx.get_entity(&200u64).expect("query");
    assert_eq!(ids, vec![3u128]);

    // A non-existent key returns empty
    let ids = idx.get_entity(&999u64).expect("query");
    assert!(ids.is_empty());
}

/// topic_index write and query.
#[test]
fn topic_index_write_and_query() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let idx = InvertedIndex::new(store.db_arc());

    idx.add_topic(42u64, 10u128).expect("write");
    idx.add_topic(42u64, 20u128).expect("append");

    let ids = idx.get_topic(&42u64).expect("query");
    assert_eq!(ids, vec![10u128, 20u128]);
}

/// goal_index write and query.
#[test]
fn goal_index_write_and_query() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let idx = InvertedIndex::new(store.db_arc());

    idx.add_goal(1u64, 7u128).expect("write");
    let ids = idx.get_goal(&1u64).expect("query");
    assert_eq!(ids, vec![7u128]);
}

/// event_index write and query.
#[test]
fn event_index_write_and_query() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let idx = InvertedIndex::new(store.db_arc());

    idx.add_event(5u64, 99u128).expect("write");
    let ids = idx.get_event(&5u64).expect("query");
    assert_eq!(ids, vec![99u128]);
}

/// temporal_index write and query (u32 key).
#[test]
fn temporal_index_write_and_query() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let idx = InvertedIndex::new(store.db_arc());

    idx.add_temporal(202401u32, 50u128).expect("write");
    idx.add_temporal(202401u32, 51u128).expect("append");

    let ids = idx.get_temporal(&202401u32).expect("query");
    assert_eq!(ids, vec![50u128, 51u128]);
}

/// Dedup: repeated writes of the same (key, memory_id) do not append duplicates.
#[test]
fn inverted_index_dedup() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let idx = InvertedIndex::new(store.db_arc());

    idx.add_entity(10u64, 1u128).expect("write");
    idx.add_entity(10u64, 1u128)
        .expect("write the same id again");
    idx.add_entity(10u64, 1u128)
        .expect("write the same id a third time");

    let ids = idx.get_entity(&10u64).expect("query");
    assert_eq!(ids, vec![1u128], "duplicate writes should be deduped");
}
