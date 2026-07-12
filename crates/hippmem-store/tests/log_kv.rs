//! acceptance test: memory_log append-only + memory_kv read/write roundtrip

use hippmem_store::kv::KvStore;
use hippmem_store::memory_log::MemoryLog;
use hippmem_store::store::{RedbStore, Store};
use tempfile::tempdir;

// ── memory_log: append-only ──

/// memory_log is append-only: attempting to overwrite an existing id returns an error.
#[test]
fn memory_log_append_only_rejects_overwrite() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let log = MemoryLog::new(store.db_arc());

    log.append(1u128, b"first record")
        .expect("first append should succeed");

    let result = log.append(1u128, b"second attempt");
    assert!(
        result.is_err(),
        "appending the same MemoryId/u128 should return an error (append-only)"
    );
}

/// memory_log successfully appends different ids.
#[test]
fn memory_log_append_different_ids() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let log = MemoryLog::new(store.db_arc());

    log.append(1u128, b"record-1").expect("append id=1");
    log.append(2u128, b"record-2").expect("append id=2");
    log.append(3u128, b"record-3").expect("append id=3");
}

/// memory_log can read back appended records.
#[test]
fn memory_log_read_back() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let log = MemoryLog::new(store.db_arc());

    log.append(42u128, b"hello world").expect("append");

    let read = log.get(&42u128).expect("read").expect("should exist");
    assert_eq!(read, b"hello world");
}

/// memory_log returns None for a non-existent record.
#[test]
fn memory_log_get_missing_returns_none() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let log = MemoryLog::new(store.db_arc());

    let result = log.get(&999u128).expect("read");
    assert!(result.is_none(), "a non-existent record should return None");
}

// ── memory_kv: read/write roundtrip ──

/// memory_kv put/get roundtrip.
#[test]
fn memory_kv_put_get_roundtrip() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let kv = KvStore::new(store.db_arc());

    kv.put(10u128, b"value-10").expect("put should succeed");

    let retrieved = kv.get(&10u128).expect("get should succeed");
    assert_eq!(retrieved, Some(b"value-10".to_vec()));
}

/// memory_kv get on a non-existent key returns None.
#[test]
fn memory_kv_get_missing_returns_none() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let kv = KvStore::new(store.db_arc());

    let result = kv.get(&999u128).expect("get should succeed");
    assert!(result.is_none(), "a non-existent key should return None");
}

/// memory_kv put overwriting an existing key succeeds (unlike memory_log).
#[test]
fn memory_kv_put_overwrites_existing() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let kv = KvStore::new(store.db_arc());

    kv.put(10u128, b"original").expect("first put");
    kv.put(10u128, b"updated")
        .expect("overwrite put should succeed");

    let retrieved = kv.get(&10u128).expect("get").unwrap();
    assert_eq!(retrieved, b"updated");
}

/// memory_kv bulk put/get.
#[test]
fn memory_kv_bulk_roundtrip() {
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("test.redb");
    let store = RedbStore::open(&path).expect("open store");
    let kv = KvStore::new(store.db_arc());

    for i in 0u128..100 {
        let val = format!("record-{i}").into_bytes();
        kv.put(i, &val).expect("put");
    }

    for i in 0u128..100 {
        let expected = format!("record-{i}").into_bytes();
        let got = kv.get(&i).expect("get").unwrap();
        assert_eq!(got, expected, "roundtrip mismatch at {i}");
    }
}
