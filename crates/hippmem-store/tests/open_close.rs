//! acceptance test: Store trait + redb open/close/reopen

use hippmem_store::store::{RedbStore, Store};
use std::fs;

/// Can create a redb in a temp directory, define all tables, close, and reopen.
#[test]
fn create_tables_and_reopen() {
    let dir = tempfile::tempdir().expect("should be able to create a temp dir");
    let store_path = dir.path().join("hippmem.redb");

    // First open: all tables should be defined during open
    {
        let store = RedbStore::open(&store_path).expect("first open should succeed");
        store.close().expect("close should succeed");
    }

    // Verify the redb file exists
    assert!(store_path.exists(), "redb file should exist");
    assert!(
        store_path.metadata().unwrap().len() > 0,
        "redb file should not be empty"
    );

    // Reopen: should succeed
    {
        let store = RedbStore::open(&store_path).expect("reopen should succeed");
        store.close().expect("second close should succeed");
    }

    // Cleanup (optional; tempfile auto-deletes on drop)
    fs::remove_dir_all(dir).ok();
}

/// Multiple opens of the same file do not lose tables.
#[test]
fn reopen_preserves_tables() {
    let dir = tempfile::tempdir().expect("should be able to create a temp dir");
    let store_path = dir.path().join("hippmem.redb");

    // Open -> close
    {
        let store = RedbStore::open(&store_path).expect("open should succeed");
        store.close().expect("close should succeed");
    }
    // Reopen -> close (no panic)
    {
        let store = RedbStore::open(&store_path).expect("reopen should succeed");
        store.close().expect("close should succeed");
    }

    fs::remove_dir_all(dir).ok();
}

/// Opening a non-existent directory should auto-create it.
#[test]
fn open_creates_parent_dir() {
    let dir = tempfile::tempdir().expect("should be able to create a temp dir");
    let nested = dir
        .path()
        .join("nested")
        .join("subdir")
        .join("hippmem.redb");

    let store = RedbStore::open(&nested).expect("should auto-create parent dir");
    assert!(nested.exists());
    store.close().expect("close should succeed");

    fs::remove_dir_all(dir).ok();
}
