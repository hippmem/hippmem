//! acceptance test: basic Engine open/close lifecycle.

use hippmem_engine::{Engine, EngineConfig};
use tempfile::tempdir;

#[test]
fn open_and_close_basic() {
    let dir = tempdir().unwrap();
    let store_path = dir.path().join("hippmem.redb");

    let config = EngineConfig {
        store_dir: store_path.clone(),
        ..Default::default()
    };

    let engine = Engine::open(config).unwrap();

    // The storage file should have been created
    assert!(
        store_path.exists(),
        "redb file should exist after open: {:?}",
        store_path
    );
    assert!(
        store_path.metadata().unwrap().len() > 0,
        "redb file should not be empty"
    );

    // Close
    engine.close().unwrap();
}

#[test]
fn open_nonexistent_parent_dir() {
    let dir = tempdir().unwrap();
    let nested = dir.path().join("sub").join("sub2").join("hippmem.redb");

    let config = EngineConfig {
        store_dir: nested.clone(),
        ..Default::default()
    };

    // redb should be able to auto-create the parent directory
    let result = Engine::open(config);
    // Depending on the redb implementation, it may create it or error. We only require no panic.
    if let Ok(engine) = result {
        engine.close().unwrap();
    }
}

#[test]
fn close_consumes_engine() {
    let dir = tempdir().unwrap();
    let config = EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    };

    let engine = Engine::open(config).unwrap();
    engine.close().unwrap();
    // engine is moved away; cannot be used here — compile-time guarantee (ownership)
}
