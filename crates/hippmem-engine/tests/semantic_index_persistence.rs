//! Semantic index persistence contract tests (0.4.2, semantic-index-persistence).
//!
//! The dense vector index and the binary code index are in-memory structures;
//! since 0.4.2 the dense vectors are persisted in DENSE_VECTORS at write time
//! and both indexes are rebuilt from the store on open. These tests pin the
//! reopen behavior: exact retrieval restoration, explicit degradation for
//! stores without persisted vectors, and determinism across reopen.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Timestamp;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};
use tempfile::tempdir;

fn ctx() -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: Timestamp(1_700_000_000_000),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

fn retrieve_ctx() -> RetrieveContext {
    RetrieveContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        recent_memory_ids: vec![],
    }
}

fn write(engine: &Engine, text: &str) {
    engine
        .write(WriteMemoryInput {
            content: text.to_string(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
}

fn run(engine: &Engine, query: &str) -> Vec<(String, f32)> {
    engine
        .retrieve(RetrieveInput {
            query: query.to_string(),
            context: retrieve_ctx(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap()
        .results
        .iter()
        .map(|r| (r.memory.content.raw.clone(), r.final_score))
        .collect()
}

fn open_engine(store_dir: &std::path::Path) -> Engine {
    Engine::open(EngineConfig {
        store_dir: store_dir.to_path_buf(),
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn dense_index_survives_reopen() {
    let dir = tempdir().unwrap();
    let store_dir = dir.path().join("hippmem.redb");

    let before = {
        let engine = open_engine(&store_dir);
        write(&engine, "Xiaoming lives in Haidian District, Beijing.");
        write(
            &engine,
            "Xiaoming is a computer science student at Peking University.",
        );
        assert!(
            !engine.semantic_index_degraded(),
            "a freshly written store must have a live dense index"
        );
        run(&engine, "Where does Xiaoming live?")
    };
    assert!(!before.is_empty());

    let after = {
        let engine = open_engine(&store_dir);
        assert!(
            !engine.semantic_index_degraded(),
            "reopen must rebuild the dense index from DENSE_VECTORS"
        );
        run(&engine, "Where does Xiaoming live?")
    };

    assert_eq!(
        before, after,
        "reopen must restore the exact retrieval behavior (dense + binary + lexical channels)"
    );
}

#[test]
fn store_without_vectors_is_explicitly_degraded() {
    let dir = tempdir().unwrap();
    let store_dir = dir.path().join("hippmem.redb");
    {
        let engine = open_engine(&store_dir);
        write(&engine, "Xiaoming lives in Haidian District, Beijing.");
        write(
            &engine,
            "Xiaoming is a computer science student at Peking University.",
        );
    } // close

    // Simulate an old / embedding-failed store: drop every persisted vector.
    {
        use hippmem_store::store::{RedbStore, Store, DENSE_VECTORS};
        use redb::ReadableTable;
        let store = RedbStore::open(&store_dir).unwrap();
        let db = store.db_arc();
        let txn = db.begin_write().unwrap();
        let keys: Vec<u128> = {
            let table = txn.open_table(DENSE_VECTORS).unwrap();
            table
                .iter()
                .unwrap()
                .flatten()
                .map(|(k, _)| k.value())
                .collect()
        };
        {
            let mut table = txn.open_table(DENSE_VECTORS).unwrap();
            for k in keys {
                table.remove(k).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    let engine = open_engine(&store_dir);
    assert!(
        engine.semantic_index_degraded(),
        "non-empty store with no persisted vectors must report degradation"
    );
    let results = run(&engine, "Where does Xiaoming live?");
    assert!(
        !results.is_empty(),
        "lexical and binary channels must still work on a degraded store"
    );
}

#[test]
fn reopen_is_deterministic() {
    let dir = tempdir().unwrap();
    let store_dir = dir.path().join("hippmem.redb");
    {
        let engine = open_engine(&store_dir);
        write(&engine, "Xiaoming lives in Haidian District, Beijing.");
        write(
            &engine,
            "Xiaoming is a computer science student at Peking University.",
        );
        write(
            &engine,
            "Xiaoming's goal is to become an AI architect within three years.",
        );
    }
    let first = {
        let engine = open_engine(&store_dir);
        run(&engine, "Where does Xiaoming live?")
    };
    let second = {
        let engine = open_engine(&store_dir);
        run(&engine, "Where does Xiaoming live?")
    };
    assert_eq!(
        first, second,
        "repeated reopens must produce bit-identical results"
    );
}
