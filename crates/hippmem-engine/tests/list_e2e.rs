//! acceptance test: list API.
//!
//! Tests paginated memory listing: ordering, cursor pagination, type filter, limit clamp, empty database.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput};
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

#[test]
fn list_newest_first_order() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write 5 memories, 1ms apart, to ensure ULID timestamps differ
    for i in 0..5 {
        engine
            .write(WriteMemoryInput {
                content: format!("memory content #{}", i),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let output = engine
        .list(hippmem_engine::ListInput {
            limit: 20,
            ..Default::default()
        })
        .unwrap();

    assert_eq!(output.items.len(), 5, "should return all 5 memories");
    assert_eq!(output.total, 5);
    assert!(
        output.next_cursor.is_none(),
        "all 5 fit on one page; there should be no next page"
    );

    // Verify NewestFirst: later writes come first, i.e. #4, #3, #2, #1, #0
    for i in 0..5 {
        let expected = format!("memory content #{}", 4 - i);
        assert!(
            output.items[i].content_preview.contains(&expected),
            "item {} should be '{}', actual: '{}'",
            i,
            expected,
            output.items[i].content_preview
        );
    }

    // Verify ListItem fields are non-empty / reasonable
    for item in &output.items {
        assert!(!item.content_preview.is_empty());
        assert!(item.importance >= 0.0 && item.importance <= 1.0);
    }

    engine.close().unwrap();
}

#[test]
fn list_cursor_pagination() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    for i in 0..5 {
        engine
            .write(WriteMemoryInput {
                content: format!("memory #{}", i),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    // Page 1: 3 items
    let page1 = engine
        .list(hippmem_engine::ListInput {
            limit: 3,
            ..Default::default()
        })
        .unwrap();

    assert_eq!(page1.items.len(), 3);
    assert_eq!(page1.total, 5);
    assert!(page1.next_cursor.is_some(), "there should be a next page");

    // Page 2: use the cursor to get the remaining 2 items
    let cursor = page1.next_cursor.unwrap();
    let page2 = engine
        .list(hippmem_engine::ListInput {
            limit: 3,
            cursor: Some(cursor),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(
        page2.items.len(),
        2,
        "page 2 should return the remaining 2 items"
    );
    assert!(page2.next_cursor.is_none(), "last page");

    // The two pages should not overlap
    let page1_ids: Vec<u128> = page1.items.iter().map(|item| item.id.0).collect();
    let page2_ids: Vec<u128> = page2.items.iter().map(|item| item.id.0).collect();
    for id in &page2_ids {
        assert!(
            !page1_ids.contains(id),
            "the two pages should not have overlapping IDs"
        );
    }

    engine.close().unwrap();
}

#[test]
fn list_content_type_filter() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write 2 UserStatement + 2 Decision
    engine
        .write(WriteMemoryInput {
            content: "User statement A".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Decision A".into(),
            content_type: Some(ContentType::Decision),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "User statement B".into(),
            content_type: Some(ContentType::UserStatement),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();
    engine
        .write(WriteMemoryInput {
            content: "Decision B".into(),
            content_type: Some(ContentType::Decision),
            context: ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // List only Decision
    let output = engine
        .list(hippmem_engine::ListInput {
            limit: 10,
            content_type: Some(ContentType::Decision),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(output.items.len(), 2, "should have only 2 Decision entries");
    for item in &output.items {
        assert_eq!(item.content_type, ContentType::Decision);
    }

    // List only UserStatement
    let output2 = engine
        .list(hippmem_engine::ListInput {
            limit: 10,
            content_type: Some(ContentType::UserStatement),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(
        output2.items.len(),
        2,
        "should have only 2 UserStatement entries"
    );

    engine.close().unwrap();
}

#[test]
fn list_empty_database() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let output = engine.list(hippmem_engine::ListInput::default()).unwrap();

    assert!(output.items.is_empty());
    assert_eq!(output.total, 0);
    assert!(output.next_cursor.is_none());

    engine.close().unwrap();
}

#[test]
fn list_limit_clamp() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // limit=200 should be clamped to 100
    let output = engine
        .list(hippmem_engine::ListInput {
            limit: 200,
            ..Default::default()
        })
        .unwrap();

    assert!(
        output.items.len() <= 100,
        "limit should be clamped to at most 100"
    );

    // limit=0 should be clamped to 1
    let output2 = engine
        .list(hippmem_engine::ListInput {
            limit: 0,
            ..Default::default()
        })
        .unwrap();

    assert!(
        output2.items.len() <= 1,
        "limit=0 should be clamped to at least 1"
    );

    engine.close().unwrap();
}
