//! E7 tests (0.4.0): retrieval updates ActivationState of the returned
//! memories — retrieval_count / last_retrieved_at / co_activations (03 §6
//! "每次检索后由检索侧累加"). usage_score is owned by feedback and untouched.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;
use hippmem_core::time::Timestamp;
use hippmem_engine::{
    Engine, EngineConfig, InspectQuery, InspectReport, RetrieveContext, RetrieveInput,
    WriteMemoryInput,
};
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

fn activation_of(
    engine: &Engine,
    id: hippmem_core::ids::MemoryId,
) -> (u32, Option<Timestamp>, usize) {
    match engine.inspect(InspectQuery::Memory(id)) {
        Ok(InspectReport::Memory(m)) => (
            m.unit.activation.retrieval_count,
            m.unit.activation.last_retrieved_at,
            m.unit.activation.co_activations.len(),
        ),
        _ => panic!("inspect failed for {id:?}"),
    }
}

#[test]
fn retrieval_updates_activation_state() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    let contents = [
        "Xiaoming and Lihua are high school classmates in Beijing",
        "Xiaoming is a computer science student at Peking University",
        "Lihua works as a Java engineer at Alibaba Cloud",
    ];
    for c in contents {
        engine
            .write(WriteMemoryInput {
                content: c.to_string(),
                content_type: Some(ContentType::UserStatement),
                context: ctx(),
                importance_hint: Some(0.5),
                source_refs: vec![],
            })
            .unwrap();
    }

    let out = engine
        .retrieve(RetrieveInput {
            query: "What is the relationship between Xiaoming and Lihua?".to_string(),
            context: retrieve_ctx(),
            top_k: 3,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    assert!(!out.results.is_empty(), "query must return results");

    // First retrieval: count=1, last_retrieved_at set, co-activations recorded
    // with the other result members.
    for r in &out.results {
        let (count, last_at, co_len) = activation_of(&engine, r.memory.id);
        assert_eq!(count, 1, "retrieval_count must be 1 after first retrieval");
        assert!(last_at.is_some(), "last_retrieved_at must be set");
        assert!(
            co_len >= 1,
            "co-activations must record the other result members, got {co_len}"
        );
        // usage_score untouched by retrieval (owned by feedback).
        assert_eq!(
            r.memory.activation.usage_score.value(),
            0.5,
            "usage_score must stay neutral on retrieval"
        );
    }

    // Second retrieval: count=2.
    let out2 = engine
        .retrieve(RetrieveInput {
            query: "What is the relationship between Xiaoming and Lihua?".to_string(),
            context: retrieve_ctx(),
            top_k: 3,
            max_hops: Some(1),
            retrieval_mode: RetrievalMode::Balanced,
        })
        .unwrap();
    for r in &out2.results {
        let (count, _, _) = activation_of(&engine, r.memory.id);
        assert_eq!(count, 2, "retrieval_count must accumulate");
    }
    engine.close().unwrap();
}
