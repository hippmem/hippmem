//! Enriched -> index writeback: after the strong semantic dimensions are
//! completed, write goal/event/causal keys into the corresponding inverted
//! indexes (03 §4.5, 04 §5).
//!
//! After the enriched stage, goal_index / event_index / causal_index are non-empty.

use hippmem_core::model::unit::MemoryUnit;
use hippmem_store::kv::InvertedIndex;
use xxhash_rust::xxh3::xxh3_64;

/// Write the enriched goal/event/causal keys into the corresponding inverted
/// indexes.
///
/// - Recompute goal_keys/event_keys/causal_keys from `unit.understanding`
///   (isomorphic with `goal_keys_from`/`event_keys_from`/`causal_keys_from`
///   in keys.rs)
/// - Write into `goal_index`/`event_index`/`causal_index`
/// - If a dimension's key list is empty, the corresponding index write is
///   skipped (no side effect)
///
/// # Used by
/// - write path: called after run_enrich_sync
/// - Reindex: existing enriched MemoryUnit can rebuild its indexes via this
///   function
pub fn index_enriched_keys(
    unit: &MemoryUnit,
    inverted: &InvertedIndex,
    memory_id: u128,
) -> Result<(), String> {
    let u = &unit.understanding;

    // goal_keys: stable hash of each goals[].description
    for g in &u.goals {
        let key = xxh3_64(g.description.as_bytes());
        inverted
            .add_goal(key, memory_id)
            .map_err(|e| e.to_string())?;
    }

    // event_keys: stable hash of each events[].action
    for e in &u.events {
        let key = xxh3_64(e.action.as_bytes());
        inverted
            .add_event(key, memory_id)
            .map_err(|e| e.to_string())?;
    }

    // causal_keys: combined hash of (cause -> effect) for each causal_claim
    for c in &u.causal_claims {
        let mut input = c.cause.as_bytes().to_vec();
        input.extend_from_slice(b" -> ");
        input.extend_from_slice(c.effect.as_bytes());
        let key = xxh3_64(&input);
        inverted
            .add_causal(key, memory_id)
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::ids::MemoryId;
    use hippmem_core::model::links::{
        ActivationState, AssociationKeys, LexicalSignature, SemanticSignature,
    };
    use hippmem_core::model::understanding::{
        CausalClaim, CausalKind, EventFrame, GoalFrame, GoalStatus, MemoryUnderstanding,
    };
    use hippmem_core::model::unit::{
        ContentType, Language, MemoryContent, MemoryLifecycle, MemoryStage, MemoryUnit, Provenance,
        WriteContext,
    };
    use hippmem_core::score::UnitScore;
    use hippmem_core::time::Timestamp;
    use hippmem_store::store::{RedbStore, Store};
    use tempfile::TempDir;

    fn make_enriched_unit(
        goals: Vec<&str>,
        events: Vec<&str>,
        causals: Vec<(&str, &str)>,
    ) -> MemoryUnit {
        MemoryUnit {
            schema_version: 1,
            id: MemoryId(1),
            created_at: Timestamp(0),
            updated_at: Timestamp(0),
            content: MemoryContent {
                raw: "test".into(),
                summary: None,
                normalized: None,
                language: Language::Zh,
                content_type: ContentType::UserStatement,
            },
            context: WriteContext {
                conversation_id: None,
                session_id: None,
                project_id: None,
                task_id: None,
                user_id: None,
                local_time: Timestamp(0),
                preceding_memory_ids: vec![],
                source_refs: vec![],
            },
            understanding: MemoryUnderstanding {
                entities: vec![],
                events: events
                    .iter()
                    .map(|a| EventFrame {
                        action: a.to_string(),
                        participants: vec![],
                        occurred_at: None,
                        outcome: None,
                        confidence: UnitScore::new(0.7),
                    })
                    .collect(),
                goals: goals
                    .iter()
                    .map(|d| GoalFrame {
                        description: d.to_string(),
                        status: GoalStatus::Active,
                        constraints: vec![],
                        confidence: UnitScore::new(0.8),
                    })
                    .collect(),
                decisions: vec![],
                preferences: vec![],
                emotions: vec![],
                causal_claims: causals
                    .iter()
                    .map(|(c, e)| CausalClaim {
                        cause: c.to_string(),
                        effect: e.to_string(),
                        kind: CausalKind::Explicit,
                        confidence: UnitScore::new(0.6),
                        evidence_span: None,
                    })
                    .collect(),
                contradictions: vec![],
                topics: vec![],
                importance: UnitScore::new(0.5),
                confidence: UnitScore::new(0.5),
            },
            association_keys: AssociationKeys {
                entity_keys: vec![],
                temporal_keys: vec![],
                lexical_signature: LexicalSignature { simhash: [0; 4] },
                semantic_signature: SemanticSignature {
                    lexical_simhash: [0; 4],
                    dense_embedding_ref: None,
                    binary_code: [0, 0],
                    topic_minhash: [0u32; 16],
                },
                topic_keys: vec![],
                emotion_keys: vec![],
                goal_keys: vec![],
                event_keys: vec![],
                causal_keys: vec![],
            },
            links: vec![],
            activation: ActivationState {
                last_retrieved_at: None,
                retrieval_count: 0,
                co_activations: vec![],
                usage_score: UnitScore::new(0.5),
            },
            lifecycle: MemoryLifecycle::Active,
            provenance: Provenance {
                origin: hippmem_core::model::unit::SourceKind::Conversation,
                generated_by: hippmem_core::model::unit::GeneratedBy::UserDirect,
                reliability: UnitScore::new(0.5),
                evidence_refs: vec![],
                revision_history: vec![],
            },
            stage: MemoryStage::Enriched,
        }
    }

    fn temp_inverted() -> (TempDir, InvertedIndex) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("t.redb");
        let store = RedbStore::open(&path).unwrap();
        let inverted = InvertedIndex::new(store.db_arc());
        (tmp, inverted)
    }

    #[test]
    fn writes_goal_key() {
        let (_tmp, inv) = temp_inverted();
        let unit = make_enriched_unit(vec!["learn Rust"], vec![], vec![]);
        index_enriched_keys(&unit, &inv, 1).unwrap();
        let key = xxh3_64("learn Rust".as_bytes());
        assert!(inv.get_goal(&key).unwrap().contains(&1));
    }

    #[test]
    fn writes_event_key() {
        let (_tmp, inv) = temp_inverted();
        let unit = make_enriched_unit(vec![], vec!["deploy to production"], vec![]);
        index_enriched_keys(&unit, &inv, 2).unwrap();
        let key = xxh3_64("deploy to production".as_bytes());
        assert!(inv.get_event(&key).unwrap().contains(&2));
    }

    #[test]
    fn writes_causal_key() {
        let (_tmp, inv) = temp_inverted();
        let unit = make_enriched_unit(vec![], vec![], vec![("learn Rust", "high efficiency")]);
        index_enriched_keys(&unit, &inv, 3).unwrap();
        let ck = xxh3_64("learn Rust -> high efficiency".as_bytes());
        assert!(inv.get_causal(&ck).unwrap().contains(&3));
    }

    #[test]
    fn all_three_dimensions() {
        let (_tmp, inv) = temp_inverted();
        let unit = make_enriched_unit(
            vec!["Goal-A"],
            vec!["Event-B"],
            vec![("cause-C", "effect-D")],
        );
        index_enriched_keys(&unit, &inv, 4).unwrap();
        assert!(inv
            .get_goal(&xxh3_64("Goal-A".as_bytes()))
            .unwrap()
            .contains(&4));
        assert!(inv
            .get_event(&xxh3_64("Event-B".as_bytes()))
            .unwrap()
            .contains(&4));
        assert!(inv
            .get_causal(&xxh3_64("cause-C -> effect-D".as_bytes()))
            .unwrap()
            .contains(&4));
    }
}
