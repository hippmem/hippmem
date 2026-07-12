//! Engine::inspect — diagnostics API (05 §6, 09 §4.5).

use crate::{Engine, EngineResult, InspectQuery};
use hippmem_core::model::links::RecallChannel;

impl Engine {
    /// Diagnostic query: StoreStats/QueueStatus, etc.
    pub fn inspect(&self, query: InspectQuery) -> EngineResult<crate::InspectReport> {
        match query {
            InspectQuery::StoreStats => {
                let units = crate::retrieve_api::load_all_units(self.store.db_arc());
                let mut edge_count = 0u64;
                for u in &units {
                    edge_count += u.links.len() as u64;
                }
                Ok(crate::InspectReport::StoreStats(crate::StoreStats {
                    memory_count: units.len() as u64,
                    edge_count,
                    observing_edge_count: 0,
                    per_index_size: vec![
                        (RecallChannel::Bm25, 0),
                        (RecallChannel::EntityInverted, 0),
                    ],
                    queue_backlog: 0,
                    store_bytes: 0,
                }))
            }
            InspectQuery::QueueStatus => {
                Ok(crate::InspectReport::QueueStatus(crate::QueueStatus {
                    pending_enrich: 0,
                    pending_consolidate: 0,
                    in_flight: 0,
                    oldest_pending_age_ms: 0,
                }))
            }
            InspectQuery::Memory(id) => {
                let units = crate::retrieve_api::load_all_units(self.store.db_arc());
                let unit = units
                    .iter()
                    .find(|u| u.id == id)
                    .cloned()
                    .ok_or(crate::EngineError::NotFound(id))?;

                let out_edges: Vec<crate::EdgeView> = unit
                    .links
                    .iter()
                    .map(|l| crate::EdgeView {
                        from: unit.id,
                        to: l.target_id,
                        link_type: l.link_type,
                        strength: l.strength.value(),
                        confidence: l.confidence.value(),
                        activation_count: l.activation_count,
                        evidence: l.evidence.note.clone().unwrap_or_default(),
                    })
                    .collect();

                // In-edges: iterate over all memories, finding edges where target_id == queried id
                let in_edges: Vec<crate::EdgeView> = units
                    .iter()
                    .flat_map(|other| {
                        other.links.iter().filter_map(move |l| {
                            if l.target_id == id {
                                Some(crate::EdgeView {
                                    from: other.id,
                                    to: l.target_id,
                                    link_type: l.link_type,
                                    strength: l.strength.value(),
                                    confidence: l.confidence.value(),
                                    activation_count: l.activation_count,
                                    evidence: l.evidence.note.clone().unwrap_or_default(),
                                })
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

                let stage = unit.stage;
                let lifecycle = unit.lifecycle.clone();

                Ok(crate::InspectReport::Memory(Box::new(
                    crate::MemoryInspect {
                        unit,
                        out_edges,
                        in_edges,
                        stage,
                        lifecycle,
                    },
                )))
            }
            _ => Ok(crate::InspectReport::StoreStats(crate::StoreStats {
                memory_count: 0,
                edge_count: 0,
                observing_edge_count: 0,
                per_index_size: vec![],
                queue_backlog: 0,
                store_bytes: 0,
            })),
        }
    }
}
