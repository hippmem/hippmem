//! Engine::delete — memory deletion (memory-management, M3).
//!
//! Delete-cascade: removes the memory from every *state* table (memory_kv,
//! the six inverted indexes, the graph overlay including in-edges held by
//! other memories, dense vectors, context links). *Audit* records stay:
//! activation_log and memory_log are append-only (constitution C7), and
//! query fingerprints / retrieval paths are historical traces.

use crate::{DeleteInput, DeleteOutput, Engine, EngineError, EngineResult};

impl Engine {
    /// Deletes the given memories and their cascade (edges, indexes,
    /// vectors, context links). Audit records are kept.
    pub fn delete(&self, input: DeleteInput) -> EngineResult<DeleteOutput> {
        use hippmem_store::context_links::remove_memory_links;
        use hippmem_store::graph::GraphStore;
        use hippmem_store::kv::{remove_memory_from_indexes, KvStore};
        use hippmem_store::store::{DENSE_VECTORS, LINK_OVERLAY, MEMORY_KV};

        let db = self.store.db_arc();
        let kv = KvStore::new(db.clone());
        let mut deleted: u64 = 0;
        let mut edges_removed: u64 = 0;

        // Load all units once: the in-edge cascade (memories pointing at a
        // deleted one) needs every unit's link list.
        let units = crate::retrieve_api::load_all_units(db.clone());

        for id in &input.memory_ids {
            // Not found → skip (idempotent-ish: only existing memories count).
            if kv.get(&id.0)?.is_none() {
                continue;
            }

            // 1) State removal (single transaction): memory_kv, dense
            //    vector, out-edge overlay.
            {
                let txn = db
                    .begin_write()
                    .map_err(|e| EngineError::Store(e.to_string()))?;
                {
                    let mut t = txn
                        .open_table(MEMORY_KV)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                    t.remove(id.0)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                }
                {
                    let mut t = txn
                        .open_table(DENSE_VECTORS)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                    t.remove(id.0)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                }
                {
                    let mut t = txn
                        .open_table(LINK_OVERLAY)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                    t.remove(id.0)
                        .map_err(|e| EngineError::Store(e.to_string()))?;
                }
                txn.commit()
                    .map_err(|e| EngineError::Store(e.to_string()))?;
            }

            // 2) In-edge cascade: strip links pointing at the deleted
            //    memory from every other unit (unit + graph overlay).
            let graph = GraphStore::new(db.clone());
            for unit in &units {
                if unit.id.0 == id.0 {
                    continue;
                }
                let remaining: Vec<_> = unit
                    .links
                    .iter()
                    .filter(|l| l.target_id.0 != id.0)
                    .cloned()
                    .collect();
                if remaining.len() != unit.links.len() {
                    let removed = (unit.links.len() - remaining.len()) as u64;
                    graph
                        .put_outgoing(unit.id, &remaining)
                        .map_err(EngineError::Store)?;
                    // Keep the authoritative unit in sync with the overlay.
                    if let Some(raw) = kv.get(&unit.id.0)? {
                        if let Ok((mut u, _)) = bincode::serde::decode_from_slice::<
                            hippmem_core::model::unit::MemoryUnit,
                            _,
                        >(
                            raw.as_slice(), bincode::config::standard()
                        ) {
                            u.links = remaining;
                            let enc =
                                bincode::serde::encode_to_vec(&u, bincode::config::standard())
                                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                            kv.put(unit.id.0, &enc)
                                .map_err(|e| EngineError::Store(e.to_string()))?;
                        }
                    }
                    edges_removed += removed;
                }
            }

            // 3) Inverted indexes + context links.
            remove_memory_from_indexes(db.clone(), id.0)
                .map_err(|e| EngineError::Store(e.to_string()))?;
            remove_memory_links(db.clone(), id.0).map_err(EngineError::Store)?;

            deleted += 1;
        }

        Ok(DeleteOutput {
            deleted,
            edges_removed,
        })
    }
}
