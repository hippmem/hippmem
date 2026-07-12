//! Association graph persistence: bidirectional edge storage (link_overlay, 04 §5).
//!
//! Supports querying outgoing/incoming edges by MemoryId, for spreading activation and explain.

use crate::store::LINK_OVERLAY;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::AssociationLink;
use redb::{Database, ReadableDatabase};
use std::sync::Arc;

/// Association graph accessor.
pub struct GraphStore {
    db: Arc<Database>,
}

impl GraphStore {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Writes the outgoing-edge list of a node.
    pub fn put_outgoing(&self, node_id: MemoryId, links: &[AssociationLink]) -> Result<(), String> {
        let data = bincode::serde::encode_to_vec(links, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        let txn = self
            .db
            .begin_write()
            .map_err(|e| format!("begin_write: {}", e))?;
        {
            let mut table = txn
                .open_table(LINK_OVERLAY)
                .map_err(|e| format!("open_table: {}", e))?;
            table
                .insert(node_id.0, data.as_slice())
                .map_err(|e| format!("insert: {}", e))?;
        }
        txn.commit().map_err(|e| format!("commit: {}", e))?;
        Ok(())
    }

    /// Reads the outgoing-edge list of a node.
    pub fn get_outgoing(&self, node_id: &MemoryId) -> Result<Vec<AssociationLink>, String> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| format!("begin_read: {}", e))?;
        let table = txn
            .open_table(LINK_OVERLAY)
            .map_err(|e| format!("open_table: {}", e))?;
        if let Some(value) = table.get(node_id.0).map_err(|e| format!("get: {}", e))? {
            bincode::serde::decode_from_slice::<Vec<AssociationLink>, _>(
                value.value(),
                bincode::config::standard(),
            )
            .map(|(links, _)| links)
            .map_err(|e| format!("decode: {}", e))
        } else {
            Ok(vec![])
        }
    }
}
