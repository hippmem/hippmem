//! Activation log: records retrieval, usage, and co-activation events (03 §8, 05 §6).
//!
//! Persisted to the redb ACTIVATION_LOG table, consumed by Hebbian / decay logic.

use crate::store::ACTIVATION_LOG;
use redb::{Database, ReadableDatabase, ReadableTable};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A single activation record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActivationRecord {
    pub retrieval_id: u64,
    pub used_memory_ids: Vec<u64>,
    pub signal: String,
    pub recorded_at_ms: i64,
}

/// Activation log accessor.
pub struct ActivationLogger {
    db: Arc<Database>,
}

impl ActivationLogger {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Appends an activation record (key = auto-increment id, simplified as retrieval_id + timestamp).
    pub fn record(&self, rec: &ActivationRecord) -> Result<(), String> {
        let key = (rec.retrieval_id as u128) << 32 | (rec.recorded_at_ms as u128 & 0xFFFF_FFFF);
        let data = bincode::serde::encode_to_vec(rec, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        let txn = self
            .db
            .begin_write()
            .map_err(|e| format!("begin_write: {}", e))?;
        {
            let mut table = txn
                .open_table(ACTIVATION_LOG)
                .map_err(|e| format!("open_table: {}", e))?;
            table
                .insert(key, data.as_slice())
                .map_err(|e| format!("insert: {}", e))?;
        }
        txn.commit().map_err(|e| format!("commit: {}", e))?;
        Ok(())
    }

    /// Reads all records.
    pub fn read_all(&self) -> Result<Vec<ActivationRecord>, String> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| format!("begin_read: {}", e))?;
        let table = txn
            .open_table(ACTIVATION_LOG)
            .map_err(|e| format!("open_table: {}", e))?;
        let iter = table.iter().map_err(|e| format!("iter: {}", e))?;
        let mut recs = Vec::new();
        for entry in iter.flatten() {
            let (_key, value) = entry;
            if let Ok((rec, _)) = bincode::serde::decode_from_slice::<ActivationRecord, _>(
                value.value(),
                bincode::config::standard(),
            ) {
                recs.push(rec);
            }
        }
        Ok(recs)
    }
}
