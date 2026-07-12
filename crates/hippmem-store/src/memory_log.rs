//! Raw memory log: append-only storage (Constitution C7).
//!
//! Each record is keyed by `MemoryId` (u128) and stores bincode-encoded raw bytes.
//! Existing keys cannot be overwritten.

use crate::store::{StoreError, StoreResult, MEMORY_LOG};
use redb::{Database, ReadableDatabase, ReadableTable};
use std::sync::Arc;

/// Raw memory log: append-only, no modifications, no deletions.
///
/// Corresponds to the `memory_log` table in 04 §5.
pub struct MemoryLog {
    db: Arc<Database>,
}

impl MemoryLog {
    /// Creates a log handle.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Appends a record.
    ///
    /// - `id`: u128 representation of the MemoryId.
    /// - `data`: raw bytes (upper layer is responsible for serialization).
    ///
    /// ## Errors
    /// - If `id` already exists, returns `StoreError::RecordExists`.
    pub fn append(&self, id: u128, data: &[u8]) -> StoreResult<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(MEMORY_LOG)?;
            // Check whether it already exists (append-only)
            if table.get(id)?.is_some() {
                return Err(StoreError::RecordExists(id));
            }
            table.insert(id, data)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Reads a record.
    ///
    /// Returns `None` if the record does not exist.
    pub fn get(&self, id: &u128) -> StoreResult<Option<Vec<u8>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(MEMORY_LOG)?;
        let val = table.get(*id)?;
        Ok(val.map(|v| v.value().to_vec()))
    }

    /// Reads all records (for Reindex / crash recovery).
    ///
    /// Returns `Vec<(id: u128, data: Vec<u8>)>`, the raw bytes of each record.
    /// The caller is responsible for deserializing the bytes into a `MemoryUnit`.
    pub fn read_all(&self) -> StoreResult<Vec<(u128, Vec<u8>)>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(MEMORY_LOG)?;
        let mut records = Vec::new();
        for entry in table.iter()? {
            let (key, value) = entry?;
            records.push((key.value(), value.value().to_vec()));
        }
        Ok(records)
    }
}
