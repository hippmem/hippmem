//! memory_kv: memory KV store, allows overwriting.
//!
//! Each record is keyed by `MemoryId` (u128) and stores bincode-encoded MemoryUnit bytes.
//! Unlike memory_log, kv allows overwriting existing records.

use crate::store::{
    StoreResult, CAUSAL_INDEX, ENTITY_INDEX, EVENT_INDEX, GOAL_INDEX, LINK_OVERLAY, MEMORY_KV,
    MEMORY_LOG, TEMPORAL_INDEX, TOPIC_INDEX,
};
use redb::{Database, ReadableDatabase, ReadableTable};
use std::sync::Arc;

/// Memory KV store: MemoryId -> bincode bytes.
///
/// Corresponds to the `memory_kv` table in 04 §5.
pub struct KvStore {
    db: Arc<Database>,
}

impl KvStore {
    /// Creates a KV handle.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Writes a record, overwriting any existing value.
    ///
    /// - `id`: u128 representation of the MemoryId.
    /// - `data`: bincode-encoded MemoryUnit bytes.
    pub fn put(&self, id: u128, data: &[u8]) -> StoreResult<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(MEMORY_KV)?;
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
        let table = txn.open_table(MEMORY_KV)?;
        let val = table.get(*id)?;
        Ok(val.map(|v| v.value().to_vec()))
    }
}

// ── Inverted indexes ──

/// Attribute inverted index: five dimensions — Entity/Topic/Goal/Event/Temporal.
///
/// Each dimension uses an attribute key (e.g. EntityKey=u64) as the index key
/// and stores the associated `Vec<MemoryId>` (bincode-encoded).
///
/// Corresponds to `entity_index`/`topic_index`/`goal_index`/`event_index`/`temporal_index` in 04 §5.
pub struct InvertedIndex {
    db: Arc<Database>,
}

impl InvertedIndex {
    /// Creates an inverted-index handle from a `RedbStore`.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    // ── Entity ──

    /// Appends a memory_id to `entity_index`.
    pub fn add_entity(&self, key: u64, id: u128) -> StoreResult<()> {
        self.append_to_u64_table(ENTITY_INDEX, key, id)
    }

    /// Queries `entity_index`.
    pub fn get_entity(&self, key: &u64) -> StoreResult<Vec<u128>> {
        self.read_u64_table(ENTITY_INDEX, key)
    }

    // ── Topic ──

    /// Appends a memory_id to `topic_index`.
    pub fn add_topic(&self, key: u64, id: u128) -> StoreResult<()> {
        self.append_to_u64_table(TOPIC_INDEX, key, id)
    }

    /// Queries `topic_index`.
    pub fn get_topic(&self, key: &u64) -> StoreResult<Vec<u128>> {
        self.read_u64_table(TOPIC_INDEX, key)
    }

    // ── Goal ──

    /// Appends a memory_id to `goal_index`.
    pub fn add_goal(&self, key: u64, id: u128) -> StoreResult<()> {
        self.append_to_u64_table(GOAL_INDEX, key, id)
    }

    /// Queries `goal_index`.
    pub fn get_goal(&self, key: &u64) -> StoreResult<Vec<u128>> {
        self.read_u64_table(GOAL_INDEX, key)
    }

    // ── Event ──

    /// Appends a memory_id to `event_index`.
    pub fn add_event(&self, key: u64, id: u128) -> StoreResult<()> {
        self.append_to_u64_table(EVENT_INDEX, key, id)
    }

    /// Queries `event_index`.
    pub fn get_event(&self, key: &u64) -> StoreResult<Vec<u128>> {
        self.read_u64_table(EVENT_INDEX, key)
    }

    // ── Causal ──

    /// Appends a memory_id to `causal_index`.
    pub fn add_causal(&self, key: u64, id: u128) -> StoreResult<()> {
        self.append_to_u64_table(CAUSAL_INDEX, key, id)
    }

    /// Queries `causal_index`.
    pub fn get_causal(&self, key: &u64) -> StoreResult<Vec<u128>> {
        self.read_u64_table(CAUSAL_INDEX, key)
    }

    // ── Temporal ──

    /// Appends a memory_id to `temporal_index`.
    pub fn add_temporal(&self, key: u32, id: u128) -> StoreResult<()> {
        self.append_to_u32_table(TEMPORAL_INDEX, key, id)
    }

    /// Queries `temporal_index`.
    pub fn get_temporal(&self, key: &u32) -> StoreResult<Vec<u128>> {
        self.read_u32_table(TEMPORAL_INDEX, key)
    }

    /// Batch-writes all inverted-index keys (single transaction, avoids the O(N) cost of per-key commits).
    ///
    /// Writes entity/topic/temporal/goal/event/causal keys in one redb transaction.
    #[allow(clippy::too_many_arguments)]
    pub fn add_all(
        &self,
        entity_keys: &[u64],
        topic_keys: &[u64],
        temporal_keys: &[u32],
        goal_keys: &[u64],
        event_keys: &[u64],
        causal_keys: &[u64],
        id: u128,
    ) -> StoreResult<()> {
        if entity_keys.is_empty()
            && topic_keys.is_empty()
            && temporal_keys.is_empty()
            && goal_keys.is_empty()
            && event_keys.is_empty()
            && causal_keys.is_empty()
        {
            return Ok(());
        }

        let txn = self.db.begin_write()?;
        {
            // Entity
            if !entity_keys.is_empty() {
                let mut table = txn.open_table(ENTITY_INDEX)?;
                for &key in entity_keys {
                    let mut ids = decode_ids(table.get(key)?);
                    if !ids.contains(&id) {
                        ids.push(id);
                        let encoded =
                            bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                                .expect("bincode encoding Vec<u128> should not fail");
                        table.insert(key, encoded.as_slice())?;
                    }
                }
            }
            // Topic
            if !topic_keys.is_empty() {
                let mut table = txn.open_table(TOPIC_INDEX)?;
                for &key in topic_keys {
                    let mut ids = decode_ids(table.get(key)?);
                    if !ids.contains(&id) {
                        ids.push(id);
                        let encoded =
                            bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                                .expect("bincode encoding Vec<u128> should not fail");
                        table.insert(key, encoded.as_slice())?;
                    }
                }
            }
            // Temporal (u32 keys)
            if !temporal_keys.is_empty() {
                let mut table = txn.open_table(TEMPORAL_INDEX)?;
                for &key in temporal_keys {
                    let mut ids = decode_ids(table.get(key)?);
                    if !ids.contains(&id) {
                        ids.push(id);
                        let encoded =
                            bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                                .expect("bincode encoding Vec<u128> should not fail");
                        table.insert(key, encoded.as_slice())?;
                    }
                }
            }
            // Goal
            if !goal_keys.is_empty() {
                let mut table = txn.open_table(GOAL_INDEX)?;
                for &key in goal_keys {
                    let mut ids = decode_ids(table.get(key)?);
                    if !ids.contains(&id) {
                        ids.push(id);
                        let encoded =
                            bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                                .expect("bincode encoding Vec<u128> should not fail");
                        table.insert(key, encoded.as_slice())?;
                    }
                }
            }
            // Event
            if !event_keys.is_empty() {
                let mut table = txn.open_table(EVENT_INDEX)?;
                for &key in event_keys {
                    let mut ids = decode_ids(table.get(key)?);
                    if !ids.contains(&id) {
                        ids.push(id);
                        let encoded =
                            bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                                .expect("bincode encoding Vec<u128> should not fail");
                        table.insert(key, encoded.as_slice())?;
                    }
                }
            }
            // Causal
            if !causal_keys.is_empty() {
                let mut table = txn.open_table(CAUSAL_INDEX)?;
                for &key in causal_keys {
                    let mut ids = decode_ids(table.get(key)?);
                    if !ids.contains(&id) {
                        ids.push(id);
                        let encoded =
                            bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                                .expect("bincode encoding Vec<u128> should not fail");
                        table.insert(key, encoded.as_slice())?;
                    }
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    // ── Internal helpers ──

    /// Appends a memory_id to a u64-key table (with dedup).
    fn append_to_u64_table(
        &self,
        def: redb::TableDefinition<'static, u64, &[u8]>,
        key: u64,
        id: u128,
    ) -> StoreResult<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(def)?;
            let mut ids = decode_ids(table.get(key)?);
            if !ids.contains(&id) {
                ids.push(id);
                let encoded = bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                    .expect("bincode encoding Vec<u128> should not fail");
                table.insert(key, encoded.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Reads the memory_id list from a u64-key table.
    fn read_u64_table(
        &self,
        def: redb::TableDefinition<'static, u64, &[u8]>,
        key: &u64,
    ) -> StoreResult<Vec<u128>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(def)?;
        Ok(decode_ids(table.get(*key)?))
    }

    /// Appends a memory_id to a u32-key table (with dedup).
    fn append_to_u32_table(
        &self,
        def: redb::TableDefinition<'static, u32, &[u8]>,
        key: u32,
        id: u128,
    ) -> StoreResult<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(def)?;
            let mut ids = decode_ids(table.get(key)?);
            if !ids.contains(&id) {
                ids.push(id);
                let encoded = bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                    .expect("bincode encoding Vec<u128> should not fail");
                table.insert(key, encoded.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Reads the memory_id list from a u32-key table.
    fn read_u32_table(
        &self,
        def: redb::TableDefinition<'static, u32, &[u8]>,
        key: &u32,
    ) -> StoreResult<Vec<u128>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(def)?;
        Ok(decode_ids(table.get(*key)?))
    }
}

/// Decodes a bincode-encoded Vec<u128> from a redb AccessGuard.
fn decode_ids(entry: Option<redb::AccessGuard<&[u8]>>) -> Vec<u128> {
    match entry {
        Some(v) => {
            let (ids, _n): (Vec<u128>, usize) =
                bincode::serde::decode_from_slice(v.value(), bincode::config::standard())
                    .unwrap_or_default();
            ids
        }
        None => Vec::new(),
    }
}

/// Batch persistence: merges all redb writes for a memory unit into a single transaction.
///
/// Writes memory_log (optional) + memory_kv + 6 inverted indexes + link_overlay
/// in one go, avoiding the transaction overhead of per-operation commits.
#[allow(clippy::too_many_arguments)]
pub fn persist_memory_unit(
    db: Arc<Database>,
    id: u128,
    bincode_unit: &[u8],
    bincode_links: &[u8],
    entity_keys: &[u64],
    topic_keys: &[u64],
    temporal_keys: &[u32],
    goal_keys: &[u64],
    event_keys: &[u64],
    causal_keys: &[u64],
    skip_memory_log: bool,
) -> StoreResult<()> {
    let txn = db.begin_write()?;
    {
        // memory_log (append-only, skip for reindex)
        if !skip_memory_log {
            let mut table = txn.open_table(MEMORY_LOG)?;
            if table.get(id)?.is_some() {
                return Err(crate::store::StoreError::RecordExists(id));
            }
            table.insert(id, bincode_unit)?;
        }

        // memory_kv (overwritable)
        {
            let mut table = txn.open_table(MEMORY_KV)?;
            table.insert(id, bincode_unit)?;
        }

        // link_overlay
        {
            let mut table = txn.open_table(LINK_OVERLAY)?;
            table.insert(id, bincode_links)?;
        }

        // Inverted indexes (batch-written in a single transaction, reuses add_all logic)
        let append_ids =
            |table: &mut redb::Table<u64, &[u8]>, key: u64, id: u128| -> StoreResult<()> {
                let mut ids = decode_ids(table.get(key)?);
                if !ids.contains(&id) {
                    ids.push(id);
                    let encoded = bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                        .expect("bincode encoding Vec<u128> should not fail");
                    table.insert(key, encoded.as_slice())?;
                }
                Ok(())
            };

        if !entity_keys.is_empty() {
            let mut table = txn.open_table(ENTITY_INDEX)?;
            for &key in entity_keys {
                append_ids(&mut table, key, id)?;
            }
        }
        if !topic_keys.is_empty() {
            let mut table = txn.open_table(TOPIC_INDEX)?;
            for &key in topic_keys {
                append_ids(&mut table, key, id)?;
            }
        }
        if !goal_keys.is_empty() {
            let mut table = txn.open_table(GOAL_INDEX)?;
            for &key in goal_keys {
                append_ids(&mut table, key, id)?;
            }
        }
        if !event_keys.is_empty() {
            let mut table = txn.open_table(EVENT_INDEX)?;
            for &key in event_keys {
                append_ids(&mut table, key, id)?;
            }
        }
        if !causal_keys.is_empty() {
            let mut table = txn.open_table(CAUSAL_INDEX)?;
            for &key in causal_keys {
                append_ids(&mut table, key, id)?;
            }
        }
        if !temporal_keys.is_empty() {
            let mut table = txn.open_table(TEMPORAL_INDEX)?;
            for &key in temporal_keys {
                let mut ids = decode_ids(table.get(key)?);
                if !ids.contains(&id) {
                    ids.push(id);
                    let encoded = bincode::serde::encode_to_vec(&ids, bincode::config::standard())
                        .expect("bincode encoding Vec<u128> should not fail");
                    table.insert(key, encoded.as_slice())?;
                }
            }
        }
    }
    txn.commit()?;
    Ok(())
}
