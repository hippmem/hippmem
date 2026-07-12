//! Store trait and redb implementation.
//!
//! Corresponds to 04 §5 storage layout, ADR-001 (redb).

use redb::{Database, ReadableDatabase, ReadableTableMetadata, TableDefinition};
use std::path::Path;
use std::sync::Arc;

// ── Table definitions (04 §5) ──

/// `(MemoryId) -> RawRecord` bincode bytes, append-only (Constitution C7).
pub const MEMORY_LOG: TableDefinition<u128, &[u8]> = TableDefinition::new("memory_log");

/// `(MemoryId) -> MemoryUnit` bincode bytes.
pub const MEMORY_KV: TableDefinition<u128, &[u8]> = TableDefinition::new("memory_kv");

/// Entity inverted index: `(EntityKey) -> Vec<MemoryId>` bincode.
pub const ENTITY_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("entity_index");

/// Topic inverted index: `(TopicKey) -> Vec<MemoryId>` bincode.
pub const TOPIC_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("topic_index");

/// Goal inverted index: `(GoalKey) -> Vec<MemoryId>` bincode.
pub const GOAL_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("goal_index");

/// Event inverted index: `(EventKey) -> Vec<MemoryId>` bincode.
pub const EVENT_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("event_index");

/// Temporal inverted index: `(TemporalKey) -> Vec<MemoryId>` bincode.
pub const TEMPORAL_INDEX: TableDefinition<u32, &[u8]> = TableDefinition::new("temporal_index");

/// Causal inverted index: `(CausalKey) -> Vec<MemoryId>` bincode.
pub const CAUSAL_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("causal_index");

/// Association graph overlay: `(MemoryId) -> OutLinks/InLinks` bincode.
pub const LINK_OVERLAY: TableDefinition<u128, &[u8]> = TableDefinition::new("link_overlay");

/// Summary overlay: `(MemoryId) -> override relationships` bincode.
pub const SUMMARY_OVERLAY: TableDefinition<u128, &[u8]> = TableDefinition::new("summary_overlay");

/// Correction/conflict overlay: `(MemoryId) -> correction/conflict/deprecation relationships` bincode.
pub const CORRECTION_OVERLAY: TableDefinition<u128, &[u8]> =
    TableDefinition::new("correction_overlay");

/// Activation log: appends retrieval traces and co-activation events.
pub const ACTIVATION_LOG: TableDefinition<u128, &[u8]> = TableDefinition::new("activation_log");

/// Consolidation queue: pending background tasks.
pub const CONSOLIDATION_QUEUE: TableDefinition<u128, &[u8]> =
    TableDefinition::new("consolidation_queue");

/// Set of u128-key tables: used for batch initialization during `open`.
const U128_TABLES: &[TableDefinition<u128, &[u8]>] = &[
    MEMORY_LOG,
    MEMORY_KV,
    LINK_OVERLAY,
    SUMMARY_OVERLAY,
    CORRECTION_OVERLAY,
    ACTIVATION_LOG,
    CONSOLIDATION_QUEUE,
];

/// Set of u64-key tables.
const U64_TABLES: &[TableDefinition<u64, &[u8]>] = &[
    ENTITY_INDEX,
    TOPIC_INDEX,
    GOAL_INDEX,
    EVENT_INDEX,
    CAUSAL_INDEX,
];

/// Set of u32-key tables.
const U32_TABLES: &[TableDefinition<u32, &[u8]>] = &[TEMPORAL_INDEX];

// ── Store trait ──

/// Storage abstraction: defines the basic lifecycle of a persistent store.
///
/// Corresponds to the `hippmem-store` responsibility in 04 §2.
/// The first version only includes open/close; read/write methods are added in subsequent tasks.
pub trait Store: Sized {
    /// Opens or creates the storage.
    ///
    /// - Creates the file if the path does not exist; creates parent directories if missing.
    /// - All tables defined in 04 §5 are initialized during open.
    fn open(path: impl AsRef<Path>) -> Result<Self, StoreError>;

    /// Closes the storage and releases resources (redb auto-flushes on drop).
    fn close(self) -> Result<(), StoreError>;
}

// ── RedbStore ──

/// redb-backed implementation of Store.
///
/// Wraps `Arc<redb::Database>` and manages all tables defined in 04 §5.
pub struct RedbStore {
    db: Arc<Database>,
}

// ── StoreError ──

/// Store-layer error type: covers redb sub-errors and IO errors.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Database-level error (create/open/file lock, etc.).
    #[error("Database error: {0}")]
    Database(#[from] redb::DatabaseError),

    /// Transaction-level error.
    #[error("Transaction error: {0}")]
    Transaction(#[from] redb::TransactionError),

    /// Table operation error.
    #[error("Table error: {0}")]
    Table(#[from] redb::TableError),

    /// Commit error.
    #[error("Commit error: {0}")]
    Commit(#[from] redb::CommitError),

    /// Storage operation error (get/insert, etc.).
    #[error("Storage error: {0}")]
    Storage(#[from] redb::StorageError),

    /// Underlying IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Record already exists (append-only conflict).
    #[error("Record already exists, cannot be overwritten: id={0}")]
    RecordExists(u128),
}

/// Store-layer Result alias.
pub type StoreResult<T> = Result<T, StoreError>;

// ── StoreStats ──

/// Storage diagnostic stats: reflects the record counts of each table.
///
/// Corresponds to 05#inspect, Constitution C9 (diagnosability / traceability).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreStats {
    /// Number of records in `memory_log`.
    pub memory_log_count: u64,
    /// Number of records in `memory_kv`.
    pub memory_kv_count: u64,
    /// Number of keys in `entity_index`.
    pub entity_index_size: u64,
    /// Number of keys in `topic_index`.
    pub topic_index_size: u64,
    /// Number of keys in `goal_index`.
    pub goal_index_size: u64,
    /// Number of keys in `event_index`.
    pub event_index_size: u64,
    /// Number of keys in `temporal_index`.
    pub temporal_index_size: u64,
    /// Number of keys in `causal_index`.
    pub causal_index_size: u64,
}

impl Store for RedbStore {
    fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let path = path.as_ref();

        // Auto-create parent directories
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let db = Database::create(path)?;

        // Create all 12 tables (redb creates a table on first open_table)
        {
            let txn = db.begin_write()?;
            for def in U128_TABLES {
                let _ = txn.open_table(*def)?;
            }
            for def in U64_TABLES {
                let _ = txn.open_table(*def)?;
            }
            for def in U32_TABLES {
                let _ = txn.open_table(*def)?;
            }
            txn.commit()?;
        }

        Ok(Self { db: Arc::new(db) })
    }

    fn close(self) -> StoreResult<()> {
        // redb 3.x: Database auto-flushes on drop; Arc drops when the last reference is released.
        drop(self.db);
        Ok(())
    }
}

impl RedbStore {
    /// Returns a reference to the inner redb Database.
    pub fn db(&self) -> &Database {
        &self.db
    }

    /// Returns the inner `Arc<Database>` (for use by submodules such as MemoryLog/KvStore).
    pub fn db_arc(&self) -> Arc<Database> {
        Arc::clone(&self.db)
    }

    /// Reads the record counts of each table, returns `StoreStats`.
    ///
    /// For diagnostics and observability (Constitution C9).
    pub fn stats(&self) -> StoreResult<StoreStats> {
        let txn = self.db.begin_read()?;

        let memory_log_count = txn.open_table(MEMORY_LOG)?.len()?;
        let memory_kv_count = txn.open_table(MEMORY_KV)?.len()?;
        let entity_index_size = txn.open_table(ENTITY_INDEX)?.len()?;
        let topic_index_size = txn.open_table(TOPIC_INDEX)?.len()?;
        let goal_index_size = txn.open_table(GOAL_INDEX)?.len()?;
        let event_index_size = txn.open_table(EVENT_INDEX)?.len()?;
        let temporal_index_size = txn.open_table(TEMPORAL_INDEX)?.len()?;
        let causal_index_size = txn.open_table(CAUSAL_INDEX)?.len()?;

        Ok(StoreStats {
            memory_log_count,
            memory_kv_count,
            entity_index_size,
            topic_index_size,
            goal_index_size,
            event_index_size,
            temporal_index_size,
            causal_index_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redb::TableHandle;

    /// Unit test: table-definition constant names match 04 §5.
    #[test]
    fn table_definitions_have_names() {
        assert_eq!(MEMORY_LOG.name(), "memory_log");
        assert_eq!(MEMORY_KV.name(), "memory_kv");
        assert_eq!(ENTITY_INDEX.name(), "entity_index");
        assert_eq!(TOPIC_INDEX.name(), "topic_index");
        assert_eq!(GOAL_INDEX.name(), "goal_index");
        assert_eq!(EVENT_INDEX.name(), "event_index");
        assert_eq!(TEMPORAL_INDEX.name(), "temporal_index");
        assert_eq!(CAUSAL_INDEX.name(), "causal_index");
        assert_eq!(LINK_OVERLAY.name(), "link_overlay");
        assert_eq!(SUMMARY_OVERLAY.name(), "summary_overlay");
        assert_eq!(CORRECTION_OVERLAY.name(), "correction_overlay");
        assert_eq!(ACTIVATION_LOG.name(), "activation_log");
        assert_eq!(CONSOLIDATION_QUEUE.name(), "consolidation_queue");
    }

    /// All 13 tables = all tables defined in 04 §5.
    #[test]
    fn all_tables_count_is_thirteen() {
        let total = U128_TABLES.len() + U64_TABLES.len() + U32_TABLES.len();
        assert_eq!(total, 13);
    }
}
