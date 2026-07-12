//! Engine::consolidate — consolidation API (05 §5, 09 §4.4).

use crate::{ConsolidationReport, ConsolidationScope, Engine, EngineError, EngineResult};
use hippmem_consolidation::hebbian::ActivationLog;
use hippmem_consolidation::worker::ConsolidationWorker;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::unit::MemoryUnit;
use hippmem_core::time::{Clock, SystemClock};
use hippmem_model::deterministic::summarize::DeterministicSummarizer;
use hippmem_store::activation_log::ActivationLogger;
use hippmem_store::kv::KvStore;
use hippmem_store::memory_log::MemoryLog;
use hippmem_store::store::{
    ACTIVATION_LOG, CAUSAL_INDEX, CONSOLIDATION_QUEUE, CORRECTION_OVERLAY, ENTITY_INDEX,
    EVENT_INDEX, GOAL_INDEX, LINK_OVERLAY, MEMORY_KV, SUMMARY_OVERLAY, TEMPORAL_INDEX, TOPIC_INDEX,
};
use std::time::Instant;

impl Engine {
    /// Runs consolidation: Hebbian→decay→compaction→summary, covering the specified scope.
    /// Reindex scope: rebuilds all secondary indexes from memory_log (no data loss).
    pub fn consolidate(&self, scope: ConsolidationScope) -> EngineResult<ConsolidationReport> {
        if matches!(scope, ConsolidationScope::Reindex) {
            return self.consolidate_reindex();
        }
        self.consolidate_incremental()
    }

    /// Standard incremental consolidation (Hebbian→decay→compaction→summary).
    fn consolidate_incremental(&self) -> EngineResult<ConsolidationReport> {
        let start = Instant::now();
        let clock = SystemClock;
        let now = clock.now();

        // 1. Load all data in the store
        let mut units = crate::retrieve_api::load_all_units(self.store.db_arc());

        // 2. Read activation_log and build co-activation pairs
        let logger = ActivationLogger::new(self.store.db_arc());
        let mut activation_log = ActivationLog::default();
        if let Ok(records) = logger.read_all() {
            for rec in &records {
                for i in 0..rec.used_memory_ids.len() {
                    for j in (i + 1)..rec.used_memory_ids.len() {
                        let a = MemoryId(rec.used_memory_ids[i] as u128);
                        let b = MemoryId(rec.used_memory_ids[j] as u128);
                        let ts = hippmem_core::time::Timestamp::from_millis(rec.recorded_at_ms);
                        activation_log.record(a, ts, 0.5);
                        activation_log.record(b, ts, 0.5);
                    }
                }
            }
        }
        let co_activations = activation_log.co_activation_pairs(3_600_000);

        // 3. Run consolidation cycle (Hebbian→decay→compaction→summary)
        let summarizer = DeterministicSummarizer;
        let mut worker = ConsolidationWorker::default();
        let cycle_stats = worker.run_cycle(&mut units, &co_activations, now, Some(&summarizer));

        // 4. Persist the modified units back to the store
        let kv = KvStore::new(self.store.db_arc());
        for unit in &units {
            let bincode_unit = bincode::serde::encode_to_vec(unit, bincode::config::standard())
                .map_err(|e| EngineError::Internal(e.to_string()))?;
            kv.put(unit.id.0, &bincode_unit)
                .map_err(|e| EngineError::Store(e.to_string()))?;
        }

        // 4b. Persist summary memory
        if let Some(ref summary_unit) = cycle_stats.summary_unit {
            let bincode_summary =
                bincode::serde::encode_to_vec(summary_unit, bincode::config::standard())
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
            kv.put(summary_unit.id.0, &bincode_summary)
                .map_err(|e| EngineError::Store(e.to_string()))?;
            let graph = hippmem_store::graph::GraphStore::new(self.store.db_arc());
            graph
                .put_outgoing(summary_unit.id, &summary_unit.links)
                .map_err(EngineError::Store)?;
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(ConsolidationReport {
            memories_processed: units.len() as u64
                + if cycle_stats.summary_unit.is_some() {
                    1
                } else {
                    0
                },
            edges_decayed: cycle_stats.edges_decayed,
            edges_archived: cycle_stats.edges_archived,
            edges_merged: cycle_stats.hebbian_applied,
            observation_promoted: 0,
            summaries_created: cycle_stats.summaries_created,
            contradictions_found: 0,
            reindexed: false,
            elapsed_ms,
        })
    }

    /// Reindex: rebuilds all secondary indexes from memory_log (no data loss, MemoryId unchanged).
    fn consolidate_reindex(&self) -> EngineResult<ConsolidationReport> {
        let start = Instant::now();

        // 1. Read all raw records from memory_log
        let log = MemoryLog::new(self.store.db_arc());
        let raw_records = log
            .read_all()
            .map_err(|e| EngineError::Store(e.to_string()))?;
        let mut units: Vec<(u128, MemoryUnit)> = Vec::with_capacity(raw_records.len());
        for (id, data) in &raw_records {
            let (unit, _): (MemoryUnit, _) =
                bincode::serde::decode_from_slice(data, bincode::config::standard()).map_err(
                    |e| EngineError::Internal(format!("failed to deserialize MemoryUnit: {}", e)),
                )?;
            units.push((*id, unit));
        }
        let total = units.len() as u64;

        // 2. Clear all secondary tables (preserve MEMORY_LOG)
        clear_all_secondary_tables(self.store.db_arc())
            .map_err(|e| EngineError::Store(e.to_string()))?;

        // 3. Clear the Tantivy fulltext index (rebuild after deleting the directory)
        {
            let mut ft = self.fulltext_index.lock();
            let _ = ft.commit();
            drop(ft);
            if self.fulltext_dir.exists() {
                std::fs::remove_dir_all(&self.fulltext_dir).map_err(|e| {
                    EngineError::Store(format!("failed to delete fulltext directory: {}", e))
                })?;
            }
            let new_ft = hippmem_store::fulltext::FulltextIndex::create(&self.fulltext_dir)
                .map_err(|e| {
                    EngineError::Store(format!("failed to rebuild Tantivy index: {}", e))
                })?;
            *self.fulltext_index.lock() = new_ft;
        }

        // 4. Clear the vector indexes
        {
            use hippmem_store::semantic::binary::BinaryCodeIndex;
            use hippmem_store::semantic::hnsw::FlatVectorIndex;
            *self.binary_code_index.lock() = BinaryCodeIndex::new();
            *self.dense_vector_index.lock() = FlatVectorIndex::new();
        }

        // 5. Re-write each entry with its original MemoryId
        for (id, unit) in &units {
            self.reindex_one(MemoryId(*id), unit)?;
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(ConsolidationReport {
            memories_processed: total,
            edges_decayed: 0,
            edges_archived: 0,
            edges_merged: 0,
            observation_promoted: 0,
            summaries_created: 0,
            contradictions_found: 0,
            reindexed: true,
            elapsed_ms,
        })
    }

    /// Re-processes a memory with its original MemoryId (used internally by Reindex).
    fn reindex_one(&self, id: MemoryId, unit: &MemoryUnit) -> EngineResult<()> {
        use crate::write_api::write_internal;

        let input = crate::WriteMemoryInput {
            content: unit.content.raw.clone(),
            content_type: Some(unit.content.content_type),
            context: unit.context.clone(),
            importance_hint: Some(unit.understanding.importance.value()),
            source_refs: unit.context.source_refs.clone(),
        };
        // skip_memory_log=true: the record already exists in MEMORY_LOG (constitution C7)
        write_internal(self, id, input, true, None)?;
        Ok(())
    }
}

// ── Table cleanup helpers ──

/// Clears all secondary tables, preserving MEMORY_LOG (constitution C7).
fn clear_all_secondary_tables(
    db: std::sync::Arc<redb::Database>,
) -> Result<(), hippmem_store::store::StoreError> {
    use redb::ReadableTable;

    let txn = db.begin_write()?;

    // u128 tables (excluding MEMORY_LOG)
    let u128_tables: &[redb::TableDefinition<u128, &[u8]>] = &[
        MEMORY_KV,
        LINK_OVERLAY,
        SUMMARY_OVERLAY,
        CORRECTION_OVERLAY,
        ACTIVATION_LOG,
        CONSOLIDATION_QUEUE,
    ];
    for def in u128_tables {
        let keys: Vec<u128> = {
            let table = txn.open_table(*def)?;
            table.iter()?.flatten().map(|(k, _)| k.value()).collect()
        };
        if !keys.is_empty() {
            let mut table = txn.open_table(*def)?;
            for k in &keys {
                let _ = table.remove(*k);
            }
        }
    }

    // u64 tables
    let u64_tables: &[redb::TableDefinition<u64, &[u8]>] = &[
        ENTITY_INDEX,
        TOPIC_INDEX,
        GOAL_INDEX,
        EVENT_INDEX,
        CAUSAL_INDEX,
    ];
    for def in u64_tables {
        let keys: Vec<u64> = {
            let table = txn.open_table(*def)?;
            table.iter()?.flatten().map(|(k, _)| k.value()).collect()
        };
        if !keys.is_empty() {
            let mut table = txn.open_table(*def)?;
            for k in &keys {
                let _ = table.remove(*k);
            }
        }
    }

    // u32 tables
    let u32_tables: &[redb::TableDefinition<u32, &[u8]>] = &[TEMPORAL_INDEX];
    for def in u32_tables {
        let keys: Vec<u32> = {
            let table = txn.open_table(*def)?;
            table.iter()?.flatten().map(|(k, _)| k.value()).collect()
        };
        if !keys.is_empty() {
            let mut table = txn.open_table(*def)?;
            for k in &keys {
                let _ = table.remove(*k);
            }
        }
    }

    txn.commit()?;
    Ok(())
}
