//! Engine::consolidate — consolidation API (05 §5, 09 §4.4).

use crate::signals::is_positive_signal;
use crate::{ConsolidationReport, ConsolidationScope, Engine, EngineError, EngineResult};
use hippmem_consolidation::hebbian::ActivationLog;
use hippmem_consolidation::summarize::{build_summary_unit, plan_summary_clusters};
use hippmem_consolidation::worker::ConsolidationWorker;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::unit::{MemoryLifecycle, MemoryUnit};
use hippmem_core::time::{Clock, SystemClock};
use hippmem_model::deterministic::summarize::DeterministicSummarizer;
use hippmem_store::activation_log::ActivationLogger;
use hippmem_store::kv::KvStore;
use hippmem_store::memory_log::MemoryLog;
use hippmem_store::store::{
    ACTIVATION_LOG, CAUSAL_INDEX, CONSOLIDATION_QUEUE, CORRECTION_OVERLAY, ENTITY_INDEX,
    EVENT_INDEX, GOAL_INDEX, LINK_OVERLAY, MEMORY_KV, SUMMARY_OVERLAY, TEMPORAL_INDEX, TOPIC_INDEX,
};
use std::collections::HashMap;
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
        //    0.3.0: only positive signals contribute (UserRejected excluded, 05 §6)
        //    E8 (0.4.0): records carry their signal weight (referenced 0.5 /
        //    confirmed 1.0 / succeeded 0.8), so edge reinforcement scales with
        //    signal strength per 03 §6.
        //    B4 (0.4.0): targeted rejects (non-empty used_memory_ids) feed the
        //    reverse-Hebbian step instead of strengthening anything.
        let logger = ActivationLogger::new(self.store.db_arc());
        let mut activation_log = ActivationLog::default();
        let mut rejected_ids: Vec<MemoryId> = Vec::new();
        if let Ok(records) = logger.read_all() {
            for rec in &records {
                if is_positive_signal(&rec.signal) {
                    let weight = match rec.signal.as_str() {
                        "Referenced" => 0.5,
                        "TaskSucceeded" => 0.8,
                        _ => 1.0, // UserConfirmedCorrect
                    };
                    for i in 0..rec.used_memory_ids.len() {
                        for j in (i + 1)..rec.used_memory_ids.len() {
                            // P1 回归：used_memory_ids 是完整 u128（MemoryId 原值），禁止截断
                            let a = MemoryId(rec.used_memory_ids[i]);
                            let b = MemoryId(rec.used_memory_ids[j]);
                            let ts = hippmem_core::time::Timestamp::from_millis(rec.recorded_at_ms);
                            activation_log.record(a, ts, weight);
                            activation_log.record(b, ts, weight);
                        }
                    }
                } else if rec.signal == "UserRejected" && !rec.used_memory_ids.is_empty() {
                    // Targeted reject: the named memories get their edges weakened.
                    rejected_ids.extend(rec.used_memory_ids.iter().map(|id| MemoryId(*id)));
                }
            }
        }
        let co_activations = activation_log.co_activation_pairs(3_600_000);

        // 3. Run consolidation cycle (Hebbian→reverse Hebbian→decay→compaction)
        let mut worker = ConsolidationWorker::default();
        let cycle_stats = worker.run_cycle(&mut units, &co_activations, &rejected_ids, now);

        // 3b. Summary planning (03 §8) — 由 Engine 层负责：按 simhash 相似簇触发，
        //     covers 去重，源单元标记 Compressed{into: summary.id}
        let params = self.params.read();
        let clusters = plan_summary_clusters(
            &units,
            params.summary_similarity_threshold,
            params.summary_trigger_count,
            params.summary_low_importance_threshold,
        );
        let mut summaries: Vec<MemoryUnit> = Vec::new();
        for cluster in &clusters {
            let members: Vec<MemoryUnit> = cluster
                .iter()
                .filter_map(|id| units.iter().find(|u| u.id == *id).cloned())
                .collect();
            if members.len() != cluster.len() {
                continue; // 防御：簇成员必须全部可解析
            }
            let summary_unit = build_summary_unit(&members, &DeterministicSummarizer);
            // Confidence gating: low confidence (<0.35) does not create a summary (Constitution C7)
            if summary_unit.understanding.confidence.value() >= 0.35 {
                summaries.push(summary_unit);
            }
        }
        // 源单元标记 Compressed（随下方持久化循环一起落库）
        for summary_unit in &summaries {
            for unit in units.iter_mut() {
                if summary_unit.context.preceding_memory_ids.contains(&unit.id) {
                    unit.lifecycle = MemoryLifecycle::Compressed {
                        into: summary_unit.id,
                    };
                }
            }
        }

        // B5 (0.4.0): redirect in-edges pointing at compressed sources to their
        // summary. Without this, other memories keep "ghost edges" to a source
        // that no longer expands (F3) — and the associations that used to flow
        // through the source are lost. After redirection the graph stays
        // connected and the summary becomes reachable as an upward view via
        // graph edges (the channel B1 reserved). The redirected edge keeps its
        // strength and type.
        let mut compressed_into: HashMap<MemoryId, MemoryId> = HashMap::new();
        for summary_unit in &summaries {
            for source_id in &summary_unit.context.preceding_memory_ids {
                compressed_into.insert(*source_id, summary_unit.id);
            }
        }
        if !compressed_into.is_empty() {
            for unit in units.iter_mut() {
                for link in unit.links.iter_mut() {
                    if let Some(&summary_id) = compressed_into.get(&link.target_id) {
                        link.target_id = summary_id;
                        match link.evidence.note.as_mut() {
                            Some(note) => note.push_str(" [redirected to summary]"),
                            None => link.evidence.note = Some("redirected to summary".into()),
                        }
                    }
                }
            }
        }

        // 4. Persist the modified units back to the store — MEMORY_KV (unit
        //    bodies) AND LINK_OVERLAY (graph edges). Retrieval reads edges from
        //    the graph table, so every edge mutation in this cycle (Hebbian
        //    reinforcement, reverse Hebbian, B5 redirection) must be synced
        //    there, otherwise the changes are invisible to retrieval.
        let kv = KvStore::new(self.store.db_arc());
        let graph = hippmem_store::graph::GraphStore::new(self.store.db_arc());
        for unit in &units {
            let bincode_unit = bincode::serde::encode_to_vec(unit, bincode::config::standard())
                .map_err(|e| EngineError::Internal(e.to_string()))?;
            kv.put(unit.id.0, &bincode_unit)
                .map_err(|e| EngineError::Store(e.to_string()))?;
            graph
                .put_outgoing(unit.id, &unit.links)
                .map_err(EngineError::Store)?;
        }

        // 4b. Persist summary memories — 全索引写入（P3 回归：摘要必须可被检索，
        //     且写入 memory_log，reindex 不丢失）
        for summary_unit in &summaries {
            let input = crate::WriteMemoryInput {
                content: summary_unit.content.raw.clone(),
                content_type: Some(summary_unit.content.content_type),
                context: summary_unit.context.clone(),
                importance_hint: Some(summary_unit.understanding.importance.value()),
                source_refs: summary_unit.context.source_refs.clone(),
            };
            crate::write_api::write_internal(self, summary_unit.id, input, false, None)?;

            // 保留摘要的身份与 covers 链：
            // write_internal 按相似度重建了普通边/元数据，这里以摘要单元自身的
            // Elaboration 出边 + provenance/stage/content.summary 覆盖写回的单元，
            // 保持图、单元、身份一致（索引仍用 write_internal 生成的键）。
            // B5: 摘要的 Elaboration 出边指向已压缩的源——从图中移除（源不可达，
            // 幽灵边不得留存）；covers 链保留在 context.preceding_memory_ids 供下钻。
            let graph = hippmem_store::graph::GraphStore::new(self.store.db_arc());
            let summary_links: Vec<hippmem_core::model::links::AssociationLink> = summary_unit
                .links
                .iter()
                .filter(|l| !compressed_into.contains_key(&l.target_id))
                .cloned()
                .collect();
            graph
                .put_outgoing(summary_unit.id, &summary_links)
                .map_err(EngineError::Store)?;
            if let Some(raw) = kv
                .get(&summary_unit.id.0)
                .map_err(|e| EngineError::Store(e.to_string()))?
            {
                let (mut patched, _): (MemoryUnit, _) =
                    bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                patched.links = summary_links.clone();
                patched.provenance = summary_unit.provenance.clone();
                patched.stage = summary_unit.stage;
                patched.content.summary = summary_unit.content.summary.clone();
                let re_bincode =
                    bincode::serde::encode_to_vec(&patched, bincode::config::standard())
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                kv.put(summary_unit.id.0, &re_bincode)
                    .map_err(|e| EngineError::Store(e.to_string()))?;
            }
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(ConsolidationReport {
            memories_processed: units.len() as u64 + summaries.len() as u64,
            edges_decayed: cycle_stats.edges_decayed,
            edges_archived: cycle_stats.edges_archived,
            edges_merged: cycle_stats.hebbian_applied,
            observation_promoted: 0,
            summaries_created: summaries.len() as u64,
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
