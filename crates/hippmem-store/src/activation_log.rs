//! Activation log: records retrieval, usage, and co-activation events (03 §8, 05 §6).
//!
//! Persisted to the redb ACTIVATION_LOG table, consumed by Hebbian / decay logic.
//!
//! **MemoryId 完整性（P1 回归）**: used_memory_ids 必须是完整 u128。
//! 历史版本曾以 u64 截断落库（MemoryId 是 ULID，高位置位，截断后读回必然
//! 失配，导致 RecentActivation / Hebbian 全部指向幽灵 id）——`read_all`
//! 对旧格式做兼容解码。

use crate::store::ACTIVATION_LOG;
use redb::{Database, ReadableDatabase, ReadableTable};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A single activation record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActivationRecord {
    pub retrieval_id: u64,
    pub used_memory_ids: Vec<u128>,
    pub signal: String,
    pub recorded_at_ms: i64,
}

/// 历史格式（v0.2.0 及更早）：used_memory_ids 曾为 u64（截断丢失高位）。
/// 仅用于读取旧库时的兼容解码，新记录一律写 u128。
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyActivationRecord {
    retrieval_id: u64,
    used_memory_ids: Vec<u64>,
    signal: String,
    recorded_at_ms: i64,
}

/// Decodes a stored record, accepting both the current (u128) and legacy (u64) formats.
fn decode_record(data: &[u8]) -> Option<ActivationRecord> {
    if let Ok((rec, _)) =
        bincode::serde::decode_from_slice::<ActivationRecord, _>(data, bincode::config::standard())
    {
        return Some(rec);
    }
    bincode::serde::decode_from_slice::<LegacyActivationRecord, _>(
        data,
        bincode::config::standard(),
    )
    .ok()
    .map(|(legacy, _)| ActivationRecord {
        retrieval_id: legacy.retrieval_id,
        // 旧库的 u64 是截断值，已无法还原 ULID；保留为 u64 值（零扩展）以兼容读取
        used_memory_ids: legacy.used_memory_ids.into_iter().map(u128::from).collect(),
        signal: legacy.signal,
        recorded_at_ms: legacy.recorded_at_ms,
    })
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

    /// Reads all records (new u128 format + legacy u64 format).
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
            if let Some(rec) = decode_record(value.value()) {
                recs.push(rec);
            }
        }
        Ok(recs)
    }
}
