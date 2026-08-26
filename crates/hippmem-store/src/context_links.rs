//! Context-answer links (memory-learning-mechanism, 0.4.3).
//!
//! A confirmation binds the confirmed memory to the query's context
//! fingerprint (entity/topic hashes). Retrieval lifts a candidate when the
//! current query's fingerprint intersects the links recorded for it —
//! exact set intersection, not fuzzy retrieval (the query is never stored
//! as a memory, and no similarity search is involved).
//!
//! Tables:
//! - `QUERY_CONTEXT`: (RetrievalId) -> QueryContext, written at retrieve
//!   time so a later feedback can recover what the query was about;
//! - `CONTEXT_LINKS`: (FeatureHash) -> Vec<ContextLink>, confirmation
//!   strengthens the link between each query feature and the memory.

use crate::store::{CONTEXT_LINKS, QUERY_CONTEXT, RETRIEVAL_PATHS};
use redb::{Database, ReadableDatabase, ReadableTable};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Query context fingerprint: stable hashes of the query's entities and
/// topics (already extracted by query understanding at retrieve time).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct QueryContext {
    pub entity_hashes: Vec<u64>,
    pub topic_hashes: Vec<u64>,
}

impl QueryContext {
    pub fn is_empty(&self) -> bool {
        self.entity_hashes.is_empty() && self.topic_hashes.is_empty()
    }

    /// All feature hashes, deduplicated and sorted (deterministic order).
    pub fn feature_hashes(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self
            .entity_hashes
            .iter()
            .chain(self.topic_hashes.iter())
            .copied()
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// A context link entry: a memory that was confirmed in queries carrying
/// this feature, and how strongly. Review bookkeeping (forgetting-curve
/// model, 0.4.3): each confirmation is a review — it resets the decay clock
/// and doubles the link's half-life, so spaced reviews consolidate the link
/// into long-term memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextLink {
    pub memory_id: u128,
    pub strength: f32,
    pub review_count: u32,
    pub last_reviewed_at_ms: i64,
}

/// Base half-life for a context link (1 day); each review doubles it, capped
/// at this many doublings (up to 64 days).
pub const LINK_HALF_LIFE_BASE_MS: i64 = 86_400_000;
pub const LINK_HALF_LIFE_MAX_DOUBLINGS: u32 = 6;

/// Effective (decay-adjusted) strength of a context link at `now_ms`,
/// following the forgetting curve: strength × 0.5^(Δt / half_life).
pub fn effective_strength(link: &ContextLink, now_ms: i64) -> f32 {
    let inactive = (now_ms - link.last_reviewed_at_ms).max(0) as f64;
    let doublings = link.review_count.min(LINK_HALF_LIFE_MAX_DOUBLINGS);
    let half_life = (LINK_HALF_LIFE_BASE_MS as f64) * 2f64.powi(doublings as i32);
    (link.strength as f64 * 0.5f64.powf(inactive / half_life)) as f32
}

/// Strength cap for a single context link (bounds the retrieval boost).
pub const LINK_STRENGTH_CAP: f32 = 1.0;

/// Writes the query fingerprint for a retrieval (retrieve time).
pub fn write_query_context(
    db: Arc<Database>,
    retrieval_id: u64,
    ctx: &QueryContext,
) -> Result<(), String> {
    let encoded = bincode::serde::encode_to_vec(ctx, bincode::config::standard())
        .map_err(|e| e.to_string())?;
    let txn = db.begin_write().map_err(|e| e.to_string())?;
    {
        let mut table = txn.open_table(QUERY_CONTEXT).map_err(|e| e.to_string())?;
        table
            .insert(retrieval_id, encoded.as_slice())
            .map_err(|e| e.to_string())?;
    }
    txn.commit().map_err(|e| e.to_string())
}

/// Reads the query fingerprint of a retrieval (feedback time). Old retrievals
/// (before 0.4.3) have no entry — returns None, and feedback degrades to the
/// non-context path (no context links are written).
pub fn read_query_context(db: Arc<Database>, retrieval_id: u64) -> Option<QueryContext> {
    let txn = db.begin_read().ok()?;
    let table = txn.open_table(QUERY_CONTEXT).ok()?;
    let value = table.get(retrieval_id).ok()??;
    bincode::serde::decode_from_slice::<QueryContext, _>(value.value(), bincode::config::standard())
        .ok()
        .map(|(ctx, _)| ctx)
}

/// Strengthens the context links between every feature of the query
/// fingerprint and the confirmed memory (feedback time). Empty fingerprints
/// (no entities/topics) are a no-op — nothing to bind.
pub fn strengthen_links(
    db: Arc<Database>,
    ctx: &QueryContext,
    memory_id: u128,
    delta: f32,
    now_ms: i64,
) -> Result<(), String> {
    if ctx.is_empty() {
        return Ok(());
    }
    let txn = db.begin_write().map_err(|e| e.to_string())?;
    for feature in ctx.feature_hashes() {
        let mut table = txn.open_table(CONTEXT_LINKS).map_err(|e| e.to_string())?;
        let mut links: Vec<ContextLink> = table
            .get(feature)
            .map_err(|e| e.to_string())?
            .map(|v| {
                bincode::serde::decode_from_slice::<Vec<ContextLink>, _>(
                    v.value(),
                    bincode::config::standard(),
                )
                .map(|(l, _)| l)
                .unwrap_or_default()
            })
            .unwrap_or_default();
        match links.iter_mut().find(|l| l.memory_id == memory_id) {
            Some(l) => {
                l.strength = (l.strength + delta).min(LINK_STRENGTH_CAP);
                l.review_count = l.review_count.saturating_add(1);
                l.last_reviewed_at_ms = now_ms;
            }
            None => links.push(ContextLink {
                memory_id,
                strength: delta.min(LINK_STRENGTH_CAP),
                review_count: 1,
                last_reviewed_at_ms: now_ms,
            }),
        }
        let encoded = bincode::serde::encode_to_vec(&links, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        table
            .insert(feature, encoded.as_slice())
            .map_err(|e| e.to_string())?;
    }
    txn.commit().map_err(|e| e.to_string())
}

/// Collects context-link strengths for candidate memories, keyed by the
/// current query's feature intersection. Returns (memory_id -> strength),
/// taking the max strength across intersecting features (conservative —
/// multiple features do not stack).
pub fn collect_link_strengths(
    db: Arc<Database>,
    ctx: &QueryContext,
    now_ms: i64,
) -> HashMap<u128, f32> {
    let mut out: HashMap<u128, f32> = HashMap::new();
    if ctx.is_empty() {
        return out;
    }
    let Ok(txn) = db.begin_read() else {
        return out;
    };
    for feature in ctx.feature_hashes() {
        let Ok(table) = txn.open_table(CONTEXT_LINKS) else {
            continue;
        };
        if let Ok(Some(value)) = table.get(feature) {
            if let Ok((links, _)) = bincode::serde::decode_from_slice::<Vec<ContextLink>, _>(
                value.value(),
                bincode::config::standard(),
            ) {
                for link in links {
                    let eff = effective_strength(&link, now_ms);
                    out.entry(link.memory_id)
                        .and_modify(|s| *s = (*s).max(eff))
                        .or_insert(eff);
                }
            }
        }
    }
    out
}

/// Propagation path record: (guide memory, answer memory) — the edge that
/// carried activation from the guide to the answer during retrieval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalPath {
    pub from: u128,
    pub to: u128,
}

/// Writes the propagation paths of a retrieval (retrieve time).
pub fn write_retrieval_paths(
    db: Arc<Database>,
    retrieval_id: u64,
    paths: &[RetrievalPath],
) -> Result<(), String> {
    if paths.is_empty() {
        return Ok(());
    }
    let encoded = bincode::serde::encode_to_vec(paths, bincode::config::standard())
        .map_err(|e| e.to_string())?;
    let txn = db.begin_write().map_err(|e| e.to_string())?;
    {
        let mut table = txn.open_table(RETRIEVAL_PATHS).map_err(|e| e.to_string())?;
        table
            .insert(retrieval_id, encoded.as_slice())
            .map_err(|e| e.to_string())?;
    }
    txn.commit().map_err(|e| e.to_string())
}

/// Reads the propagation paths of a retrieval (feedback time).
pub fn read_retrieval_paths(db: Arc<Database>, retrieval_id: u64) -> Vec<RetrievalPath> {
    let Ok(txn) = db.begin_read() else {
        return vec![];
    };
    let Ok(table) = txn.open_table(RETRIEVAL_PATHS) else {
        return vec![];
    };
    let Ok(Some(value)) = table.get(retrieval_id) else {
        return vec![];
    };
    bincode::serde::decode_from_slice::<Vec<RetrievalPath>, _>(
        value.value(),
        bincode::config::standard(),
    )
    .map(|(p, _)| p)
    .unwrap_or_default()
}

/// Removes every context link entry referencing the memory (M3,
/// memory-management): the memory is stripped from all feature posting
/// lists; empty lists are removed. Query fingerprints and retrieval paths
/// are audit records and stay.
pub fn remove_memory_links(db: Arc<Database>, memory_id: u128) -> Result<(), String> {
    let txn = db.begin_write().map_err(|e| e.to_string())?;
    // Pass 1: collect feature keys whose posting list mentions the memory.
    let affected: Vec<u64> = {
        let table = txn.open_table(CONTEXT_LINKS).map_err(|e| e.to_string())?;
        table
            .iter()
            .map_err(|e| e.to_string())?
            .flatten()
            .filter_map(|(k, v)| {
                if let Ok((links, _)) = bincode::serde::decode_from_slice::<Vec<ContextLink>, _>(
                    v.value(),
                    bincode::config::standard(),
                ) {
                    if links.iter().any(|l| l.memory_id == memory_id) {
                        return Some(k.value());
                    }
                }
                None
            })
            .collect()
    };
    // Pass 2: rewrite the affected posting lists (raw copied locally so the
    // table borrow ends before mutation).
    for feature in affected {
        let mut table = txn.open_table(CONTEXT_LINKS).map_err(|e| e.to_string())?;
        let raw = table
            .get(feature)
            .map_err(|e| e.to_string())?
            .map(|v| v.value().to_vec());
        if let Some(raw) = raw {
            if let Ok((mut links, _)) = bincode::serde::decode_from_slice::<Vec<ContextLink>, _>(
                raw.as_slice(),
                bincode::config::standard(),
            ) {
                links.retain(|l| l.memory_id != memory_id);
                if links.is_empty() {
                    table.remove(feature).map_err(|e| e.to_string())?;
                } else if let Ok(enc) =
                    bincode::serde::encode_to_vec(&links, bincode::config::standard())
                {
                    table
                        .insert(feature, enc.as_slice())
                        .map_err(|e| e.to_string())?;
                }
            }
        }
    }
    txn.commit().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400_000;

    fn link(review_count: u32, last_reviewed: i64) -> ContextLink {
        ContextLink {
            memory_id: 1,
            strength: 1.0,
            review_count,
            last_reviewed_at_ms: last_reviewed,
        }
    }

    #[test]
    fn link_fades_along_forgetting_curve_and_review_resets() {
        // One day without review → half strength (base half-life 1 day).
        let fresh = link(0, 0);
        assert!((effective_strength(&fresh, DAY) - 0.5).abs() < 0.01);

        // Five reviews double the half-life five times (32 days): after 32
        // days the reviewed link is still at half strength.
        let reviewed = link(5, 0);
        assert!((effective_strength(&reviewed, 32 * DAY) - 0.5).abs() < 0.01);

        // A recent review resets the clock: just-reviewed link is nearly full.
        assert!((effective_strength(&reviewed, 60_000) - 1.0).abs() < 0.001);
    }

    /// M3 (memory-management): the delete-cascade counterpart — a removed
    /// memory is stripped from every posting list; lists left empty are
    /// dropped entirely.
    #[test]
    fn remove_memory_links_strips_and_prunes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Arc::new(redb::Database::create(dir.path().join("ctx.redb")).expect("create db"));
        let txn = db.begin_write().expect("write txn");
        txn.open_table(CONTEXT_LINKS).expect("open table");
        txn.commit().expect("commit");

        // Features 11/22/33 bind both memories; feature 44 binds only memory 1.
        let shared = QueryContext {
            entity_hashes: vec![11, 22, 33],
            topic_hashes: vec![],
        };
        strengthen_links(db.clone(), &shared, 1, 0.8, 1_000).expect("link memory 1");
        strengthen_links(db.clone(), &shared, 2, 0.5, 1_000).expect("link memory 2");
        let solo = QueryContext {
            entity_hashes: vec![44],
            topic_hashes: vec![],
        };
        strengthen_links(db.clone(), &solo, 1, 0.7, 1_000).expect("link memory 1 solo");

        remove_memory_links(db.clone(), 1).expect("remove memory 1");

        let txn = db.begin_read().expect("read txn");
        let table = txn.open_table(CONTEXT_LINKS).expect("open table");
        for feature in [11u64, 22, 33] {
            let raw = table
                .get(feature)
                .expect("read")
                .expect("shared feature key must survive");
            let (links, _) = bincode::serde::decode_from_slice::<Vec<ContextLink>, _>(
                raw.value(),
                bincode::config::standard(),
            )
            .expect("decode");
            assert_eq!(links.len(), 1, "only memory 2 remains on feature {feature}");
            assert_eq!(links[0].memory_id, 2);
        }
        assert!(
            table.get(44u64).expect("read").is_none(),
            "feature bound only to the removed memory is pruned"
        );
    }
}
