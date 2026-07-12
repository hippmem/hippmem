//! HIPPMEM · Storage and indexing layer
//!
//! Provides the `Store` trait and its redb implementation, managing:
//! - Memory log (`memory_log`) and KV (`memory_kv`)
//! - Inverted indexes (entity/topic/goal/event/temporal)
//! - Association graph, summary, correction overlay
//! - Activation log and consolidation queue
//!
//! Corresponds to 04 §5 storage layout, ADR-001 (redb).

pub mod activation_log;
pub mod fulltext;
pub mod graph;
pub mod kv;
pub mod memory_log;
pub mod semantic;
pub mod store;
