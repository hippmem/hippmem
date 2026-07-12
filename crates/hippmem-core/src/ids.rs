//! Core ID and key types (ADR-006/007, corresponds to 02#0).
//!
//! Contains MemoryId, VectorId, and key type aliases for each dimension.

use serde::{Deserialize, Serialize};

/// Unique memory ID. Generated as a ULID, carried as u128. Lexicographic order = approximate time order. See ADR-006.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemoryId(pub u128);

impl MemoryId {
    /// Generates a new MemoryId (ULID).
    /// Determinism: the time and random parts come from an injectable Clock/Rng; here the system default impl is used.
    pub fn generate() -> Self {
        let ulid = ulid::Ulid::new();
        Self(ulid.into())
    }

    /// Extracts the inner u128 value.
    pub fn as_u128(&self) -> u128 {
        self.0
    }
}

/// Reference handle of a dense vector within the semantic_index (not the vector itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId(pub u128);

impl VectorId {
    /// Extracts the inner u128 value.
    pub fn as_u128(&self) -> u128 {
        self.0
    }
}

// ── Key type aliases ──

/// Entity key: the canonical entity name hashed to u64 (xxh3, fixed seed 0, ADR-019).
pub type EntityKey = u64;

/// Topic key: the topic text hashed to u64.
pub type TopicKey = u64;

/// Goal key: the goal description hashed to u64.
pub type GoalKey = u64;

/// Event key: the event identifier hashed to u64.
pub type EventKey = u64;

/// Causal key: the causal claim hashed to u64.
pub type CausalKey = u64;

/// Emotion key: emotion category enum → u8.
pub type EmotionKey = u8;

/// Temporal key: time bucket → u32.
pub type TemporalKey = u32;
