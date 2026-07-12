//! HIPPMEM · Retrieval engine
//!
//! Multi-channel seed recall + spreading activation + rerank + explanation path.
//! Corresponds to 03 §4, 05 §3.

pub mod energy;
pub mod explain;
pub mod rerank;
pub mod seeds;
pub mod spreading;
pub mod warnings;
