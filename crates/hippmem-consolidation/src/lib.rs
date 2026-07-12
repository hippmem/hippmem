//! HIPPMEM · Consolidation engine
//!
//! Hebbian reinforcement + decay + compaction + summarization + background worker.
//! Corresponds to 03 §6-8.

pub mod compaction;
pub mod decay;
pub mod hebbian;
pub mod summarize;
pub mod worker;
