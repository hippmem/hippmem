//! HIPPMEM · Write engine
//!
//! Responsible for AssociationKeys generation, multi-dimensional candidate
//! discovery, association scoring, and edge construction.
//! Corresponds to algorithm parameters in 03.

pub mod candidates;
pub mod edges;
pub mod enrich;
pub mod explain;
pub mod keys;
pub mod scoring;
pub mod staged;
pub mod understanding;
