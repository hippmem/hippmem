//! HIPPMEM · Core domain types and infrastructure
//!
//! This crate is the foundation of HIPPMEM: it defines the core data models
//! such as MemoryUnit, AssociationLink, and ActivationState, as well as global
//! infrastructure like the Clock/Rng traits and base error types.
//!
//! ## Crate role (constitution C2)
//! - Does not depend on any other hippmem crate
//! - Does not depend on any external storage / network / model backend
//! - The public API exposes only memory-semantic types, never underlying library types
//!
//! ## Dependency direction
//! `hippmem-core` ← all other hippmem crates

// Modules will be added incrementally in subsequent tasks
pub mod config;
pub mod error;
pub mod hash;
pub mod ids;
pub mod model;
pub mod rng;
pub mod score;
pub mod time;
