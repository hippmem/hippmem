//! HIPPMEM · Evaluation Framework
//!
//! Provides eval corpus loading, baseline execution, metric computation,
//! and threshold validation. Corresponds to 06-eval-framework.md.

pub mod baselines;
pub mod bench_corpus;
pub mod corpus;
pub mod fixture_loader;
pub mod metrics;
pub mod runner;
