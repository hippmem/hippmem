//! Retrieval result risk warnings: contradiction/correction/supersession/low-confidence/stale (03 §5).
//!
//! Explicitly annotates potential issues in retrieval results, satisfying constitution C4 (explainability).

use hippmem_core::model::links::{LinkType, MemoryWarning};
use hippmem_core::model::unit::{MemoryLifecycle, MemoryUnit};

/// Generate the list of risk warnings for a retrieval result.
pub fn check_warnings(unit: &MemoryUnit, energy: f32) -> Vec<MemoryWarning> {
    let mut w = Vec::new();

    // Low energy → low confidence
    if energy < 0.20 {
        w.push(MemoryWarning::LowConfidence);
    }

    // Deprecated
    if unit.lifecycle == MemoryLifecycle::Deprecated {
        w.push(MemoryWarning::Deprecated);
    }

    // Check outgoing edges for correction/contradiction/supersession
    for link in &unit.links {
        match link.link_type {
            LinkType::Correction => {
                w.push(MemoryWarning::HasCorrection { by: link.target_id });
            }
            LinkType::Contradiction => {
                w.push(MemoryWarning::HasContradiction {
                    with: link.target_id,
                });
            }
            LinkType::Supersedes => {
                w.push(MemoryWarning::Superseded { by: link.target_id });
            }
            _ => {}
        }
    }

    // Freshness: warn when energy is extremely low
    if energy < 0.10 {
        w.push(MemoryWarning::StaleFreshness);
    }

    w
}
