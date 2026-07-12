//! Engine background runtime: enrich worker (09 §3).
//!
//! Current implementation is synchronous: enrich is executed immediately after
//! write to complete strong semantic dimensions.
//! Can later be upgraded to a tokio async background worker.

use hippmem_core::model::unit::{MemoryStage, MemoryUnit};
use hippmem_write::enrich::{enrich_unit, EnrichInput};

/// Synchronously run enrich on a memory at the Indexed stage:
/// completes goals/preferences/emotions/decisions (deterministic rules).
pub fn run_enrich_sync(unit: &mut MemoryUnit) {
    if unit.stage != MemoryStage::Indexed {
        return;
    }

    let input = EnrichInput { unit: unit.clone() };
    let output = enrich_unit(input);
    *unit = output.unit;
    unit.stage = MemoryStage::Enriched;
}
