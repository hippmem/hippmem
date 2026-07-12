//! Engine::explain — explain API (05 §4, 09 §4.3).

use crate::{Engine, EngineError, EngineResult, Explanation, LinkSummary};
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::LinkType;

impl Engine {
    /// Explains a memory: source/importance/connections/corrections/recent activations.
    pub fn explain(
        &self,
        id: MemoryId,
        _ctx: Option<crate::RetrieveContext>,
    ) -> EngineResult<Explanation> {
        let units = crate::retrieve_api::load_all_units(self.store.db_arc());
        let unit = units
            .iter()
            .find(|u| u.id == id)
            .cloned()
            .ok_or(EngineError::NotFound(id))?;

        let linked: Vec<LinkSummary> = unit
            .links
            .iter()
            .map(|l| LinkSummary {
                target: l.target_id,
                link_type: l.link_type,
                strength: l.strength.value(),
            })
            .collect();

        let corrections: Vec<MemoryId> = unit
            .links
            .iter()
            .filter(|l| l.link_type == LinkType::Correction)
            .map(|l| l.target_id)
            .collect();

        let contradictions: Vec<MemoryId> = unit
            .links
            .iter()
            .filter(|l| l.link_type == LinkType::Contradiction)
            .map(|l| l.target_id)
            .collect();

        Ok(Explanation {
            memory_id: unit.id,
            content_summary: unit.content.raw.chars().take(120).collect(),
            current_importance: unit.understanding.importance.value(),
            linked,
            corrections,
            contradictions,
            recent_activations: unit.activation.retrieval_count,
        })
    }
}
