//! Engine::traverse — BFS graph traversal API.

use crate::{EngineResult, TraverseDirection, TraverseInput, TraverseNode, TraverseOutput};
use hippmem_core::ids::MemoryId;
use std::collections::{HashMap, HashSet, VecDeque};

impl crate::Engine {
    /// BFS graph traversal: starts from start_id and explores the memory network along association edges.
    pub fn traverse(&self, input: TraverseInput) -> EngineResult<TraverseOutput> {
        let max_depth = input.max_depth.clamp(1, 5);

        let units = crate::retrieve_api::load_all_units(self.store.db_arc());
        let unit_map: HashMap<MemoryId, &hippmem_core::model::unit::MemoryUnit> =
            units.iter().map(|u| (u.id, u)).collect();

        // Validate that the start node exists
        if !unit_map.contains_key(&input.start_id) {
            return Err(crate::EngineError::NotFound(input.start_id));
        }

        let mut visited: HashSet<MemoryId> = HashSet::new();
        let mut nodes: Vec<TraverseNode> = Vec::new();
        let mut edges: Vec<crate::EdgeView> = Vec::new();
        let mut queue: VecDeque<(MemoryId, u8)> = VecDeque::new();

        visited.insert(input.start_id);
        queue.push_back((input.start_id, 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let current_unit = match unit_map.get(&current_id) {
                Some(u) => *u,
                None => continue,
            };

            // Collect candidate edges starting from current
            let candidate_edges: Vec<(MemoryId, &hippmem_core::model::links::AssociationLink)> =
                match input.direction {
                    TraverseDirection::Outgoing => current_unit
                        .links
                        .iter()
                        .map(|l| (l.target_id, l))
                        .collect(),
                    TraverseDirection::Incoming => units
                        .iter()
                        .flat_map(|other| {
                            other.links.iter().filter_map(move |l| {
                                if l.target_id == current_id {
                                    Some((other.id, l))
                                } else {
                                    None
                                }
                            })
                        })
                        .collect(),
                    TraverseDirection::Both => {
                        let mut both: Vec<(
                            MemoryId,
                            &hippmem_core::model::links::AssociationLink,
                        )> = current_unit
                            .links
                            .iter()
                            .map(|l| (l.target_id, l))
                            .collect();
                        // In-edges
                        for other in &units {
                            for l in &other.links {
                                if l.target_id == current_id {
                                    both.push((other.id, l));
                                }
                            }
                        }
                        both
                    }
                };

            for (neighbor_id, link) in &candidate_edges {
                // Filter by link_types (API-layer filter, not exposed in CLI)
                if let Some(ref types) = input.link_types {
                    if !types.contains(&link.link_type) {
                        continue;
                    }
                }

                // Record edge
                edges.push(crate::EdgeView {
                    from: current_id,
                    to: *neighbor_id,
                    link_type: link.link_type,
                    strength: link.strength.value(),
                    confidence: link.confidence.value(),
                    activation_count: link.activation_count,
                    evidence: link.evidence.note.clone().unwrap_or_default(),
                });

                // If not visited, enqueue and record node
                if visited.insert(*neighbor_id) {
                    if let Some(nu) = unit_map.get(neighbor_id) {
                        nodes.push(TraverseNode {
                            id: *neighbor_id,
                            depth: depth + 1,
                            content_preview: nu.content.raw.chars().take(100).collect(),
                            content_type: nu.content.content_type,
                            importance: nu.understanding.importance.value(),
                        });
                    }
                    queue.push_back((*neighbor_id, depth + 1));
                }
            }
        }

        Ok(TraverseOutput { nodes, edges })
    }
}
