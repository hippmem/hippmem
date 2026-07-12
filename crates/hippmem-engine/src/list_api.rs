//! Engine::list — paginated memory listing API.

use crate::{Engine, EngineResult, ListInput, ListItem, ListOutput};

impl Engine {
    /// Lists memories with pagination, fixed to NewestFirst (MemoryId descending) order.
    ///
    /// Reads all memories from the MEMORY_KV table, filters by content_type,
    /// then returns a page.
    pub fn list(&self, input: ListInput) -> EngineResult<ListOutput> {
        let limit = input.limit.clamp(1, 100);

        // Load all MemoryUnit entries
        let units = crate::retrieve_api::load_all_units(self.store.db_arc());

        // Build ListItem and filter
        let mut items: Vec<ListItem> = units
            .iter()
            .filter(|u| {
                if let Some(ref ct) = input.content_type {
                    u.content.content_type == *ct
                } else {
                    true
                }
            })
            .map(|u| ListItem {
                id: u.id,
                content_preview: u.content.raw.chars().take(100).collect(),
                content_type: u.content.content_type,
                created_at: u.created_at,
                importance: u.understanding.importance.value(),
                stage: u.stage,
                lifecycle: u.lifecycle.clone(),
                edge_count: u.links.len(),
            })
            .collect();

        // Sort by MemoryId descending (NewestFirst)
        items.sort_by_key(|item| std::cmp::Reverse(item.id));

        let total = items.len() as u64;

        // Cursor pagination
        let start_idx = if let Some(cursor) = input.cursor {
            // Find the position of the cursor ID in the sorted list, start after it
            items
                .iter()
                .position(|item| item.id.0 == cursor)
                .map(|p| p + 1) // Start from the item after the cursor
                .unwrap_or_else(|| {
                    // Cursor does not exist (may have been deleted); use binary search for the insertion point
                    items
                        .binary_search_by(|item| {
                            // Descending order, so reverse comparison direction
                            cursor.cmp(&item.id.0)
                        })
                        .unwrap_or_else(|insert_point| insert_point)
                })
        } else {
            0
        };

        let end_idx = (start_idx + limit).min(items.len());
        let has_more = end_idx < items.len();
        let page: Vec<ListItem> = if start_idx < items.len() {
            items.drain(start_idx..end_idx).collect()
        } else {
            Vec::new()
        };

        let next_cursor = if has_more {
            page.last().map(|item| item.id.0)
        } else {
            None
        };

        Ok(ListOutput {
            items: page,
            next_cursor,
            total,
        })
    }
}
