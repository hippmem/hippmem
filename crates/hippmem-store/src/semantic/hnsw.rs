//! HNSW vector index implementation (ADR-003).
//!
//! Simplified implementation: single-layer NSW graph, brute-force nearest-neighbor search.
//! Used for deterministic fallback environments; can be replaced with a multi-layer HNSW later.

use super::vector_index::{l2_squared, VectorIndex};

/// Single-layer NSW vector index: brute-force search, in-memory storage.
///
/// Sufficient for small-scale (<100k) vector scenarios; can be replaced with `hnsw_rs` or a custom multi-layer implementation later.
pub struct FlatVectorIndex {
    /// (id, vector) list.
    entries: Vec<(u128, Vec<f32>)>,
}

impl FlatVectorIndex {
    /// Creates an empty index.
    pub fn new() -> Self {
        Self { entries: vec![] }
    }
}

impl Default for FlatVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for FlatVectorIndex {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn insert(&mut self, id: u128, vector: &[f32]) -> Result<(), String> {
        // Replace an existing entry with the same id
        if let Some(pos) = self.entries.iter().position(|(eid, _)| *eid == id) {
            self.entries[pos] = (id, vector.to_vec());
        } else {
            self.entries.push((id, vector.to_vec()));
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u128, f32)>, String> {
        if self.entries.is_empty() || k == 0 {
            return Ok(vec![]);
        }
        let mut scored: Vec<(u128, f32)> = self
            .entries
            .iter()
            .map(|(id, vec)| (*id, l2_squared(query, vec)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let limit = k.min(scored.len());
        Ok(scored[..limit].to_vec())
    }

    fn remove(&mut self, id: u128) -> Result<(), String> {
        self.entries.retain(|(eid, _)| *eid != id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search() {
        let mut idx = FlatVectorIndex::new();
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.insert(2, &[0.0, 1.0]).unwrap();
        idx.insert(3, &[0.5, 0.5]).unwrap();

        let results = idx.search(&[1.0, 0.1], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // [1,0] is the nearest
    }

    #[test]
    fn remove_entry() {
        let mut idx = FlatVectorIndex::new();
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.remove(1).unwrap();
        assert_eq!(idx.len(), 0);
    }
}
