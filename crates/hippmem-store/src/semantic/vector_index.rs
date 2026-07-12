//! VectorIndex trait: vector index abstraction (ADR-003).
//!
//! Only the trait is exposed to the upper layer; concrete implementations are swappable.

/// Vector index: supports insertion, nearest-neighbor search, and deletion.
///
/// Corresponds to ADR-003: the default implementation is HNSW, but the trait isolation allows replacement.
pub trait VectorIndex: Send + Sync {
    /// Returns the number of vectors in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inserts a vector, associated with an id.
    fn insert(&mut self, id: u128, vector: &[f32]) -> Result<(), String>;

    /// Searches for the k nearest neighbors, returns a list of (id, squared L2 distance) sorted by distance ascending.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u128, f32)>, String>;

    /// Removes the entry with the given id.
    fn remove(&mut self, id: u128) -> Result<(), String>;
}

/// Binary-code index: Hamming-distance recall (ADR-004).
pub trait BinaryIndex: Send + Sync {
    /// Number of binary codes in the index.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inserts a binary code (byte array, one bit per dimension).
    fn insert(&mut self, id: u128, code: &[u8]) -> Result<(), String>;

    /// Searches for the k nearest neighbors (Hamming distance), returns a list of (id, distance) sorted by distance ascending.
    fn search(&self, query: &[u8], k: usize) -> Result<Vec<(u128, u32)>, String>;
}

/// Computes the squared L2 distance between two equal-length f32 slices.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}
