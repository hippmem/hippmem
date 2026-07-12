//! Binary-code index: Hamming-distance recall (ADR-004).
//!
//! Corresponds to the binary-code part of `SemanticSignature`,
//! using Hamming distance for approximate semantic recall (faster than full float comparison).

use super::vector_index::BinaryIndex;

/// Binary-code index: stores (id, Vec<u8>), brute-force Hamming search.
pub struct BinaryCodeIndex {
    entries: Vec<(u128, Vec<u8>)>,
}

impl BinaryCodeIndex {
    pub fn new() -> Self {
        Self { entries: vec![] }
    }
}

impl Default for BinaryCodeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryIndex for BinaryCodeIndex {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn insert(&mut self, id: u128, code: &[u8]) -> Result<(), String> {
        // Only accept equal-length codes or the first insertion
        if !self.entries.is_empty() && code.len() != self.entries[0].1.len() {
            return Err(format!(
                "Code length {} != existing {} bits",
                code.len() * 8,
                self.entries[0].1.len() * 8
            ));
        }
        if let Some(pos) = self.entries.iter().position(|(eid, _)| *eid == id) {
            self.entries[pos] = (id, code.to_vec());
        } else {
            self.entries.push((id, code.to_vec()));
        }
        Ok(())
    }

    fn search(&self, query: &[u8], k: usize) -> Result<Vec<(u128, u32)>, String> {
        if self.entries.is_empty() || k == 0 {
            return Ok(vec![]);
        }
        let mut scored: Vec<(u128, u32)> = self
            .entries
            .iter()
            .map(|(id, code)| (*id, hamming(query, code)))
            .collect();
        scored.sort_by_key(|(_, d)| *d);
        let limit = k.min(scored.len());
        Ok(scored[..limit].to_vec())
    }
}

/// Computes the Hamming distance (number of differing bits) between two equal-length byte arrays.
fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_code_zero_distance() {
        let code = vec![0b10101010u8, 0b01010101u8];
        assert_eq!(hamming(&code, &code), 0);
    }

    #[test]
    fn different_codes() {
        let a = vec![0b11111111u8];
        let b = vec![0b00000000u8];
        assert_eq!(hamming(&a, &b), 8);
    }

    #[test]
    fn binary_search_works() {
        let mut idx = BinaryCodeIndex::new();
        idx.insert(1, &[0b11110000]).unwrap();
        idx.insert(2, &[0b00001111]).unwrap();
        idx.insert(3, &[0b11111111]).unwrap();

        let r = idx.search(&[0b11110000], 2).unwrap();
        assert_eq!(r[0].0, 1); // exact match
        assert_eq!(r[0].1, 0);
    }
}
