//! acceptance test: VectorIndex + binary-code recall
//!
//! Locale-specific test data lives in `tests/fixtures/semantic/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_model::deterministic::embed::DeterministicEmbedder;
use hippmem_store::semantic::binary::BinaryCodeIndex;
use hippmem_store::semantic::hnsw::FlatVectorIndex;
use hippmem_store::semantic::vector_index::{BinaryIndex, VectorIndex};
use std::fs;

/// Discover available locale fixtures.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/semantic", env!("CARGO_MANIFEST_DIR"));
    let mut locales = vec![];
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".json") {
                locales.push(name.trim_end_matches(".json").to_string());
            }
        }
    }
    locales.sort();
    if locales.is_empty() {
        panic!("no locale fixtures found in semantic/");
    }
    locales
}

/// Load semantic fixture for a specific locale.
fn load_fixture(locale: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/semantic/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = fs::read_to_string(&path).expect("failed to read fixture");
    serde_json::from_str(&data).expect("invalid fixture")
}

/// After inserting embedder vectors into the index, nearest-neighbor recall works. All locales.
#[test]
fn vector_index_nn_recall() {
    for locale in discover_fixture_locales() {
        let embedder = DeterministicEmbedder::default();
        let fixture = load_fixture(&locale);
        let texts: Vec<String> = fixture["documents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let vectors = embedder.embed_sync(&texts).unwrap();

        let mut idx = FlatVectorIndex::new();
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(i as u128, v).unwrap();
        }

        // Querying the same content as text[0] should recall text[0] and text[2] (semantically close)
        let query_texts = vec![fixture["queries"][0].as_str().unwrap().to_string()];
        let qv = embedder.embed_sync(&query_texts).unwrap();
        let results = idx.search(&qv[0], 3).unwrap();

        // text[0] (Rust language) and text[2] (Rust memory safety) should be in the top-3
        let ids: Vec<u128> = results.iter().map(|r| r.0).collect();
        assert!(
            ids.contains(&0u128),
            "[{locale}] the Rust-language entry should be in the top-3"
        );
        assert!(
            ids.contains(&2u128),
            "[{locale}] the Rust memory-safety entry should be in the top-3"
        );

        // Distances should be non-decreasing
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    } // end locale loop
}

/// Binary-code Hamming-distance recall works.
#[test]
fn binary_code_hamming_recall() {
    let mut idx = BinaryCodeIndex::new();

    // Insert 4 32-bit codes
    idx.insert(10, &[0b11110000]).unwrap();
    idx.insert(20, &[0b11111111]).unwrap();
    idx.insert(30, &[0b00001111]).unwrap();
    idx.insert(40, &[0b00000000]).unwrap();

    // Query codes close to 0b11110000
    let results = idx.search(&[0b11110000], 2).unwrap();
    assert_eq!(results[0].0, 10, "exact match should rank first");
    assert_eq!(results[0].1, 0);

    // The second should be 0b11111111 (distance=4) or another
    assert!(results[1].1 > 0);
}

/// An empty index returns empty results.
#[test]
fn empty_index_returns_empty() {
    let idx = FlatVectorIndex::new();
    assert!(idx.search(&[1.0, 0.0], 5).unwrap().is_empty());

    let bidx = BinaryCodeIndex::new();
    assert!(bidx.search(&[0u8], 5).unwrap().is_empty());
}
