//! acceptance test: DeterministicEmbedder
//!
//! Locale-specific test texts live in `tests/fixtures/embed/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_model::deterministic::embed::DeterministicEmbedder;
use hippmem_model::traits::Embedder;
use std::fs;

/// Discover available locale fixtures.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/embed", env!("CARGO_MANIFEST_DIR"));
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
        panic!("no locale fixtures found in embed/");
    }
    locales
}

/// Load embed fixture for a specific locale.
fn load_fixture(locale: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/embed/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = fs::read_to_string(&path).expect("failed to read fixture");
    serde_json::from_str(&data).expect("invalid fixture")
}

/// The same text always yields the same vector. Runs for all available locales.
#[test]
fn same_text_same_vector() {
    for locale in discover_fixture_locales() {
        let e = DeterministicEmbedder::default();
        let fixture = load_fixture(&locale);
        let text = fixture["embed_texts"][0].as_str().unwrap().to_string();
        let texts = vec![text];
        let v1 = e.embed_sync(&texts).unwrap();
        let v2 = e.embed_sync(&texts).unwrap();
        assert_eq!(v1[0].len(), 256);
        for (a, b) in v1[0].iter().zip(v2[0].iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "[{locale}] vectors for the same text should be identical"
            );
        }
    }
}

/// Different texts produce different vectors. Runs for all available locales.
#[test]
fn different_texts_different_vectors() {
    for locale in discover_fixture_locales() {
        let e = DeterministicEmbedder::default();
        let fixture = load_fixture(&locale);
        let text_a = fixture["embed_texts"][1].as_str().unwrap();
        let text_b = fixture["embed_texts"][2].as_str().unwrap();
        let v_a = e.embed_sync(&[text_a.to_string()]).unwrap();
        let v_b = e.embed_sync(&[text_b.to_string()]).unwrap();
        let same = v_a[0]
            .iter()
            .zip(v_b[0].iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !same,
            "[{locale}] different texts should produce different vectors"
        );
    }
}

/// dim() == 256.
#[test]
fn default_dim_is_256() {
    let e = DeterministicEmbedder::default();
    assert_eq!(e.dim(), 256);
}

/// L2 norm is close to 1.0 (normalized). Runs for all available locales.
#[test]
fn embeddings_are_normalized() {
    for locale in discover_fixture_locales() {
        let e = DeterministicEmbedder::default();
        let fixture = load_fixture(&locale);
        let texts: Vec<String> = fixture["batch_texts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let vecs = e.embed_sync(&texts).unwrap();
        for v in &vecs {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "[{locale}] L2 norm {norm} should be close to 1.0"
            );
        }
    }
}

/// Empty text returns the zero vector? Returns a reasonable vector (does not panic).
#[test]
fn empty_text_does_not_panic() {
    let e = DeterministicEmbedder::default();
    let vecs = e.embed_sync(&["".to_string()]).unwrap();
    assert_eq!(vecs[0].len(), 256);
}

/// backend_id is correct.
#[test]
fn backend_id_is_deterministic_hash() {
    let e = DeterministicEmbedder::default();
    assert_eq!(e.backend_id(), "deterministic-hash");
}
