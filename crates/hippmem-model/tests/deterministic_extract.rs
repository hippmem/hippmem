//! acceptance test: DeterministicExtractor
//!
//! Locale-specific test targets live in `tests/fixtures/extract/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_core::model::unit::{ContentType, Language, MemoryContent};
use hippmem_model::deterministic::extract::DeterministicExtractor;
use hippmem_model::traits::Extractor;

/// Discover available locale fixtures.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/extract", env!("CARGO_MANIFEST_DIR"));
    let mut locales = vec![];
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".json") {
                locales.push(name.trim_end_matches(".json").to_string());
            }
        }
    }
    locales.sort();
    if locales.is_empty() {
        panic!("no locale fixtures found in extract/");
    }
    locales
}

/// Load extract fixture for a specific locale.
fn load_fixture(locale: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/extract/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = std::fs::read_to_string(&path).expect("failed to read fixture");
    serde_json::from_str(&data).expect("invalid fixture")
}

fn make_content(text: &str, locale: &str) -> MemoryContent {
    MemoryContent {
        raw: text.to_string(),
        summary: None,
        normalized: None,
        language: if locale == "zh" {
            Language::Zh
        } else {
            Language::En
        },
        content_type: ContentType::UserStatement,
    }
}

/// Text with causal connectives must produce an explicit CausalClaim.
/// All active locales are tested automatically.
#[test]
fn explicit_causal_extraction() {
    for locale in discover_fixture_locales() {
        let ext = DeterministicExtractor;
        let fixture = load_fixture(&locale);
        let content = make_content(fixture["causal_text"].as_str().unwrap(), &locale);
        let result = ext.extract_sync_immediate(&content).unwrap();
        assert!(
            !result.explicit_causals.is_empty(),
            "[{locale}] text containing '{}' should produce an explicit CausalClaim",
            fixture["causal_label"].as_str().unwrap()
        );
    }
}

/// Entity extraction: capitalized proper names are recognized. All locales.
#[test]
fn entity_extraction_from_proper_names() {
    for locale in discover_fixture_locales() {
        let ext = DeterministicExtractor;
        let fixture = load_fixture(&locale);
        let content = make_content(fixture["tech_text"].as_str().unwrap(), &locale);
        let result = ext.extract_sync_immediate(&content).unwrap();
        assert!(
            !result.entities.is_empty(),
            "[{locale}] should extract at least one entity (Rust/Tantivy)"
        );
    }
}

/// Strong extraction: preference markers are detectable. All locales.
#[test]
fn strong_extraction_preference_detection() {
    for locale in discover_fixture_locales() {
        let ext = DeterministicExtractor;
        let fixture = load_fixture(&locale);
        let content = make_content(fixture["pref_text"].as_str().unwrap(), &locale);
        let result = ext.extract_sync_strong(&content).unwrap();
        assert!(
            !result.preferences.is_empty(),
            "[{locale}] text containing '{}' should produce a PreferenceFrame",
            fixture["pref_label"].as_str().unwrap()
        );
    }
}

/// Strong extraction confidence is low. All locales.
#[test]
fn strong_extraction_low_confidence() {
    for locale in discover_fixture_locales() {
        let ext = DeterministicExtractor;
        let fixture = load_fixture(&locale);
        let content = make_content(fixture["decision_text"].as_str().unwrap(), &locale);
        let result = ext.extract_sync_strong(&content).unwrap();
        assert!(
            result.confidence.value() <= 0.5,
            "[{locale}] fallback backend confidence should be <= 0.5"
        );
    }
}

/// backend_id is correct.
#[test]
fn backend_id_is_deterministic_rules() {
    let ext = DeterministicExtractor;
    assert_eq!(ext.backend_id(), "deterministic-rules");
}

// ── Chinese entity extraction ──

/// Text with proper names: extract person, place, and ASCII entities.
/// Locale-specific assertions come from fixture. All locales tested.
#[test]
fn entity_extraction_person_and_place() {
    for locale in discover_fixture_locales() {
        let ext = DeterministicExtractor;
        let fixture = load_fixture(&locale);
        let content = make_content(fixture["entity_text"].as_str().unwrap(), &locale);
        let result = ext.extract_sync_immediate(&content).unwrap();

        let entity_texts: Vec<&str> = result.entities.iter().map(|e| e.text.as_str()).collect();
        let person = fixture["entity_person"].as_str().unwrap();
        let place = fixture["entity_place"].as_str().unwrap();
        assert!(
            entity_texts.contains(&person),
            "[{locale}] should extract person name '{}', actual entities: {entity_texts:?}",
            person
        );
        assert!(
            entity_texts.contains(&place),
            "[{locale}] should extract place name '{}', actual entities: {entity_texts:?}",
            place
        );
        assert!(
            entity_texts.contains(&"Rust"),
            "[{locale}] should extract 'Rust', actual entities: {entity_texts:?}"
        );
    }
}

/// Pure ASCII uppercase entities: Python, Java, Rust are still extracted correctly. All locales.
#[test]
fn ascii_uppercase_entities_still_work() {
    for locale in discover_fixture_locales() {
        let ext = DeterministicExtractor;
        let fixture = load_fixture(&locale);
        let content = make_content(fixture["compare_text"].as_str().unwrap(), &locale);
        let result = ext.extract_sync_immediate(&content).unwrap();

        let entity_texts: Vec<&str> = result.entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            entity_texts.contains(&"Python"),
            "[{locale}] should extract Python, actual entities: {entity_texts:?}"
        );
        assert!(
            entity_texts.contains(&"Java"),
            "[{locale}] should extract Java, actual entities: {entity_texts:?}"
        );
        assert!(
            entity_texts.contains(&"Rust"),
            "[{locale}] should extract Rust, actual entities: {entity_texts:?}"
        );
    }
}

/// Locale-specific input extracts proper names. All locales tested.
#[test]
fn named_entity_extraction_locale() {
    for locale in discover_fixture_locales() {
        let ext = DeterministicExtractor;
        let fixture = load_fixture(&locale);
        let content = make_content(fixture["cn_entity_text"].as_str().unwrap(), &locale);
        let result = ext.extract_sync_immediate(&content).unwrap();

        let entity_names: Vec<&str> = fixture["cn_entity_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(
            !result.entities.is_empty(),
            "[{locale}] should extract proper names ({}), but entity list is empty",
            entity_names.join("/")
        );
        let entity_texts: Vec<&str> = result.entities.iter().map(|e| e.text.as_str()).collect();
        let has_any = entity_names.iter().any(|name| entity_texts.contains(name));
        assert!(
            has_any,
            "[{locale}] should extract at least one proper name, actual entities: {entity_texts:?}"
        );
    }
}

/// Pure English input still extracts uppercase entities correctly (regression check).
#[test]
fn english_entities_still_extracted_regression() {
    let content = MemoryContent {
        raw: "I use Rust and Tantivy for full-text search".to_string(),
        summary: None,
        normalized: None,
        language: Language::En,
        content_type: ContentType::UserStatement,
    };
    let ext = DeterministicExtractor;
    let result = ext.extract_sync_immediate(&content).unwrap();

    let entity_texts: Vec<&str> = result.entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        entity_texts.contains(&"Rust"),
        "English input should extract Rust, actual entities: {entity_texts:?}"
    );
    assert!(
        entity_texts.contains(&"Tantivy"),
        "English input should extract Tantivy, actual entities: {entity_texts:?}"
    );
}

// ── CJK entity extraction ──

/// Entity extraction across CJK locales: Chinese (zh) is active; Japanese (ja)
/// and Korean (ko) are architectural placeholders. Adding a new language
/// requires registering an entity recognizer for the locale + uncommenting and
/// filling in the corresponding test block below.
#[test]
fn entity_extraction_supports_cjk() {
    let ext = DeterministicExtractor;
    let zh = load_fixture("zh");

    // ── Chinese (zh): jieba + rule-based entity extraction ──
    {
        let content = MemoryContent {
            raw: zh["batch_text"].as_str().unwrap().to_string(),
            summary: None,
            normalized: None,
            language: Language::Zh,
            content_type: ContentType::UserStatement,
        };
        let result = ext.extract_sync_immediate(&content).unwrap();
        let entity_texts: Vec<&str> = result.entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            entity_texts.contains(&"Transformer"),
            "[zh] should extract 'Transformer' entity, actual: {entity_texts:?}"
        );
    }

    // ── Japanese (ja): architectural placeholder ──
    // (when ja entity recognizer is registered, uncomment and add ja text)
    // {
    //     let content = MemoryContent {
    //         raw: "Japanese text about Transformer architecture in NLP".to_string(),
    //         summary: None,
    //         normalized: None,
    //         language: Language::Ja,
    //         content_type: ContentType::UserStatement,
    //     };
    //     let result = ext.extract_sync_immediate(&content).unwrap();
    //     let entity_texts: Vec<&str> =
    //         result.entities.iter().map(|e| e.text.as_str()).collect();
    //     assert!(
    //         entity_texts.contains(&"Transformer"),
    //         "[ja] should extract 'Transformer' entity"
    //     );
    // }

    // ── Korean (ko): architectural placeholder ──
    // (when ko entity recognizer is registered, uncomment and add ko text)
    // {
    //     let content = MemoryContent {
    //         raw: "Korean text about Transformer architecture in NLP".to_string(),
    //         summary: None,
    //         normalized: None,
    //         language: Language::Ko,
    //         content_type: ContentType::UserStatement,
    //     };
    //     let result = ext.extract_sync_immediate(&content).unwrap();
    //     let entity_texts: Vec<&str> =
    //         result.entities.iter().map(|e| e.text.as_str()).collect();
    //     assert!(
    //         entity_texts.contains(&"Transformer"),
    //         "[ko] should extract 'Transformer' entity"
    //     );
    // }
}
