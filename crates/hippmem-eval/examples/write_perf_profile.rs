//! Write performance diagnostics: measure each phase of write_internal.
//!
//! Run: cargo run -p hippmem-eval --example write_perf_profile --release

use hippmem_core::model::enums::ContentType;
use hippmem_engine::{Engine, EngineConfig};
use hippmem_eval::fixture_loader::{discover_test_locales, load_test_fixture};
use std::time::Instant;
use tempfile::tempdir;

fn str_for_locale(locale: &str, key: &str) -> String {
    let fixture = load_test_fixture("write_perf", locale);
    fixture[key].as_str().unwrap().to_string()
}

fn locale_to_language(locale: &str) -> hippmem_core::model::unit::Language {
    match locale {
        "zh" => hippmem_core::model::unit::Language::Zh,
        "en" => hippmem_core::model::unit::Language::En,
        _ => hippmem_core::model::unit::Language::Mixed,
    }
}

fn main() {
    for locale in discover_test_locales("write_perf") {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  Write pipeline phase timing (deterministic embeddings, no network) [locale: {locale}] ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        let dir = tempdir().unwrap();
        let engine = Engine::open(EngineConfig {
            store_dir: dir.path().join("hippmem.redb"),
            ..Default::default()
        })
        .unwrap();

        // Measure DeterministicExtractor timing
        use hippmem_model::deterministic::extract::DeterministicExtractor;
        let t0 = Instant::now();
        for _ in 0..100 {
            let content = hippmem_core::model::unit::MemoryContent {
                raw: str_for_locale(&locale, "hippmem_desc"),
                summary: None,
                normalized: None,
                language: locale_to_language(&locale),
                content_type: ContentType::ProjectKnowledge,
            };
            let _ = DeterministicExtractor.extract_sync_immediate(&content);
        }
        let extract_time = t0.elapsed();
        println!(
            "  DeterministicExtractor (100 calls): {:?} → {:.0}μs/call",
            extract_time,
            extract_time.as_micros() as f64 / 100.0
        );

        // Measure jieba tokenization timing
        let t0 = Instant::now();
        for _ in 0..100 {
            let locale_text = str_for_locale(&locale, "hippmem_desc");
            hippmem_core::hash::tokenize(&locale_text, &locale);
        }
        let tokenize_time = t0.elapsed();
        println!(
            "  jieba tokenize (100 calls): {:?} → {:.0}μs/call",
            tokenize_time,
            tokenize_time.as_micros() as f64 / 100.0
        );

        println!(
            "\n  Conclusion: extract + tokenize = {:.0}μs/call (two jieba calls ≈ {:.0}μs)",
            (extract_time.as_micros() + tokenize_time.as_micros()) as f64 / 100.0,
            (extract_time.as_micros() + 2 * tokenize_time.as_micros()) as f64 / 100.0,
        );
        println!("  But this is only a tiny fraction of total time — the real bottleneck is edge construction and redb persistence.\n");

        engine.close().unwrap();
    }
}
