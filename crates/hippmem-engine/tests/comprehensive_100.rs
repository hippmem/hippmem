//! Comprehensive test: insert 100 entries of various kinds into HippMem and verify.
//!
//! Covers ContentType: UserStatement, Preference, Decision, ProjectKnowledge,
//! TaskState, Correction, ExternalContext, Identity.
//!
//! All locale-specific test data lives in `tests/fixtures/comprehensive_100/<locale>.json`.
//! This file only holds schema definitions, locale discovery, and assertion logic.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::{LinkType, RetrievalMode};
use hippmem_core::model::unit::{MemoryStage, WriteContext};
use hippmem_engine::{
    Engine, EngineConfig, ListInput, RetrieveContext, RetrieveInput, WriteMemoryInput,
};
use serde::Deserialize;
use std::fs;
use tempfile::tempdir;

// ── Locale discovery ──

/// Discover available locales by listing fixture files in the comprehensive_100 directory.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!(
        "{}/tests/fixtures/comprehensive_100",
        env!("CARGO_MANIFEST_DIR")
    );
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
        panic!("no locale fixtures found in {dir}");
    }
    locales
}

// ── Merged fixture schema ──

/// Top-level merged fixture: entries + queries + params in one file.
#[derive(Debug, Deserialize)]
struct ComprehensiveFixture {
    entries: Vec<MemoryEntry>,
    queries: Vec<KeywordQuery>,
    params: LocaleParams,
}

/// One memory fixture entry.
#[derive(Debug, Deserialize)]
struct MemoryEntry {
    content: String,
    content_type: ContentType,
    importance: f32,
    #[allow(dead_code)]
    group: String,
}

/// Keyword retrieval query fixture entry.
#[derive(Debug, Deserialize)]
struct KeywordQuery {
    query: String,
    expected_keyword: String,
    min_results: usize,
}

/// Locale-specific test params (queries, keywords, etc.).
#[derive(Debug, Deserialize)]
struct LocaleParams {
    persist_query: String,
    persist_keyword: String,
    rust_query: String,
    why_query: String,
    explanatory_keywords: Vec<String>,
    correction_query: String,
    correction_keywords: Vec<String>,
    preference_query: String,
    preference_keywords: Vec<String>,
}

/// Load the merged fixture for a specific locale.
fn load_fixture(locale: &str) -> ComprehensiveFixture {
    let path = format!(
        "{}/tests/fixtures/comprehensive_100/{locale}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale = locale
    );
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("failed to parse fixture {path}: {e}"))
}

// ── Test helpers ──

fn make_context(conv_id: u64, ts_ms: i64) -> WriteContext {
    WriteContext {
        conversation_id: Some(conv_id),
        session_id: Some(conv_id / 10),
        project_id: Some(1),
        task_id: None,
        user_id: Some(1),
        local_time: hippmem_core::time::Timestamp(ts_ms),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

fn group_to_conv_id(group: &str) -> u64 {
    match group {
        "statements" => 1,
        "preferences" => 2,
        "decisions" => 3,
        "project_knowledge" => 4,
        "task_states" => 5,
        "corrections" => 6,
        "assistant_observations" => 7,
        "identities" => 8,
        _ => 1,
    }
}

/// Generate 100 memory write inputs from a locale-specific fixture.
fn generate_100_memories(fixture: &ComprehensiveFixture) -> Vec<(WriteMemoryInput, i64)> {
    let base_ts: i64 = 1_700_000_000_000;

    fixture
        .entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let ts = base_ts + i as i64 * 1000;
            let conv_id = group_to_conv_id(&entry.group);
            (
                WriteMemoryInput {
                    content: entry.content.clone(),
                    content_type: Some(entry.content_type),
                    context: make_context(conv_id, ts),
                    importance_hint: Some(entry.importance),
                    source_refs: vec![],
                },
                ts,
            )
        })
        .collect()
}

/// Write all memories from a fixture to the engine.
fn seed_engine(engine: &Engine, fixture: &ComprehensiveFixture) {
    let inputs = generate_100_memories(fixture);
    for (input, _) in &inputs {
        engine
            .write(WriteMemoryInput {
                content: input.content.clone(),
                content_type: input.content_type,
                context: input.context.clone(),
                importance_hint: input.importance_hint,
                source_refs: vec![],
            })
            .unwrap();
    }
}

// ═══════════════════════════════════════════════════
// Test 1: write 100 entries and verify core metrics
// ═══════════════════════════════════════════════════

#[test]
fn write_100_memories_all_indexed() {
    for locale in discover_fixture_locales() {
        let fixture = load_fixture(&locale);
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("hippmem.redb");

        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();

        let inputs = generate_100_memories(&fixture);
        let total = inputs.len();

        let mut indexed = 0;
        let mut links_total = 0;
        let mut entity_links = 0;
        let mut causal_links = 0;
        let mut content_types = std::collections::HashMap::new();

        for (input, _) in &inputs {
            let output = engine
                .write(WriteMemoryInput {
                    content: input.content.clone(),
                    content_type: input.content_type,
                    context: input.context.clone(),
                    importance_hint: input.importance_hint,
                    source_refs: vec![],
                })
                .unwrap();

            if output.stage_reached == MemoryStage::Indexed {
                indexed += 1;
            }

            links_total += output.created_links.len();
            for link in &output.created_links {
                if link.link_type == LinkType::EntityOverlap {
                    entity_links += 1;
                }
                if link.link_type == LinkType::Causal {
                    causal_links += 1;
                }
            }

            *content_types
                .entry(format!("{:?}", input.content_type.unwrap()))
                .or_insert(0) += 1;
        }

        println!("\n========== Write 100 memories [{locale}] ==========");
        println!("Total written: {}", total);
        println!("Stage=Indexed: {}", indexed);
        println!("Total created links: {}", links_total);
        println!("  - EntityOverlap links: {}", entity_links);
        println!("  - CausalChain links: {}", causal_links);
        println!(
            "Avg links per entry: {:.1}",
            links_total as f64 / total as f64
        );
        println!();
        println!("ContentType distribution:");
        for (ct, count) in content_types.iter() {
            println!("  {}: {}", ct, count);
        }
        println!("==========================================\n");

        assert_eq!(
            indexed, total,
            "[{locale}] all 100 memories must reach Indexed stage"
        );
        assert!(
            links_total > 0,
            "[{locale}] should produce at least some association links"
        );

        let list_out = engine
            .list(ListInput {
                limit: 200,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            list_out.items.len(),
            total,
            "[{locale}] List should return all 100 memories"
        );
        assert!(
            list_out.total >= total as u64,
            "[{locale}] total should be >= 100"
        );

        engine.close().unwrap();
    }
}

// ═══════════════════════════════════════════════════
// Test 2: retrieval verification
// ═══════════════════════════════════════════════════

#[test]
fn retrieve_100_memories_quality() {
    for locale in discover_fixture_locales() {
        let fixture = load_fixture(&locale);
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("hippmem.redb");

        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();

        seed_engine(&engine, &fixture);

        let ctx = RetrieveContext::default();
        let queries = &fixture.queries;

        let mut pass = 0;
        let mut fail = 0;
        let total_queries = queries.len();

        println!("\n========== 100-memory retrieval quality [{locale}] ==========");
        println!("(deterministic embedder + rule extractor environment)");
        for q in queries {
            let results = engine
                .retrieve(RetrieveInput {
                    query: q.query.clone(),
                    context: ctx.clone(),
                    top_k: 5,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            let has_keyword = results
                .results
                .iter()
                .any(|r| r.memory.content.raw.contains(&q.expected_keyword));

            let enough_results = results.results.len() >= q.min_results;

            let status = if has_keyword && enough_results {
                pass += 1;
                "✅"
            } else if has_keyword {
                pass += 1;
                "⚠️"
            } else {
                fail += 1;
                "❌"
            };

            println!(
                "{} [{}] retrieve '{}' → {} results, top-1: '{}' (score={:.3})",
                status,
                if has_keyword { "hit" } else { "miss" },
                q.query,
                results.results.len(),
                results
                    .results
                    .first()
                    .map(|r| r.memory.content.raw.chars().take(40).collect::<String>())
                    .unwrap_or_default(),
                results
                    .results
                    .first()
                    .map(|r| r.final_score)
                    .unwrap_or(0.0),
            );

            if !has_keyword {
                println!(
                    "  all results: {:?}",
                    results
                        .results
                        .iter()
                        .map(|r| r.memory.content.raw.chars().take(40).collect::<String>())
                        .collect::<Vec<_>>()
                );
            }
        }

        println!(
            "\nRetrieval quality [{locale}]: {}/{} passed, {} failed",
            pass, total_queries, fail
        );
        println!(
            "Pass rate: {:.1}%",
            pass as f64 / total_queries as f64 * 100.0
        );
        println!("===============================================\n");

        assert!(
            pass as f64 / total_queries as f64 >= 0.6,
            "[{locale}] retrieval pass rate should be >= 60%, actual {:.1}%",
            pass as f64 / total_queries as f64 * 100.0
        );

        engine.close().unwrap();
    }
}

// ═══════════════════════════════════════════════════
// Test 3: persistence — reopen after close
// ═══════════════════════════════════════════════════

#[test]
fn persist_and_reopen_100_memories() {
    for locale in discover_fixture_locales() {
        let fixture = load_fixture(&locale);
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("hippmem.redb");

        // Round 1: write 100 entries
        {
            let engine = Engine::open(EngineConfig {
                store_dir: store_path.clone(),
                ..Default::default()
            })
            .unwrap();

            seed_engine(&engine, &fixture);
            engine.close().unwrap();
        }

        // Round 2: reopen and verify
        {
            let engine = Engine::open(EngineConfig {
                store_dir: store_path.clone(),
                ..Default::default()
            })
            .unwrap();

            let list_out = engine
                .list(ListInput {
                    limit: 200,
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(
                list_out.items.len(),
                100,
                "[{locale}] should still have 100 memories after reopen"
            );

            let results = engine
                .retrieve(RetrieveInput {
                    query: fixture.params.persist_query.clone(),
                    context: RetrieveContext::default(),
                    top_k: 3,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            assert!(
                !results.results.is_empty(),
                "[{locale}] should be able to retrieve '{}'-related memories after reopen",
                fixture.params.persist_keyword
            );
            assert!(
                results.results.iter().any(|r| r
                    .memory
                    .content
                    .raw
                    .contains(&fixture.params.persist_keyword)),
                "[{locale}] should contain user identity memory"
            );

            println!(
                "\nPersistence test [{locale}]: retrieve '{}' after reopen → {} results, top-1: '{}'",
                fixture.params.persist_keyword,
                results.results.len(),
                results
                    .results
                    .first()
                    .map(|r| r.memory.content.raw.chars().take(50).collect::<String>())
                    .unwrap_or_default()
            );

            engine.close().unwrap();
        }
    }
}

// ═══════════════════════════════════════════════════
// Test 4: association graph — verify EntityOverlap links
// ═══════════════════════════════════════════════════

#[test]
fn graph_entity_overlap_verification() {
    for locale in discover_fixture_locales() {
        let fixture = load_fixture(&locale);
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("hippmem.redb");

        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();

        let inputs = generate_100_memories(&fixture);
        let mut all_outputs = Vec::new();
        for (input, _) in &inputs {
            let output = engine
                .write(WriteMemoryInput {
                    content: input.content.clone(),
                    content_type: input.content_type,
                    context: input.context.clone(),
                    importance_hint: input.importance_hint,
                    source_refs: vec![],
                })
                .unwrap();
            all_outputs.push(output);
        }

        let total_entity_links: usize = all_outputs
            .iter()
            .map(|o| {
                o.created_links
                    .iter()
                    .filter(|l| l.link_type == LinkType::EntityOverlap)
                    .count()
            })
            .sum();

        println!("\n========== Graph association [{locale}] ==========");
        println!("Total EntityOverlap links: {}", total_entity_links);
        println!(
            "Avg EntityOverlap per entry: {:.1}",
            total_entity_links as f64 / 100.0
        );
        assert!(
            total_entity_links >= 30,
            "[{locale}] 100 entity-sharing memories should produce at least 30 EntityOverlap links, actual {}",
            total_entity_links
        );

        let results = engine
            .retrieve(RetrieveInput {
                query: fixture.params.rust_query.clone(),
                context: RetrieveContext::default(),
                top_k: 10,
                max_hops: Some(2),
                retrieval_mode: RetrievalMode::Deep,
            })
            .unwrap();

        let rust_count = results
            .results
            .iter()
            .filter(|r| r.memory.content.raw.contains("Rust"))
            .count();
        println!(
            "retrieve '{}' → {} results, containing 'Rust': {}",
            fixture.params.rust_query,
            results.results.len(),
            rust_count
        );
        assert!(
            rust_count >= 3,
            "[{locale}] retrieving Rust should return at least 3 memories containing Rust"
        );
        println!("==================================\n");

        engine.close().unwrap();
    }
}

// ═══════════════════════════════════════════════════
// Test 5: filter List by ContentType
// ═══════════════════════════════════════════════════

#[test]
fn list_filter_by_content_type() {
    for locale in discover_fixture_locales() {
        let fixture = load_fixture(&locale);
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("hippmem.redb");

        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();

        seed_engine(&engine, &fixture);

        let types_to_check = vec![
            (ContentType::Decision, "Decision", 20),
            (ContentType::Preference, "Preference", 15),
            (ContentType::ProjectKnowledge, "ProjectKnowledge", 20),
            (ContentType::TaskState, "TaskState", 15),
            (ContentType::Correction, "Correction", 5),
            (ContentType::UserStatement, "UserStatement", 15),
            (ContentType::AssistantObservation, "AssistantObs", 10),
        ];

        println!("\n========== ContentType filter [{locale}] ==========");
        for (ct, name, expected) in &types_to_check {
            let list_out = engine
                .list(ListInput {
                    limit: 50,
                    content_type: Some(*ct),
                    ..Default::default()
                })
                .unwrap();
            println!(
                "{}: expected={}, actual={} {}",
                name,
                expected,
                list_out.items.len(),
                if list_out.items.len() == *expected {
                    "✅"
                } else {
                    "⚠️"
                }
            );
            assert_eq!(
                list_out.items.len(),
                *expected,
                "[{locale}] {name} should have exactly {expected} entries",
            );
        }
        println!("============================================\n");

        engine.close().unwrap();
    }
}

// ═══════════════════════════════════════════════════
// Test 6: question-aware boosting
// ═══════════════════════════════════════════════════

#[test]
fn question_aware_boosting() {
    for locale in discover_fixture_locales() {
        let fixture = load_fixture(&locale);
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("hippmem.redb");

        let engine = Engine::open(EngineConfig {
            store_dir: store_path.clone(),
            ..Default::default()
        })
        .unwrap();

        seed_engine(&engine, &fixture);

        let ctx = RetrieveContext::default();
        let p = &fixture.params;

        // "Why" query → expect results containing causal keywords
        {
            let results = engine
                .retrieve(RetrieveInput {
                    query: p.why_query.clone(),
                    context: ctx.clone(),
                    top_k: 5,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            let has_explanatory = results.results.iter().any(|r| {
                p.explanatory_keywords
                    .iter()
                    .any(|kw| r.memory.content.raw.contains(kw))
            });

            println!(
                "\n[{locale}] {} → top-1: '{}' (explanatory={})",
                p.why_query,
                results
                    .results
                    .first()
                    .map(|r| r.memory.content.raw.chars().take(50).collect::<String>())
                    .unwrap_or_default(),
                has_explanatory
            );
            assert!(
                has_explanatory,
                "[{locale}] 'why' query should return results with explanatory markers"
            );
        }

        // Correction query
        {
            let results = engine
                .retrieve(RetrieveInput {
                    query: p.correction_query.clone(),
                    context: ctx.clone(),
                    top_k: 5,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            let has_correction = results.results.iter().any(|r| {
                p.correction_keywords
                    .iter()
                    .any(|kw| r.memory.content.raw.contains(kw))
            });

            println!(
                "[{locale}] correction query → top-1: '{}' (correction={})",
                results
                    .results
                    .first()
                    .map(|r| r.memory.content.raw.chars().take(60).collect::<String>())
                    .unwrap_or_default(),
                has_correction
            );
            assert!(
                has_correction,
                "[{locale}] correction query should return results with correction markers"
            );
        }

        // Preference query
        {
            let results = engine
                .retrieve(RetrieveInput {
                    query: p.preference_query.clone(),
                    context: ctx,
                    top_k: 5,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            let has_pref = results.results.iter().any(|r| {
                p.preference_keywords
                    .iter()
                    .any(|kw| r.memory.content.raw.contains(kw))
            });

            println!(
                "[{locale}] preference query → top-1: '{}' (preference={})",
                results
                    .results
                    .first()
                    .map(|r| r.memory.content.raw.chars().take(60).collect::<String>())
                    .unwrap_or_default(),
                has_pref
            );
            assert!(
                has_pref,
                "[{locale}] preference query should return results with preference info"
            );
        }

        engine.close().unwrap();
    }
}
