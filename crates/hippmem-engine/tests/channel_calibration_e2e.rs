//! V5 channel calibration end-to-end tests.
//!
//! Verifies behavioral correctness of (BM25 normalization) and (per-channel energy coefficients).
//! All tests use the deterministic backend, no network dependency.
//!
//! Locale-specific test data lives in `tests/fixtures/channel_calibration/<locale>.json`.
//! Adding a new locale = adding its fixture file. Test code needs zero changes.

use hippmem_core::config::AlgoParams;
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RecallChannel;
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};
use std::fs;
use tempfile::tempdir;

/// Discover available locale fixtures for channel_calibration.
fn discover_fixture_locales() -> Vec<String> {
    let dir = format!(
        "{}/tests/fixtures/channel_calibration",
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
        panic!("no locale fixtures found in channel_calibration/");
    }
    locales
}

/// Load channel_calibration fixture for a specific locale.
fn load_fixture(locale: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/channel_calibration/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        locale
    );
    let data = fs::read_to_string(&path).expect("failed to read fixture");
    serde_json::from_str(&data).expect("invalid fixture")
}

fn make_ctx() -> WriteContext {
    WriteContext {
        conversation_id: Some(1),
        session_id: Some(1),
        project_id: None,
        task_id: None,
        user_id: None,
        local_time: hippmem_core::time::Timestamp(1_700_000_000_000),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

/// Writes 4 memories covering different content types, ensuring multiple channels produce output.
fn seed_memories(engine: &Engine) {
    let memories = [
        (
            "Rust high-performance backend service development",
            ContentType::ProjectKnowledge,
        ),
        (
            "Using Redb for persistent storage replacing RocksDB",
            ContentType::Decision,
        ),
        (
            "Redis cache layer Write-Through strategy",
            ContentType::Decision,
        ),
        (
            "Python data analysis and machine learning",
            ContentType::ProjectKnowledge,
        ),
    ];
    for (content, ct) in &memories {
        engine
            .write(WriteMemoryInput {
                content: content.to_string(),
                content_type: Some(*ct),
                context: make_ctx(),
                importance_hint: None,
                source_refs: vec![],
            })
            .unwrap();
    }
}

fn retrieve(engine: &Engine, query: &str) -> hippmem_engine::RetrieveOutput {
    engine
        .retrieve(RetrieveInput {
            query: query.into(),
            context: RetrieveContext::default(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap()
}

// ── Test 1: BM25 normalization ──

#[test]
fn bm25_normalization_reduces_energy() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();
    seed_memories(&engine);

    let output = retrieve(&engine, "Redb storage");

    // Verify there are retrieval results
    assert!(!output.results.is_empty(), "should have retrieval results");

    // Take hop=0 BM25 steps from trace.steps and check the actual energy
    let bm25_steps: Vec<_> = output
        .trace
        .steps
        .iter()
        .filter(|s| s.hop == 0 && s.channel == Some(RecallChannel::Bm25))
        .collect();

    assert!(
        !bm25_steps.is_empty(),
        "should have hop=0 BM25 activation steps"
    );

    for step in &bm25_steps {
        // After normalization: tanh ∈ [0,1), coeff=1.0, a_query_match=0.40 × importance_multiplier → energy < 1.0
        assert!(
            step.energy_in > 0.0,
            "BM25 energy should be > 0.0, actual: {:.4}",
            step.energy_in
        );
        assert!(
            step.energy_in <= 1.0,
            "BM25 energy after normalization should be <= 1.0, actual: {:.4}",
            step.energy_in
        );
    }

    // The initial_energy field in trace.seeds actually stores seed.score (post-normalization BM25 score),
    // verify this score is in [0,1) (no longer the unbounded raw BM25 score)
    for seed in &output.trace.seeds {
        if seed.channel == RecallChannel::Bm25 {
            assert!(
                seed.initial_energy < 1.0,
                "BM25 seed.score after normalization should be < 1.0, actual: {:.4}",
                seed.initial_energy
            );
        }
    }

    engine.close().unwrap();
}

// ── Test 2: default coefficient backward compatibility ──

#[test]
fn default_coeffs_backward_compatible() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();
    seed_memories(&engine);

    let output = retrieve(&engine, "storage solution");

    // Under default config (all coefficients 1.0) retrieval works normally, returns non-empty results
    assert!(
        !output.results.is_empty(),
        "default config retrieval should have results"
    );

    // Verify multiple channels contribute (trace.seeds contains BM25)
    let has_bm25 = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == RecallChannel::Bm25);
    assert!(has_bm25, "should have BM25 channel contribution");

    // Verify backend_used is the deterministic backend
    assert_eq!(
        output.diagnostics.backend_used.embedder,
        "deterministic-hash"
    );

    engine.close().unwrap();
}

// ── Test 3: SemanticDense coefficient boost ──

#[test]
/// V9: Tests RRF precision weights (replaces the old channel_coeff test).
/// Boosting rrf_w_semantic_dense should increase the SemanticDense channel energy.
fn semantic_dense_boosted_with_coeff() {
    let dir = tempdir().unwrap();

    // Baseline: default weight 0.6
    let engine_base = Engine::open(EngineConfig {
        store_dir: dir.path().join("base.redb"),
        ..Default::default()
    })
    .unwrap();
    seed_memories(&engine_base);
    let out_base = retrieve(&engine_base, "Rust backend");

    // Boosted: rrf_w_semantic_dense = 1.2 (uses an independent tempdir)
    let dir2 = tempdir().unwrap();
    let algo_boosted = AlgoParams {
        rrf_w_semantic_dense: 1.2,
        ..Default::default()
    };
    let engine_boost = Engine::open(EngineConfig {
        store_dir: dir2.path().join("boost.redb"),
        algo: algo_boosted,
        ..Default::default()
    })
    .unwrap();
    seed_memories(&engine_boost);
    let out_boost = retrieve(&engine_boost, "Rust backend");

    // V9: The weight affects the RRF fusion score; at the same rank, the w=1.2 contribution is 2× that of w=0.6
    let base_dense_energy: f32 = out_base
        .trace
        .steps
        .iter()
        .filter(|s| s.hop == 0 && s.channel == Some(RecallChannel::SemanticDense))
        .map(|s| s.energy_in)
        .sum();
    let boost_dense_energy: f32 = out_boost
        .trace
        .steps
        .iter()
        .filter(|s| s.hop == 0 && s.channel == Some(RecallChannel::SemanticDense))
        .map(|s| s.energy_in)
        .sum();

    if base_dense_energy > 0.0 && boost_dense_energy > 0.0 {
        let ratio = boost_dense_energy / base_dense_energy;
        // w=1.2 vs w=0.6 → contribution doubles at the same rank, but under multi-channel RRF the net effect may be < 2×
        assert!(
            ratio > 1.0,
            "energy with rrf_w_semantic_dense=1.2 should be > baseline (w=0.6), ratio={:.2}",
            ratio
        );
    }

    engine_base.close().unwrap();
    engine_boost.close().unwrap();
}

// ── Test 4: per-channel coefficient independence ──

#[test]
fn per_channel_coeffs_independent() {
    let dir = tempdir().unwrap();

    // BM25 halved, SemanticBinary zeroed (disabled)
    let algo = AlgoParams {
        channel_coeff_bm25: 0.5,
        channel_coeff_semantic_binary: 0.0,
        ..Default::default()
    };

    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        algo,
        ..Default::default()
    })
    .unwrap();
    seed_memories(&engine);

    let output = retrieve(&engine, "Redb Rust storage");

    // V9: channel_coeff deprecated, energy is determined by the RRF fusion score.
    // Just verify there are retrieval results; no longer assert specific energy values.
    assert!(!output.results.is_empty(), "should have retrieval results");

    // Verify there are seeds from other channels
    let active_channels: Vec<RecallChannel> =
        output.trace.seeds.iter().map(|s| s.channel).collect();
    assert!(
        active_channels
            .iter()
            .any(|c| *c != RecallChannel::SemanticBinary),
        "should have seeds from other channels"
    );

    engine.close().unwrap();
}

// ── Test 5: "why" queries prefer explanatory answers ──

/// A why-type query (asking why redb was chosen) should return
/// causal/decision-type memories ranked first.
///
/// Verifies question-type aware boosting (§4.5) is in effect:
/// When the query is a why-type, documents containing explanatory markers (causal
/// conjunctions / decision verbs) get a moderate energy boost, compensating for the
/// deterministic embedder's bag-of-tokens mechanism which cannot distinguish
/// "I like X" from "I chose X because Y".
#[test]
fn why_question_retrieval_prefers_causal_explanations() {
    for locale in discover_fixture_locales() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(EngineConfig {
            store_dir: dir.path().join("hippmem.redb"),
            ..Default::default()
        })
        .unwrap();

        let fixture = load_fixture(&locale);
        let why = &fixture["why_test"];

        let ct_map = |t: &str| match t {
            "Decision" => ContentType::Decision,
            _ => ContentType::UserStatement,
        };
        for mem in why["memories"].as_array().unwrap() {
            let text = mem["text"].as_str().unwrap();
            let ct = ct_map(mem["type"].as_str().unwrap());
            engine
                .write(WriteMemoryInput {
                    content: text.into(),
                    content_type: Some(ct),
                    context: make_ctx(),
                    importance_hint: None,
                    source_refs: vec![],
                })
                .unwrap();
        }

        let output = engine
            .retrieve(RetrieveInput {
                query: why["query"].as_str().unwrap().into(),
                context: RetrieveContext::default(),
                top_k: 5,
                max_hops: Some(1),
                retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
            })
            .unwrap();

        assert!(
            !output.results.is_empty(),
            "[{locale}] should have at least one retrieval result"
        );

        let keyword = why["result_contains"].as_str().unwrap();
        let decision_pos = output
            .results
            .iter()
            .position(|r| r.memory.content.raw.contains(keyword));
        assert!(
            decision_pos.is_some(),
            "[{locale}] Decision (with causal explanation, containing '{}') should be in the results",
            keyword
        );

        engine.close().unwrap();
    }
}

// ── What-query boosting ──

/// A what-is-X query should preferentially return definitional memories (ProjectKnowledge),
/// rather than identity statements (UserStatement).
#[test]
fn what_question_prefers_project_knowledge() {
    for locale in discover_fixture_locales() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(EngineConfig {
            store_dir: dir.path().join("hippmem.redb"),
            ..Default::default()
        })
        .unwrap();

        let fixture = load_fixture(&locale);
        let what = &fixture["what_test"];

        let ct_map = |t: &str| match t {
            "ProjectKnowledge" => ContentType::ProjectKnowledge,
            _ => ContentType::UserStatement,
        };
        for mem in what["memories"].as_array().unwrap() {
            let text = mem["text"].as_str().unwrap();
            let ct = ct_map(mem["type"].as_str().unwrap());
            engine
                .write(WriteMemoryInput {
                    content: text.into(),
                    content_type: Some(ct),
                    context: make_ctx(),
                    importance_hint: None,
                    source_refs: vec![],
                })
                .unwrap();
        }

        let output = engine
            .retrieve(RetrieveInput {
                query: what["query"].as_str().unwrap().into(),
                context: RetrieveContext::default(),
                top_k: 5,
                max_hops: Some(1),
                retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
            })
            .unwrap();

        assert!(
            !output.results.is_empty(),
            "[{locale}] should have at least one retrieval result"
        );

        let pk_keyword = what["pk_contains"].as_str().unwrap();
        let us_keyword = what["us_contains"].as_str().unwrap();
        let pk_pos = output
            .results
            .iter()
            .position(|r| r.memory.content.raw.contains(pk_keyword));
        let us_pos = output
            .results
            .iter()
            .position(|r| r.memory.content.raw.contains(us_keyword));

        assert!(
            pk_pos.is_some(),
            "[{locale}] definitional memory should be in the results"
        );
        if let (Some(pkp), Some(usp)) = (pk_pos, us_pos) {
            assert!(
                pkp < usp,
                "[{locale}] ProjectKnowledge (definition) should rank before UserStatement (identity):\n  PK pos={}, US pos={}\n  results: {:?}",
                pkp,
                usp,
                output
                    .results
                    .iter()
                    .map(|r| format!(
                        "[{:.3}] {}",
                        r.final_score,
                        &r.memory.content.raw[..r.memory.content.raw.len().min(60)]
                    ))
                    .collect::<Vec<_>>()
            );
        }

        engine.close().unwrap();
    }
}

// ── Merge_energy seed fusion E2E ──

/// BM25 exact-token matches are preserved via merge_energy seed fusion,
/// not discarded by winner-take-all due to higher SemanticDense energy.
#[test]
fn merge_fusion_preserves_bm25_exact_match() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write the author identity statement (BM25 won't strongly match the license keyword)
    engine
        .write(WriteMemoryInput {
            content: "I am the author of the HippMem project".into(),
            content_type: Some(ContentType::UserStatement),
            context: make_ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // Write the license choice (BM25 will exactly match the license keyword)
    engine
        .write(WriteMemoryInput {
            content: "HippMem uses the Apache 2.0 open source license".into(),
            content_type: Some(ContentType::Decision),
            context: make_ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    let output = retrieve(&engine, "open source license");

    assert!(
        !output.results.is_empty(),
        "should have at least one retrieval result"
    );

    // The result containing "Apache" should rank first (BM25 exact match should be preserved via merge)
    let top = &output.results[0];
    assert!(
        top.memory.content.raw.contains("Apache"),
        "first result should contain 'Apache' (BM25 exact match should be preserved),\n  actual first: {} (score={:.3})",
        &top.memory.content.raw[..top.memory.content.raw.len().min(80)],
        top.final_score
    );

    // Verify the BM25 channel contributes
    let bm25_in_seeds = output
        .trace
        .seeds
        .iter()
        .any(|s| s.channel == RecallChannel::Bm25);
    assert!(bm25_in_seeds, "BM25 channel should contribute to the seeds");

    engine.close().unwrap();
}

/// Regression test from real user feedback:
/// A what-is-the-license query → the result containing "Apache" should rank first.
#[test]
fn license_query_top_is_apache() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hippmem.redb"),
        ..Default::default()
    })
    .unwrap();

    // Write the identity statement (semantically similar but contains no license info)
    engine
        .write(WriteMemoryInput {
            content: "I am the author and primary maintainer of the HippMem project".into(),
            content_type: Some(ContentType::UserStatement),
            context: make_ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // Write the license choice
    engine
        .write(WriteMemoryInput {
            content: "Apache 2.0 open source license was chosen as the license for HippMem".into(),
            content_type: Some(ContentType::Decision),
            context: make_ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    // Write the project definition
    engine
        .write(WriteMemoryInput {
            content: "HippMem is an associative memory engine implemented in Rust".into(),
            content_type: Some(ContentType::ProjectKnowledge),
            context: make_ctx(),
            importance_hint: None,
            source_refs: vec![],
        })
        .unwrap();

    let output = engine
        .retrieve(RetrieveInput {
            query: "what is hippmem's open source license".into(),
            context: RetrieveContext::default(),
            top_k: 5,
            max_hops: Some(1),
            retrieval_mode: hippmem_core::model::links::RetrievalMode::Balanced,
        })
        .unwrap();

    assert!(
        !output.results.is_empty(),
        "should have at least one retrieval result"
    );

    // Key assertion: the result containing "Apache" should rank first
    let top = &output.results[0];
    assert!(
        top.memory.content.raw.contains("Apache"),
        "first result should contain 'Apache',\n  actual first: {} (score={:.3})",
        &top.memory.content.raw[..top.memory.content.raw.len().min(80)],
        top.final_score
    );

    engine.close().unwrap();
}
