//! Personal memory evaluation
//!
//! This test uses the API Embedding backend (non-degraded) to persist a set of personal memories
//! and asserts ranking correctness for key queries.
//! Design goals:
//!   1. Verify that the Top-1 of a causal query ("Why did I choose redb?") is the causal explanation, not a factual statement
//!   2. Verify that the Top-1 of an identity query ("Who am I?") is identity information, not a side description
//!   3. Output full diagnostic information for tuning analysis
//!
//! Prerequisite: set the OPENAI_API_KEY environment variable
//! Run: cargo test -p hippmem-eval --test chenming_personal_eval --features api-backends -- --nocapture

#![allow(dead_code)] // test helpers enabled on demand
//!
//! Test data is loaded from tests/fixtures/personal_memory_eval/<locale>.json per P7.
//! Each fixture defines M1 (identity), M2 (projects), M3 (causal) memories and
//! locale-specific query strings / keywords.

mod common;

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_engine::{Engine, RetrieveInput, WriteMemoryInput};
#[cfg(feature = "api-backends")]
use tempfile::tempdir;

// ═══════════════════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Truncate long text for printing
fn short(s: &str) -> String {
    s.chars().take(80).collect()
}

/// Write the three personal memories (identity, projects, causal) for a given locale.
fn write_memories(engine: &Engine, fixture: &serde_json::Value) {
    for (content_key, ct, importance) in [
        ("m1_identity", ContentType::Preference, Some(0.7)),
        ("m2_projects", ContentType::ProjectKnowledge, Some(0.6)),
        ("m3_causal", ContentType::Decision, Some(0.9)),
    ] {
        let content = fixture[content_key].as_str().unwrap().to_string();
        let out = engine
            .write(WriteMemoryInput {
                content: content.to_string(),
                content_type: Some(ct),
                context: common::write_ctx(),
                importance_hint: importance,
                source_refs: vec![],
            })
            .unwrap();
        println!(
            "  wrote: id={} content=\"{}\" links={}",
            out.memory_id.0,
            short(&content),
            out.created_links.len()
        );
    }
}

fn retrieve_diagnostic(engine: &Engine, query: &str, top_k: usize) {
    for mode in [
        RetrievalMode::Balanced,
        RetrievalMode::Diagnostic,
        RetrievalMode::Deep,
    ] {
        println!("\n  --- retrieval mode: {:?} ---", mode);
        let results = engine
            .retrieve(RetrieveInput {
                query: query.to_string(),
                context: common::retrieve_ctx(),
                top_k,
                max_hops: Some(2),
                retrieval_mode: mode,
            })
            .unwrap();

        println!(
            "  returned {} results (hops={}):",
            results.results.len(),
            results.trace.hops_used
        );
        for (i, r) in results.results.iter().enumerate() {
            let dims: Vec<String> = r
                .matched_dimensions
                .iter()
                .map(|d| format!("{:?}", d))
                .collect();
            let trace_summary: Vec<String> = r
                .activation_trace
                .iter()
                .map(|a| format!("{:?}(+{:.3})", a.channel, a.energy_out))
                .collect();
            println!(
                "    {}. [{:.3}] {} | dims: {:?} | trace: {:?}",
                i + 1,
                r.final_score,
                short(&r.memory.content.raw),
                dims,
                trace_summary,
            );
        }

        // Channel seed info
        println!("  seeds ({}):", results.trace.seeds.len());
        for seed in &results.trace.seeds {
            println!(
                "    {:?} id={} energy={:.3}",
                seed.channel, seed.id.0, seed.initial_energy
            );
        }

        // Diagnostics
        if !results.diagnostics.channel_contributions.is_empty() {
            println!("  channel contributions:");
            for (ch, seed_count) in &results.diagnostics.channel_contributions {
                println!("    {:?}: seed_count={}", ch, seed_count);
            }
        }
        println!("  latency: {}ms", results.diagnostics.latency_ms);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Causal query — "Why did I choose redb?"
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg(feature = "api-backends")]
fn test_causal_query_why_redb() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Test 1: Causal query — Why did I choose redb?               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    for locale in common::discover_test_locales("personal_memory_eval") {
        println!("\n  ── locale: {} ──", locale);
        let fixture = common::load_test_fixture("personal_memory_eval", &locale);

        let dir = tempdir().unwrap();
        let engine = common::open_engine(&dir);
        write_memories(&engine, &fixture);

        let query = fixture["why_query"].as_str().unwrap();
        println!("\n  query: \"{}\"", query);

        let results = engine
            .retrieve(RetrieveInput {
                query: query.to_string(),
                context: common::retrieve_ctx(),
                top_k: 5,
                max_hops: Some(2),
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();

        assert!(
            results.results.len() >= 2,
            "expected at least 2 results, got {}",
            results.results.len()
        );

        let top1_content = &results.results[0].memory.content.raw;
        let top1_score = results.results[0].final_score;

        println!("\n  Top-1 [{:.3}]: \"{}\"", top1_score, short(top1_content));
        println!(
            "  Top-1 dims: {:?}",
            results.results[0]
                .matched_dimensions
                .iter()
                .map(|d| format!("{:?}", d))
                .collect::<Vec<_>>()
        );

        // Core assertion: Top-1 should be the causal explanation (M3), not the identity statement (M1)
        let top1_is_causal = fixture["causal_keywords"]
            .as_array()
            .unwrap()
            .iter()
            .any(|kw| top1_content.contains(kw.as_str().unwrap()));

        if top1_is_causal {
            println!("\n  ✅ Top-1 correct: causal explanation ranks first!");
        } else {
            // Print Top-2/3 to aid diagnosis
            println!("\n  ❌ Top-1 is not the causal explanation!");
            for i in 1..results.results.len().min(3) {
                let r = &results.results[i];
                println!(
                    "    #{} [{:.3}]: \"{}\"",
                    i + 1,
                    r.final_score,
                    short(&r.memory.content.raw)
                );
            }
        }

        retrieve_diagnostic(&engine, query, 5);

        // Strict assertion: with the API backend, the Top-1 of the causal query must be the causal explanation
        assert!(
            top1_is_causal,
            "API backend: Top-1 of 'Why did I choose redb?' must be the causal explanation (M3)"
        );

        engine.close().unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: Identity query — "Who am I?"
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg(feature = "api-backends")]
fn test_identity_query_who_am_i() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Test 2: Identity query — Who am I?                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    for locale in common::discover_test_locales("personal_memory_eval") {
        println!("\n  ── locale: {} ──", locale);
        let fixture = common::load_test_fixture("personal_memory_eval", &locale);

        let dir = tempdir().unwrap();
        let engine = common::open_engine(&dir);
        write_memories(&engine, &fixture);

        let query = fixture["who_query"].as_str().unwrap();
        println!("\n  query: \"{}\"", query);

        let results = engine
            .retrieve(RetrieveInput {
                query: query.to_string(),
                context: common::retrieve_ctx(),
                top_k: 5,
                max_hops: Some(2),
                retrieval_mode: RetrievalMode::Balanced,
            })
            .unwrap();

        assert!(
            results.results.len() >= 2,
            "expected at least 2 results, got {}",
            results.results.len()
        );

        let top1_content = &results.results[0].memory.content.raw;
        let top1_score = results.results[0].final_score;

        println!("\n  Top-1 [{:.3}]: \"{}\"", top1_score, short(top1_content));

        // Core assertion: Top-1 should be the identity statement (M1), not the project experience (M2)
        let top1_is_identity = fixture["identity_keywords"]
            .as_array()
            .unwrap()
            .iter()
            .any(|kw| top1_content.contains(kw.as_str().unwrap()));

        if top1_is_identity {
            println!("\n  ✅ Top-1 correct: identity information ranks first!");
        } else {
            println!("\n  ❌ Top-1 is not identity information!");
            for i in 1..results.results.len().min(3) {
                let r = &results.results[i];
                println!(
                    "    #{} [{:.3}]: \"{}\"",
                    i + 1,
                    r.final_score,
                    short(&r.memory.content.raw)
                );
            }
        }

        retrieve_diagnostic(&engine, query, 5);

        // Strict assertion: with the API backend, the Top-1 of the identity query must be identity information
        assert!(
            top1_is_identity,
            "API backend: Top-1 of 'Who am I?' must be identity information (M1)"
        );

        engine.close().unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Comprehensive — check ranking of all queries at once
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg(feature = "api-backends")]
fn test_all_queries_ranking() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Test 3: Comprehensive ranking check                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    for locale in common::discover_test_locales("personal_memory_eval") {
        println!("\n  ── locale: {} ──", locale);
        let fixture = common::load_test_fixture("personal_memory_eval", &locale);

        let dir = tempdir().unwrap();
        let engine = common::open_engine(&dir);
        write_memories(&engine, &fixture);

        // Query definitions: (query, expected_top1_keywords, description)
        // Chinese queries and keywords loaded from fixture per P7.
        let queries: Vec<(String, Vec<&str>, &str)> = vec![
            (
                fixture["why_query"].as_str().unwrap().to_string(),
                fixture["causal_keywords"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect(),
                "causal: why redb",
            ),
            (
                fixture["who_query"].as_str().unwrap().to_string(),
                fixture["identity_keywords"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect(),
                "identity: who am I",
            ),
            ("redb".to_string(), vec!["redb"], "keyword: redb"),
            (
                "RocksDB".to_string(),
                vec!["RocksDB", "compilation"],
                "keyword: RocksDB",
            ),
            (
                fixture["job_query"].as_str().unwrap().to_string(),
                fixture["identity_keywords"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect(),
                "occupation: what do I do",
            ),
        ];

        let mut passed = 0u32;
        let total = queries.len() as u32;

        for (query, expected_keywords, desc) in &queries {
            println!("\n  ── {} ──", desc);
            println!("  query: \"{}\"", query);

            let results = engine
                .retrieve(RetrieveInput {
                    query: query.to_string(),
                    context: common::retrieve_ctx(),
                    top_k: 5,
                    max_hops: Some(2),
                    retrieval_mode: RetrievalMode::Balanced,
                })
                .unwrap();

            if results.results.is_empty() {
                println!("  ❌ no results");
                continue;
            }

            let top1 = &results.results[0];
            let top1_text = &top1.memory.content.raw;
            let hit = expected_keywords.iter().any(|kw| top1_text.contains(kw));

            println!(
                "  Top-1 [{:.3}]: \"{}\" ",
                top1.final_score,
                short(top1_text)
            );
            println!(
                "  hit expected keywords: {} (expected: {:?})",
                if hit { "✅" } else { "❌" },
                expected_keywords
            );

            if hit {
                passed += 1;
            } else {
                for i in 1..results.results.len().min(3) {
                    let r = &results.results[i];
                    let t = &r.memory.content.raw;
                    let h = expected_keywords.iter().any(|kw| t.contains(kw));
                    println!(
                        "    #{} [{:.3}]: \"{}\" {}",
                        i + 1,
                        r.final_score,
                        short(t),
                        if h { "✅" } else { "" }
                    );
                }
            }
        }

        let rate = passed as f64 / total as f64;
        println!("\n  ═══════════════════════════════════════");
        println!("  overall: {}/{} = {:.0}%", passed, total, rate * 100.0);

        // Strict assertion: with the API backend, all 5 queries' Top-1 must hit expected keywords
        assert_eq!(
            passed, total,
            "API backend: all {} queries' Top-1 must hit expected keywords, got {}/{}",
            total, passed, total
        );

        engine.close().unwrap();
    }
}
