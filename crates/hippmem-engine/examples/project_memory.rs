//! HIPPMEM project-memory demo — show how associative memory connects knowledge
//! across sessions, surfacing WHY something happened, not just WHAT.
//!
//! Simulates an AI coding assistant that remembers a project ("Nova") across 4 sessions.
//! Demonstrates:
//!   1. Write pipeline: entity extraction, indexing, association discovery
//!   2. Multi-channel recall: 5 channels (BM25 + Entity + Semantic ×2 + Topic)
//!      vs the 1–2 channels a typical system uses
//!   3. Per-channel seed breakdown: which channel found what
//!
//! Run: cargo run --example project_memory
//! No API key required — uses the deterministic degraded backend (zero-config).
//! With real model backends, the graph edges and spreading activation add
//! another dimension (see --features api-backends).

use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::{RecallChannel, RetrievalMode};
use hippmem_core::model::unit::WriteContext;
use hippmem_engine::{Engine, EngineConfig, RetrieveContext, RetrieveInput, WriteMemoryInput};

// ── Demo fixture: 11 memories across 4 sessions ──

struct MemoryEntry {
    content: &'static str,
    content_type: ContentType,
    session_id: u64,
    importance: f32,
}

fn nova_memories() -> Vec<MemoryEntry> {
    vec![
        // Session 1 — Architecture discussion (Monday)
        MemoryEntry {
            content: "We're building Nova, a distributed task queue for Rust microservices.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 1,
            importance: 0.75,
        },
        MemoryEntry {
            content: "The key requirement is exactly-once delivery semantics.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 1,
            importance: 0.85,
        },
        MemoryEntry {
            content: "We're evaluating Redis Streams vs PostgreSQL as the message broker.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 1,
            importance: 0.7,
        },
        MemoryEntry {
            content: "Redis Streams don't guarantee exactly-once delivery without significant additional work — the consumer group model has edge cases around failure recovery.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 1,
            importance: 0.9,
        },
        // Session 2 — Implementation (Wednesday)
        MemoryEntry {
            content: "Decided to use PostgreSQL with SKIP LOCKED for the task queue. The transactional semantics match our exactly-once requirement perfectly.",
            content_type: ContentType::Decision,
            session_id: 2,
            importance: 0.9,
        },
        MemoryEntry {
            content: "Schema: tasks table with id, payload, status, priority, scheduled_at columns.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 2,
            importance: 0.6,
        },
        MemoryEntry {
            content: "Workers poll with SELECT ... FOR UPDATE SKIP LOCKED for safe concurrent dequeue — no two workers can claim the same task.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 2,
            importance: 0.7,
        },
        // Session 3 — Bug fix (Friday)
        MemoryEntry {
            content: "Race condition discovered: under high load (>500 tasks/sec), two workers occasionally claimed the same task. The SKIP LOCKED window had a sub-millisecond gap.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 3,
            importance: 0.8,
        },
        MemoryEntry {
            content: "Fixed the double-claim bug by adding a claimed_by worker_id column and a UNIQUE constraint on (id, claimed_by). Now the database enforces the exclusivity guarantee.",
            content_type: ContentType::Correction,
            session_id: 3,
            importance: 0.85,
        },
        // Session 4 — Performance optimization (next Monday)
        MemoryEntry {
            content: "Added a composite index on (status, priority, scheduled_at). Worker poll latency dropped from 12ms to 1.2ms — a 10x improvement.",
            content_type: ContentType::ProjectKnowledge,
            session_id: 4,
            importance: 0.7,
        },
        MemoryEntry {
            content: "The developer prefers Rust's type system and compile-time guarantees for distributed systems — catches entire classes of concurrency bugs before they reach production.",
            content_type: ContentType::Preference,
            session_id: 4,
            importance: 0.5,
        },
    ]
}

fn make_context(session_id: u64, ts_ms: i64) -> WriteContext {
    WriteContext {
        conversation_id: Some(session_id),
        session_id: Some(session_id),
        project_id: Some(1),
        task_id: None,
        user_id: Some(1),
        local_time: hippmem_core::time::Timestamp(ts_ms),
        preceding_memory_ids: vec![],
        source_refs: vec![],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   HIPPMEM Demo — AI Agent Long-Term Project Memory          ║");
    println!("║   Scenario: Building \"Nova\" (distributed task queue)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Open engine ──
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("nova.redb"),
        ..Default::default()
    })?;

    // ── Phase 1: Write memories across 4 sessions ──
    println!("━━━ PHASE 1: Writing project memories ━━━");
    println!();

    let entries = nova_memories();
    let base_ts: i64 = 1_700_000_000_000;
    let mut current_session: u64 = 0;
    let session_names = [
        "Architecture discussion",
        "Implementation",
        "Bug fix",
        "Performance optimization",
    ];
    let mut si = 0usize;
    let mut total_links = 0usize;

    for (i, entry) in entries.iter().enumerate() {
        if entry.session_id != current_session {
            current_session = entry.session_id;
            let label = session_names.get(si).unwrap_or(&"?");
            println!("  ── Session {}: {} ──", current_session, label);
            si += 1;
        }

        let out = engine.write(WriteMemoryInput {
            content: entry.content.to_string(),
            content_type: Some(entry.content_type),
            context: make_context(entry.session_id, base_ts + i as i64 * 3_600_000),
            importance_hint: Some(entry.importance),
            source_refs: vec![],
        })?;

        let links = out.created_links.len();
        total_links += links;
        println!(
            "    #{:02} [{}] +{} associations  {}",
            out.memory_id.0,
            format_ct(entry.content_type),
            links,
            clip(entry.content, 66),
        );
    }

    // Force fulltext index flush so BM25 sees all documents
    engine.flush_fulltext();

    println!();
    println!(
        "  ✅ {} memories written, {} associations created across 4 sessions.",
        entries.len(),
        total_links
    );
    println!();

    // ── Phase 2: Retrieval ──
    println!("━━━ PHASE 2: Retrieval ━━━");
    println!();
    println!("  Query: \"Why did we choose PostgreSQL as the message broker?\"");
    println!();

    let results = engine.retrieve(RetrieveInput {
        query: "Why did we choose PostgreSQL as the message broker?".into(),
        context: RetrieveContext::default(),
        top_k: 8,
        max_hops: Some(2),
        retrieval_mode: RetrievalMode::Deep,
    })?;

    // ── 2a: Multi-channel seed breakdown ──
    println!("  ── Multi-channel seed recall ──");
    println!();

    for ch in &[
        RecallChannel::Bm25,
        RecallChannel::EntityInverted,
        RecallChannel::SemanticDense,
        RecallChannel::SemanticBinary,
        RecallChannel::TopicCluster,
    ] {
        let channel_seeds: Vec<_> = results
            .trace
            .seeds
            .iter()
            .filter(|s| s.channel == *ch)
            .collect();
        if channel_seeds.is_empty() {
            continue;
        }
        println!(
            "    {:20}  {:>2} candidate(s)",
            format!("{:?}:", ch),
            channel_seeds.len()
        );
        for seed in channel_seeds.iter().take(3) {
            // Match seed to result memory for content preview
            let preview = results
                .results
                .iter()
                .find(|r| r.memory.id.0 == seed.id.0)
                .map(|r| clip(&r.memory.content.raw, 60))
                .unwrap_or_else(|| "(pruned during merge)".to_string());
            println!(
                "        #{} @{:.3}  {}",
                seed.id.0, seed.initial_energy, preview
            );
        }
        if channel_seeds.len() > 3 {
            println!("        ... and {} more", channel_seeds.len() - 3);
        }
    }

    println!();
    println!(
        "    Total: {} seeds across {} channels → {} merged results",
        results.trace.seeds.len(),
        5, // we listed 5 channels above
        results.results.len(),
    );
    println!();

    // ── 2b: Final ranked results ──
    println!("  ── Final ranked results (RRF-fused + reranked) ──");
    println!();
    for (rank, r) in results.results.iter().enumerate() {
        let dims: Vec<String> = r
            .matched_dimensions
            .iter()
            .map(|d| format!("{:?}", d))
            .collect();
        println!(
            "    #{}. [{:.3}] {}",
            rank + 1,
            r.final_score,
            clip(&r.memory.content.raw, 72),
        );
        println!("         dims: [{}]", dims.join(", "));
    }
    println!();

    // ── Phase 3: Diagnostics ──
    println!("━━━ PHASE 3: Diagnostics ━━━");
    println!();
    println!(
        "  hops_used: {}  |  steps: {}  |  latency: {}ms",
        results.trace.hops_used,
        results.trace.steps.len(),
        results.diagnostics.latency_ms,
    );

    // Show channel contribution counts from diagnostics
    if !results.diagnostics.channel_contributions.is_empty() {
        println!("  Channel contributions:");
        for (ch, count) in &results.diagnostics.channel_contributions {
            println!("    {:?}: {} results", ch, count);
        }
    }
    println!();

    // ── Phase 4: What this means ──
    println!("━━━ Why this matters ━━━");
    println!();
    println!("  A typical RAG system has 1–2 recall channels (BM25 + embeddings).");
    println!(
        "  HIPPMEM uses 5: Bm25, EntityInverted, SemanticDense, SemanticBinary, TopicCluster."
    );
    println!();
    println!("  Each channel sees the query through a different lens:");
    println!("    BM25          → keyword match (\"PostgreSQL\", \"message broker\")");
    println!("    EntityInverted → shared entities (PostgreSQL, Redis, Rust, Nova, ...)");
    println!("    SemanticDense  → meaning-level similarity (dense vectors)");
    println!("    SemanticBinary → fast approximate semantic match (binary codes)");
    println!("    TopicCluster   → same topic cluster (\"distributed systems\", \"databases\")");
    println!();
    println!("  Different channels find different memories. RRF fusion combines them");
    println!("  so the final ranking is stronger than any single channel alone.");
    println!();
    println!("  With --features api-backends, real models also populate Entity,",);
    println!("  Causal, and Topic dimensions, and the graph edges enable multi-hop");
    println!("  spreading activation for deeper associative recall.");
    println!();

    engine.close()?;
    Ok(())
}

fn clip(s: &str, max_len: usize) -> String {
    let truncated: String = s.chars().take(max_len).collect();
    if s.chars().count() > max_len {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

fn format_ct(ct: ContentType) -> &'static str {
    match ct {
        ContentType::Decision => "DECISION  ",
        ContentType::Preference => "PREFERENCE",
        ContentType::ProjectKnowledge => "KNOWLEDGE ",
        ContentType::Correction => "CORRECTION",
        _ => "OTHER     ",
    }
}
