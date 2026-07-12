# Cookbook

> Scenario recipes: each recipe is a complete solution for "I want to do X", with code you can copy and run directly.
> All recipes use the Engine public API (`hippmem-engine`); no need to understand internal crates.

---

## Recipe 1: Agent Conversation Memory

**Scenario**: An AI Agent needs to remember context across multi-turn conversations. At the start of each turn it retrieves relevant history, and at the end it writes the current turn's content.

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput, RetrieveContext};
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::RetrievalMode;
use hippmem_core::model::unit::WriteContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("memory.redb"),
        ..Default::default()
    })?;

    let session_id = 42u64;

    // ── Round 1 of conversation ──
    // User speaks → first retrieve history (empty on first turn) → write new memory
    let ctx = RetrieveContext {
        session_id: Some(session_id),
        ..Default::default()
    };
    let history = engine.retrieve(RetrieveInput {
        query: "user preferences and background".into(),
        context: ctx,
        top_k: 5,
        max_hops: None,
        retrieval_mode: RetrievalMode::Balanced,
    })?;
    println!("Retrieved {} historical memories", history.results.len());

    engine.write(WriteMemoryInput {
        content: "The user prefers Rust, values clean error handling, and dislikes excessive use of unwrap.".into(),
        content_type: Some(ContentType::Preference),
        context: WriteContext { session_id: Some(session_id), ..Default::default() },
        importance_hint: Some(0.6),
        source_refs: vec![],
    })?;

    // ── Round 2 of conversation ──
    // User asks a new question → retrieves the previous turn's memory
    let history = engine.retrieve(RetrieveInput {
        query: "What are the user's requirements for error handling?".into(),
        context: RetrieveContext { session_id: Some(session_id), ..Default::default() },
        top_k: 3,
        max_hops: None,
        retrieval_mode: RetrievalMode::Balanced,
    })?;

    assert!(!history.results.is_empty());
    for r in &history.results {
        println!("[{:.3}] {}", r.final_score, r.memory.content.raw);
    }

    engine.close()?;
    Ok(())
}
```

**Key tips**:
- Use `session_id` to associate memories from the same session.
- Pass the current session in `RetrieveContext` to gain temporal bias.
- Call `write` at the end of each turn to store new memories.

---

## Recipe 2: Project Knowledge Base

**Scenario**: Record technical decisions and the evolution of a project, so you can trace back "why we chose this" at any time.

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput};
use hippmem_core::model::enums::ContentType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("project.redb"),
        ..Default::default()
    })?;

    // Store project knowledge
    engine.write(WriteMemoryInput {
        content: "Chose Rust as the primary language for the project due to its performance and memory safety guarantees.".into(),
        content_type: Some(ContentType::Decision),
        context: Default::default(),
        importance_hint: Some(0.9),
        source_refs: vec![],
    })?;

    engine.write(WriteMemoryInput {
        content: "Selected redb over RocksDB for the storage layer — redb is pure Rust, compiles easily, and is embedded-friendly.".into(),
        content_type: Some(ContentType::Decision),
        context: Default::default(),
        importance_hint: Some(0.9),
        source_refs: vec![],
    })?;

    engine.write(WriteMemoryInput {
        content: "Using tonic for the gRPC API layer — the team is familiar with the Rust ecosystem and it outperforms REST + JSON.".into(),
        content_type: Some(ContentType::Decision),
        context: Default::default(),
        importance_hint: Some(0.8),
        source_refs: vec![],
    })?;

    // Trace back technical decisions
    let results = engine.retrieve(RetrieveInput {
        query: "Why was redb chosen over RocksDB?".into(),
        top_k: 3,
        ..Default::default()
    })?;

    for r in &results.results {
        println!("[{:.3}] {} (dims: {:?})",
            r.final_score,
            r.memory.content.raw.chars().take(80).collect::<String>(),
            r.matched_dimensions
        );
    }
    // → High scores will hit edges sharing the entities "RocksDB" / "redb"

    engine.close()?;
    Ok(())
}
```

**Key tips**:
- Use `ContentType::Decision` to tag decisions.
- Set `importance_hint` to 0.8+ to ensure important decisions are not decayed.
- The engine automatically discovers shared entities ("Rust", "redb", "tonic") and builds links.

---

## Recipe 3: User Preference Tracking

**Scenario**: Track changes in user preferences over the long term, sensing preference drift and contradictions.

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput};
use hippmem_core::model::enums::ContentType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("prefs.redb"),
        ..Default::default()
    })?;

    // Early preference
    let m1 = engine.write(WriteMemoryInput {
        content: "The user prefers SQLite for local storage because it requires zero configuration.".into(),
        content_type: Some(ContentType::Preference),
        context: Default::default(),
        importance_hint: Some(0.5),
        source_refs: vec![],
    })?;

    // Later changed
    let m2 = engine.write(WriteMemoryInput {
        content: "The user says SQLite concurrency is insufficient and has fully migrated to redb.".into(),
        content_type: Some(ContentType::Preference),
        context: Default::default(),
        importance_hint: Some(0.6),
        source_refs: vec![],
    })?;

    // Query current preference
    let results = engine.retrieve(RetrieveInput {
        query: "What storage does the user use now?".into(),
        top_k: 3,
        ..Default::default()
    })?;

    for r in &results.results {
        println!("[{:.3}] {}", r.final_score, r.memory.content.raw);
        // Check risk warnings
        for w in &r.warnings {
            println!("  ⚠ {:?}", w);
        }
    }

    // Inspect preference evolution
    let explanation = engine.explain(m2.memory_id, None)?;
    println!("New preference linked to {} old memories", explanation.linked.len());

    engine.close()?;
    Ok(())
}
```

**Key tips**:
- Memories of type `Preference` participate in preference evolution tracking.
- Contradiction detection automatically flags inconsistent preferences.
- Use `explain` to inspect the preference drift chain.

---

## Recipe 4: Decision Audit

**Scenario**: Record the full chain of every important decision — what cause → what decision → what outcome → what correction.

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput, FeedbackInput, UsageSignal};
use hippmem_core::model::enums::ContentType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("audit.redb"),
        ..Default::default()
    })?;

    // Problem
    engine.write(WriteMemoryInput {
        content: "Production issue: memory spikes to 4GB within 5 minutes of deployment, then OOM killer triggers.".into(),
        content_type: Some(ContentType::Event),
        context: Default::default(),
        importance_hint: Some(0.9),
        source_refs: vec![],
    })?;

    // Root cause analysis
    engine.write(WriteMemoryInput {
        content: "Root cause: the index is fully loaded into memory at startup. 1M records × 4KB = 4GB. Should switch to lazy loading.".into(),
        content_type: Some(ContentType::Reflection),
        context: Default::default(),
        importance_hint: Some(0.9),
        source_refs: vec![],
    })?;

    // Decision
    let decision = engine.write(WriteMemoryInput {
        content: "Decision: switch index loading to mmap + lazy page faults. Estimated startup memory reduction: 4GB → 200MB.".into(),
        content_type: Some(ContentType::Decision),
        context: Default::default(),
        importance_hint: Some(1.0),
        source_refs: vec![],
    })?;

    // Verify decision outcome
    engine.write(WriteMemoryInput {
        content: "After deploying lazy loading, startup memory stabilized at 180MB. OOM incidents eliminated. Decision validated.".into(),
        content_type: Some(ContentType::Event),
        context: Default::default(),
        importance_hint: Some(0.8),
        source_refs: vec![],
    })?;

    // Feedback: decision succeeded
    engine.feedback(FeedbackInput {
        retrieval_id: 1,
        used_memory_ids: vec![decision.memory_id],
        signal: UsageSignal::TaskSucceeded,
    })?;

    // Trace back the decision chain
    let results = engine.retrieve(RetrieveInput {
        query: "How was the OOM issue resolved? What was the decision process?".into(),
        top_k: 5,
        ..Default::default()
    })?;

    for r in &results.results {
        println!("[{:.3}] {}", r.final_score, r.memory.content.raw);
    }
    // → Should be ordered by causal chain: Event → Reflection → Decision → Event (verification)

    engine.close()?;
    Ok(())
}
```

**Key tips**:
- Event → Reflection → Decision → Event forms a complete audit chain.
- `importance_hint: 1.0` tags critical decisions and protects them from decay.
- `UsageSignal::TaskSucceeded` strengthens associations around correct decisions.

---

## Recipe 5: Offline Deployment (Zero External API Dependencies)

**Scenario**: Run HIPPMEM in a network-isolated environment — no external API calls, only the deterministic degraded backend.

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput, WriteWarning};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("offline.redb"),
        // embedder defaults to Deterministic 256d SimHash, no API needed
        ..Default::default()
    })?;

    // Write — succeeds even without API (carries WriteWarning)
    let output = engine.write(WriteMemoryInput {
        content: "HIPPMEM supports fully offline operation with zero external API dependencies.".into(),
        ..Default::default()
    })?;

    for w in &output.warnings {
        match w {
            WriteWarning::ExtractorDegraded => {
                println!("ℹ Using deterministic rule extraction (non-LLM), basic dimensions still available");
            }
            WriteWarning::StrongDimsDeferred => {
                println!("ℹ Strong semantic dimensions (goal/preference/emotion) deferred");
            }
            _ => println!("⚠ {:?}", w),
        }
    }

    // Retrieve — degraded backend results are still explainable
    let results = engine.retrieve(RetrieveInput {
        query: "Can this work offline?".into(),
        top_k: 3,
        ..Default::default()
    })?;

    assert!(!results.results.is_empty());
    println!("Offline retrieval returned {} results", results.results.len());

    // Diagnostics
    println!("embedder: {} (expected: deterministic_hash)",
        results.diagnostics.backend_used.embedder);

    engine.close()?;
    Ok(())
}
```

**Key tips**:
- `EmbedderConfig::default()` uses the deterministic 256d SimHash degraded backend (default).
- Inspect `WriteWarning` to see which capabilities are degraded.
- In offline deployment, basic retrieval is still available (BM25 + entity + temporal channels).
- The semantic channel uses SimHash instead of dense vectors.

---

## Recipe 6: Memory Hygiene (Periodic Consolidation)

**Scenario**: Run consolidation periodically to keep the memory store healthy — decay weak associations, clean the observation zone, and compress similar memories.

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, ConsolidationScope, InspectQuery};
use hippmem_core::model::enums::ContentType;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("hygiene.redb"),
        ..Default::default()
    })?;

    // Write 20 memories
    for i in 0..20 {
        engine.write(WriteMemoryInput {
            content: format!("Memory #{}: This is test data for consolidation benchmarks.", i),
            content_type: Some(ContentType::UserStatement),
            context: Default::default(),
            importance_hint: if i < 5 { Some(0.1) } else { Some(0.5) },
            source_refs: vec![],
        })?;
    }

    // Before consolidation
    let before = engine.inspect(InspectQuery::StoreStats)?;
    if let hippmem_engine::InspectReport::StoreStats(s) = before {
        println!("Before consolidation: {} memories, {} edges", s.memory_count, s.edge_count);
    }

    // Wait briefly so timestamps differ
    thread::sleep(Duration::from_millis(10));

    // Run incremental consolidation
    let report = engine.consolidate(ConsolidationScope::Incremental)?;
    println!(
        "Consolidation done: processed {} memories, decayed {} edges, archived {} edges, took {}ms",
        report.memories_processed, report.edges_decayed,
        report.edges_archived, report.elapsed_ms
    );

    // After consolidation
    let after = engine.inspect(InspectQuery::StoreStats)?;
    if let hippmem_engine::InspectReport::StoreStats(s) = after {
        println!("After consolidation: {} memories, {} edges", s.memory_count, s.edge_count);
    }

    engine.close()?;
    Ok(())
}
```

**Key tips**:
- `Incremental` is suitable for scheduled jobs (cron / background worker).
- Use `inspect(StoreStats)` before consolidation to understand the current state.
- Memories with low `importance_hint` are more likely to be decayed.
- In production, it is recommended to run `Incremental` once per hour.

---

## Recipe 7: Debug Retrieval with diagnose

**Scenario**: When retrieval results are not satisfactory, use diagnostic mode to locate which stage is causing the problem.

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput};
use hippmem_core::model::links::RetrievalMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let engine = Engine::open(EngineConfig {
        store_dir: dir.path().join("diag.redb"),
        ..Default::default()
    })?;

    // Write some data
    engine.write(WriteMemoryInput {
        content: "The user works in San Francisco and commutes by BART daily.".into(),
        ..Default::default()
    })?;
    engine.write(WriteMemoryInput {
        content: "The user has a cat named Momo, fed twice daily.".into(),
        ..Default::default()
    })?;

    // Retrieve in Diagnostic mode
    let results = engine.retrieve(RetrieveInput {
        query: "What is the user's pet's name?".into(),
        top_k: 5,
        retrieval_mode: RetrievalMode::Diagnostic,  // full diagnostics
        ..Default::default()
    })?;

    // Inspect diagnostics
    let diag = &results.diagnostics;
    println!("Latency: {}ms", diag.latency_ms);
    println!("embedder: {}", diag.backend_used.embedder);
    println!("reranker: {:?}", diag.backend_used.reranker);
    println!("Reranked: {}", if diag.reranked { "yes" } else { "no" });
    println!("Pruned branches: {}", diag.pruned_branches);

    // Per-channel contributions
    for (channel, count) in &diag.channel_contributions {
        println!("  {:?} → {} seeds", channel, count);
    }

    // Inspect activation trace
    println!("Spread {} hops, merged {} nodes",
        results.trace.hops_used, results.trace.merged_count);
    println!("Seed count: {}", results.trace.seeds.len());
    for seed in &results.trace.seeds {
        println!("  {:?} channel, initial energy {:.3}", seed.channel, seed.initial_energy);
    }

    engine.close()?;
    Ok(())
}
```

**Key tips**:
- `RetrievalMode::Diagnostic` returns full diagnostics but is slower.
- Inspect `channel_contributions` to see which channel contributes the most.
- Inspect `pruned_branches` to check for over-pruning.
- Inspect `trace.seeds` to check whether seed recall is sufficient.

---

## More recipes?

The HIPPMEM Cookbook is continuously updated. If you have a real-world use case to share, PRs are welcome.
