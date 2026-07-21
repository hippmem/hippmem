# Quick Start (5 minutes)

> Goal: clone the repo → write a memory → retrieve it → understand why it was retrieved.

## Prerequisites

- Rust 1.95+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Default mode: no GPU, no API key, no network required
- Optional: plug in an OpenAI-compatible API for higher semantic accuracy (see [User Guide §5](user-guide.md#5-choosing-an-embedder-backend))

## Step 1: Clone and build

```bash
git clone https://github.com/hippmem/hippmem.git
cd hippmem
cargo build --release
```

Once built, the binary is at `target/release/hippmem`.

```bash
# Verify
./target/release/hippmem --version
# → hippmem 0.1.0
```

## Step 2: Write a memory

```bash
./target/release/hippmem write \
  -c "The user is a software engineer who prefers Rust and uses redb for embedded storage." \
  -t Preference
```

Expected output:

```
✓ memory_id: 12345678901234567890 stage: Indexed links: 0
```

> If this is the first memory, `links` is 0 (no other memories to associate with). Once you write a second one, edges will be created automatically.

## Step 3: Write a few more memories

```bash
# Write a related fact
./target/release/hippmem write \
  -c "The user has contributed to multiple Rust open-source projects, including CLI tools and database bindings." \
  -t ProjectKnowledge

# Write a decision
./target/release/hippmem write \
  -c "Chose redb over RocksDB because redb is pure Rust, compiles quickly, and works well as an embedded store." \
  -t Decision
```

The engine has now discovered associations automatically — the second memory shares the "Rust" entity with the first, and the third has a causal relationship with the first two.

## Step 4: Retrieve memories

```bash
./target/release/hippmem retrieve -q "Why did the user choose redb?" -k 5
```

Expected output (similar):

```
1. [0.782] Chose redb over RocksDB because redb is pure Rust, compiles quickly... (dims: [Causal, EntityOverlap, SemanticSimilar])
2. [0.543] The user is a software engineer who prefers Rust and uses redb... (dims: [EntityOverlap, SemanticSimilar])
3. [0.421] The user has contributed to multiple Rust open-source projects... (dims: [Temporal, EntityOverlap])
```

Each result carries: a score, a content snippet, and the matched dimensions (telling you "why this was recalled").

## Step 5: Understand why

```bash
# Use the memory_id from Step 2
./target/release/hippmem explain -m 12345678901234567890
```

Expected output:

```
memory: The user is a software engineer... importance: 0.800 links: 2 corrections: 0
```

The engine tells you this memory is linked to 2 other memories, with a current importance of 0.8.

## Step 6: Inspect engine internals

```bash
./target/release/hippmem inspect store-stats
# → memories: 3 edges: 2 backlog: 0

./target/release/hippmem consolidate
# → processed: 3 decayed: 0 elapsed: 0ms
```

## As a Rust library

If you want to call it from code:

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput};

let engine = Engine::open(EngineConfig::default())?;

engine.write(WriteMemoryInput {
    content: "The user prefers Rust for backend development.".into(),
    ..Default::default()
})?;

let results = engine.retrieve(RetrieveInput {
    query: "What language does the user prefer?".into(),
    top_k: 5,
    ..Default::default()
})?;

for r in &results.results {
    println!("[{:.3}] {}", r.final_score, r.memory.content.raw);
}
```

Full example: `crates/hippmem-engine/examples/basic_usage.rs`

```bash
cargo run --example basic_usage
```

### See associative memory in action

```bash
cargo run --example project_memory
```

Simulates an AI coding assistant that remembers a project across 4 sessions.
Shows multi-channel seed recall across 5 channels (BM25 keyword, entity index,
semantic dense + binary, topic clustering), RRF fusion, and why associative
retrieval finds more than keyword search alone.

## Next steps

- [User Guide](user-guide.md) — understand core concepts and the full workflow
- [API Reference](api-reference.md) — method signatures and type definitions
- [Cookbook](cookbook.md) — copy-paste recipes for real scenarios
- [Core Concepts](concepts.md) — go deeper into how the engine "thinks"
