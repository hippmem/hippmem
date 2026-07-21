# HIPPMEM

[![CI](https://github.com/hippmem/hippmem/actions/workflows/ci.yml/badge.svg)](https://github.com/hippmem/hippmem/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/hippmem-engine.svg)](https://crates.io/crates/hippmem-engine)
[![Documentation](https://docs.rs/hippmem-engine/badge.svg)](https://docs.rs/hippmem-engine)
[![License](https://img.shields.io/badge/license-AGPL--3.0--only-blue.svg)](LICENSE)

**A native associative memory engine for AI agents, written in Rust.**

HIPPMEM gives an AI agent long-term memory that works like a colleague who remembers context. Instead of storing text chunks and searching them by vector similarity, it discovers associations between memories at write time and retrieves them via spreading activation at read time — so the agent recalls not just *what* was said, but *how things connect* and *why*.

It runs fully offline with a deterministic fallback backend (no GPU, no API key, no network required), yet plugs into OpenAI / Anthropic for higher semantic accuracy when available.

---

## What is HIPPMEM?

Most memory solutions today are either vector databases (store embeddings, search by similarity) or RAG pipelines (stitch a retriever onto an LLM). Both guess relationships at query time, after the write-time context is already lost.

HIPPMEM takes a different approach: **associations are discovered when a memory is written**, while the full context — conversation, entities, goals, causality — is available. Retrieval then spreads activation from seed memories along typed association edges, surfacing a connected context network rather than a bag of similar texts. The graph also evolves with use: frequently co-activated connections strengthen (Hebbian learning), and stale ones decay.

The result is memory with continuity — closer to a long-term collaborator than a search box.

---

## Key Features

- **Write-time association discovery** — extracts entities, topics, events, goals, decisions, and causal links on the fly, scores candidate associations across multiple dimensions, and builds typed edges into a native graph.
- **Spreading activation retrieval** — multi-channel seed recall (BM25 + entity + semantic + temporal + topic + graph) fused by RRF, then spreading activation over the association graph with cycle elimination and energy pruning.
- **Explanation traces** — every result explains *why* it was recalled, via `activation_trace` + `matched_dimensions` + warnings.
- **Hebbian evolution** — co-activated connections strengthen automatically; long-unused weak edges decay; compaction and summarization keep the graph healthy.
- **Deterministic fallback backend** — the entire pipeline (embedding, extraction, rerank, summarization) works offline with zero external API calls.
- **Single-file storage** — all state in one redb file plus a Tantivy full-text directory and HNSW vector index. No external database.
- **Rust library + CLI + gRPC server** — one core, three deployment shapes.

---

## Quick Start

```bash
git clone https://github.com/hippmem/hippmem.git
cd hippmem
cargo build

# Write your first memory
cargo run --bin hippmem -- write \
  -c "The user is a software engineer who prefers Rust." -t Preference

# Retrieve
cargo run --bin hippmem -- retrieve -q "what does the user prefer?" -k 3

# Explain why a memory was recalled (replace <memory_id> with the id from above)
cargo run --bin hippmem -- explain -m <memory_id>
```

No GPU, API key, or network connection is needed — the deterministic fallback backend handles everything by default.

For a 5-minute walkthrough, see the [Quick Start guide](docs/quickstart.md).

### Run a demo in 10 seconds

```bash
# See the engine in action — cross-session associative memory
cargo run --example project_memory

# Minimal library usage (1 write + 1 retrieve)
cargo run --example basic_usage
```

No API key, no network, no GPU required.

---

## Architecture Overview

```
Write Path            Storage              Retrieval Path
────────────          ───────              ──────────────
write(content)                            multi-channel seed recall
  │ extract keys       ┌──────────────┐    (BM25 / entity / semantic /
  │ discover    ──────▶│  redb file   │◀──  temporal / topic / graph)
  │  candidates        │  + Tantivy   │          │
  │ score & build      │  + HNSW      │          ▼
  │  edges             │  + graph     │   spreading activation
  ▼ enrich stage       └──────────────┘          │
MemoryUnit                                       ▼
                                          rerank → explain → results
```

HIPPMEM is a Cargo workspace of nine crates with strictly downward, acyclic dependencies. See [docs/architecture/design.md](docs/architecture/design.md) for the full crate topology, and [docs/architecture/](docs/architecture/) for deep dives into each subsystem.

---

## Documentation

| Goal | Start here |
|------|-----------|
| 5-minute quick start | [Quick Start](docs/quickstart.md) |
| Core concepts | [Concepts](docs/concepts.md) |
| Full user guide | [User Guide](docs/user-guide.md) |
| API signatures | [API Reference](docs/api-reference.md) |
| gRPC usage | [gRPC Guide](docs/grpc-guide.md) |
| Copy-paste recipes | [Cookbook](docs/cookbook.md) |
| Configuration & tuning | [Configuration](docs/configuration.md) |
| Compare with other solutions | [Comparison](docs/comparison.md) |
| Integration patterns | [Integration](docs/integration.md) |
| Architecture deep dives | [Architecture docs](docs/architecture/) |

---

## Development

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                         # deterministic fallback, no network
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development setup, commit conventions, and DCO requirements.

---

## License

HIPPMEM uses a two-tier licensing model (see [COPYRIGHT](COPYRIGHT) for the full overview):

- **Apache 2.0** — infrastructure crates (`hippmem-core`, `hippmem-model`, `hippmem-store`).
- **AGPL-3.0-only** — algorithm and product crates (`hippmem-write`, `hippmem-retrieval`, `hippmem-consolidation`, `hippmem-engine`, `hippmem-grpc`, `hippmem-eval`).

A commercial license is available for use cases incompatible with AGPL-3.0-only — contact hippmem@gmail.com.

---

## Contact & Community

- **Issues & PRs**: [github.com/hippmem/hippmem](https://github.com/hippmem/hippmem)
- **Security reports**: see [SECURITY.md](SECURITY.md) — do **not** open a public issue.
- **Conduct**: see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- **Commercial licensing**: hippmem@gmail.com
