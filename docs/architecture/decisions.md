# Architecture Decision Records (ADR) Index

> This index summarizes the key technical decisions that shape HIPPMEM. Each entry is a one-line summary with its ADR number; the full rationale, rejected alternatives, and impact scope are recorded in the internal decision log.

HIPPMEM's technology choices favor pure-Rust libraries (no C/C++ build dependencies) so that CI builds cleanly in a fresh environment. Any new dependency or change to an existing selection requires a new ADR before introduction.

| ADR | Decision | Summary |
|-----|----------|---------|
| 001 | Embedded KV store: **redb** | Pure-Rust, single-file, MVCC transactions with crash recovery; fits the append-only + overlay layout. Rejected RocksDB (C++ dependency), sled (stagnant), SQLite (model mismatch). |
| 002 | Fulltext index: **Tantivy** | Rust-native, Lucene-grade BM25, embeddable; avoids a second storage engine. Rejected SQLite FTS5. |
| 003 | Vector index: **hnsw_rs** behind a self-owned `VectorIndex` trait | Pure-Rust HNSW for 100K–1M scale; the trait isolates the upper layer so quantization/faiss can be swapped later. |
| 004 | Signatures and similarity: self-owned signatures + **simsimd** for distance | `SemanticSignature` (SimHash / binary code / MinHash) is self-owned product semantics; only the distance operator is borrowed. |
| 005 | Serialization: **serde + bincode** (storage) / **serde_json** (diagnostics, eval) | bincode is compact and fast for hot paths; JSON is human-readable for diagnostics. Schema versioning via a `u16` field. |
| 006 | ID generation: **ULID**, newtype `MemoryId(u128)` | Monotonic, time-prefixed, lexicographically sortable; no central coordination. Time and random parts come from injectable `Clock`/`Rng`. |
| 007 | Time: `Timestamp(i64)` Unix milliseconds + **time** crate + injectable `Clock` | i64 ms is convenient for indexing, decay, and temporal bucketing; `time` is smaller than `chrono`; logic-layer "now" always comes from `Clock`. |
| 008 | Configuration: **TOML + figment** (defaults < file < env) | TOML is the Rust default; figment's multi-source merge supports "CI uses fallback / local switches to real backend." |
| 009 | Error handling: **thiserror** (libraries) / **anyhow** (binaries) | Libraries need matchable structured errors; applications need convenient context chains. |
| 010 | Async runtime: **tokio** (multi-thread) | Ecosystem default; drives the background consolidation queue, concurrent multi-channel recall, and async model API calls. |
| 011 | External protocol: in-process Rust API + **tonic** (gRPC) thin wrapper; HTTP gateway is a Non-Goal | The Rust API is primary; gRPC is a thin transport mapping. No HTTP REST gateway. |
| 012 | Model API client: **reqwest** hand-written thin clients | Avoids third-party SDK version/dependency coupling; three endpoints are easy to control, mock, and degrade. Keys read from environment variables. |
| 013 | Association graph: self-owned in-memory adjacency list + redb overlay; no graph database | Spreading-activation traversal, pruning, cycle-breaking, and fan-out control are core product algorithms that must be fully self-owned. |
| 014 | Testing: **cargo test + proptest + insta** | Invariants (serialization round-trip, edge dedup) via property tests; complex structured output (traces, explanation paths) via snapshot tests. |
| 015 | Observability: **tracing + tracing-subscriber** | Structured spans mirror the retrieval pipeline (seed → spread → merge → rerank); `inspect`/`explain` read structured records rather than ad-hoc state. |
| 016 | Workspace: multi-crate cargo workspace | Isolates core/store/model/retrieval/consolidation/engine/eval; shortens compile scope and keeps dependency direction acyclic. |
| 017 | Concurrency data structures: **parking_lot** locks + **dashmap** hot tier | parking_lot is faster and smaller than std locks; dashmap suits the hot tier's concurrent-read-heavy profile. |
| 018 | Tokenization / BM25 text processing: Tantivy tokenizer + **jieba-rs** (Chinese) | English/code uses Tantivy's built-in tokenizer; Chinese uses jieba-rs. Language is selected per `MemoryContent.language`. |
| 019 | Key hash: **xxhash (xxh3, 64-bit)**, fixed algorithm and seed (0) | Keys enter persistent indexes, so a single fixed algorithm is mandated for cross-version data compatibility. Rejected ahash (not cross-process stable). |
| 020 | CLI argument parsing: **clap** derive | Rust CLI de facto standard; derive macros reduce boilerplate, auto-generate `--help`, support env-variable fallback. CLI-only dependency, not in the library tree. |
| 021 | gRPC code generation: **tonic-build + prost** | Replaces hand-written proto mappings; `build.rs` compiles `.proto` into strongly typed code validated at compile time. |
| 022 | Structured logging: **tracing-subscriber** | Official companion to `tracing`; supports `RUST_LOG` level control and JSON output; integrates seamlessly with tokio. |
