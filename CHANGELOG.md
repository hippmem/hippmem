# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.2.1] — 2026-08-07

### Added
- `RetrieveOutput` now exposes a `retrieval_id` so callers can link an
  `Engine::feedback` record back to the retrieval that produced it.

[0.2.1]: https://github.com/hippmem/hippmem/releases/tag/v0.2.1

## [0.2.0] — 2026-07-30

### Changed
- **Embedder naming**: `Deterministic` → **`Hash`** (256d SimHash, offline default),
  `OpenAiCompatible` → **`Neural`** (API-based, higher semantic accuracy).
- **No more feature flags**: both embedders are always compiled. Choose at runtime
  via config (`provider = "hash" | "neural"`), env vars, or CLI (`--embedding-provider`).
- CLI default model: `text-embedding-v4` (DashScope) → `text-embedding-3-small` (OpenAI).
- All user documentation updated to the Hash/Neural terminology (EN + ZH).

### Fixed
- Removed dead-code `#[cfg(feature = "api-backends")]` guards; API-dependent tests
  now skip at runtime when `OPENAI_API_KEY` is missing.
- Mock HTTP test now performs a real round-trip against a local server.

### Added
- `hippmem-eval` bumped to 0.2.0 (aligned MINOR with the engine).

[0.2.0]: https://github.com/hippmem/hippmem/releases/tag/v0.2.0

## [0.1.0] — 2026-07-12

Initial public release of HIPPMEM.

### Added
- Native association memory engine with write-time association discovery.
- Multi-channel recall: BM25 + entity + semantic + temporal + topic + graph.
- Spreading activation retrieval with explanation traces.
- Hebbian consolidation, decay, and compaction.
- RRF (Reciprocal Rank Fusion) channel fusion.
- Deterministic fallback backend — fully offline, zero external API dependencies.
- gRPC server and CLI.
- Evaluation framework with 10 task types across 50+ test corpora.
- Tiered licensing: Apache 2.0 (infrastructure crates) / AGPL-3.0-only (algorithm + product crates).
- Commercial license option for proprietary use cases.

[0.1.0]: https://github.com/hippmem/hippmem/releases/tag/v0.1.0
