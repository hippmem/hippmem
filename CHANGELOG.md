# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

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
