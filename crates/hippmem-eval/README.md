# hippmem-eval

Evaluation framework for HIPPMEM — corpus loading, baseline comparison, metric computation, and threshold verification.

Part of [HIPPMEM](https://github.com/hippmem/hippmem) — a native associative memory engine for AI agents, written in Rust.

## What this crate provides

- **Corpus types** — `EvalCase`, `EvalWrite`, `GroundTruth`, etc. for deserializing eval cases from JSON fixtures
- **Bench corpus types** — `BenchDataset`, `CategoryQuerySet`, `CategoryTextSet`, etc. for benchmark datasets
- **Five baseline systems** — BM25-only, embedding-only, hybrid, RAG summary memory, and HIPPMEM full
- **Metric computation** — Recall@K, Precision@K, explanation accuracy, contradiction awareness
- **Runner** — `run_case()` and `run_suite()` for executing eval cases against any baseline

## Fixture structure

```
fixtures/
├── corpus/                # Eval corpus (10 task types, 53 cases each)
│   ├── en/                # English-language cases
│   └── zh/                # Chinese-language cases
└── bench/                 # Benchmark datasets
    ├── en/                # English-language benchmarks
    └── zh/                # Chinese-language benchmarks
```

See each directory's README for details on the corpus format and task types.

## Documentation

- [Project README](https://github.com/hippmem/hippmem#readme)
- [Architecture: eval framework](https://github.com/hippmem/hippmem/blob/main/docs/architecture/design.md)

## License

AGPL-3.0-only — see [COPYRIGHT](https://github.com/hippmem/hippmem/blob/main/COPYRIGHT) for the full two-tier licensing model.
Commercial licenses available: hippmem@gmail.com
