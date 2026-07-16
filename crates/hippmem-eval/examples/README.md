# HIPPMEM eval examples

Diagnostic and profiling tools for the evaluation framework.

## diagnose_p1_miss

Deep-dive diagnostic for P@1 (Precision@1) misses. Analyzes channel score
distribution to understand why certain queries fail to rank a correct answer
at position 1.

```bash
cargo run -p hippmem-eval --example diagnose_p1_miss --features api-backends
```

Requires `OPENAI_API_KEY` (or compatible) for the embedding backend.

## write_perf_profile

Micro-benchmark for the write pipeline. Measures per-phase timing
(extraction, tokenization) across locales.

```bash
cargo run -p hippmem-eval --example write_perf_profile --release
```

No network dependency — uses the deterministic fallback backend.
