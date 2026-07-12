# hippmem-engine

HIPPMEM native association memory engine — the unified public Rust API orchestration layer, plus the `hippmem` CLI and `hippmem-server` gRPC server.

This is the crate most users want. It wires together the write path, retrieval path, and consolidation into a single `Engine` with seven public methods.

Part of [HIPPMEM](https://github.com/hippmem/hippmem) — a native associative memory engine for AI agents, written in Rust.

## Quick start

```bash
cargo add hippmem-engine
```

```rust
use hippmem_engine::Engine;

let engine = Engine::open(config)?;
engine.write(WriteMemoryInput { content: "The user prefers Rust.".into(), ..Default::default() })?;
let results = engine.retrieve(RetrieveInput { query: "what does the user prefer?".into(), k: 3, ..Default::default() })?;
```

Runs fully offline with the deterministic fallback backend — no GPU, API key, or network required.

## Documentation

- [Project README](https://github.com/hippmem/hippmem#readme)
- [Quick start (5 min)](https://github.com/hippmem/hippmem/blob/main/docs/quickstart.md)
- [User guide](https://github.com/hippmem/hippmem/blob/main/docs/user-guide.md)
- [API reference](https://github.com/hippmem/hippmem/blob/main/docs/api-reference.md)
- API docs: <https://docs.rs/hippmem-engine>

## License

AGPL-3.0-only — see [COPYRIGHT](https://github.com/hippmem/hippmem/blob/main/COPYRIGHT) for the full two-tier licensing model. A commercial license is available for use cases incompatible with AGPL-3.0-only.
