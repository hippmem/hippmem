# Contributing to HIPPMEM

Thanks for your interest in contributing to HIPPMEM! This document explains how to report issues, submit changes, and follow the project conventions.

## Reporting Bugs

- Open an issue at <https://github.com/hippmem/hippmem/issues>.
- Include the HIPPMEM version (`cargo run --bin hippmem -- --version`), Rust version (`rustc --version`), OS, and a minimal reproduction.
- For security vulnerabilities, do **not** open a public issue — see [SECURITY.md](SECURITY.md).

## Submitting a Pull Request

1. Fork the repository.
2. Create a branch from `main`:
   ```bash
   git checkout -b my-fix
   ```
3. Make your changes. Keep PRs focused — one logical change per PR.
4. Ensure all checks pass (see [Development Setup](#development-setup)).
5. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add fuzzy entity matching
   fix: correct RRF weight for temporal channel
   docs: clarify spreading activation formula
   refactor: extract seed normalization
   test: add corpus for multilingual retrieval
   chore: bump dependencies
   ```
6. Open a PR against `main` and describe the change, the motivation, and any trade-offs.

## Development Setup

HIPPMEM is a Cargo workspace. Rust 1.95+ is required.

```bash
# Build everything
cargo build --workspace

# Run the full test suite (uses the deterministic fallback backend, no network)
cargo test --workspace

# Build examples
cargo build --workspace --examples

# Build the CLI and gRPC server binaries
cargo build --bin hippmem
cargo build --bin hippmem-server
```

No GPU, API key, or network connection is needed for the default build or tests. The deterministic fallback backend provides full offline coverage.

### Optional: API backends

To exercise the OpenAI/Anthropic backends (feature-gated, not required for CI):

```bash
cargo test --workspace --features api-backends
```

## Code Style

Format and lint before pushing:

```bash
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
```

CI enforces both. Treat warnings as errors.

Conventions:
- Library code must not `unwrap`/`panic` on recoverable errors — return `Result`.
- Do not use `unsafe` without an approved ADR.
- Do not call `SystemTime::now()` or global RNG directly in library logic — use the injected `Clock`/`Rng` traits.
- Code comments and documentation are in English.
- Match the surrounding code's naming, density, and idioms.

## Commit Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short imperative summary>

<optional body explaining why and what trade-offs>

<optional footer>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.

## Licensing

HIPPMEM uses a two-tier licensing model (see [COPYRIGHT](COPYRIGHT) for the full overview):

- **Apache 2.0** — infrastructure crates (`hippmem-core`, `hippmem-model`, `hippmem-store`).
- **AGPL-3.0-only** — algorithm and product crates (`hippmem-write`, `hippmem-retrieval`, `hippmem-consolidation`, `hippmem-engine`, `hippmem-grpc`, `hippmem-eval`).

By contributing, you agree that your contributions will be licensed under the same terms as the crate they touch. A commercial license is available for use cases incompatible with AGPL-3.0-only — contact hippmem@gmail.com.

## DCO (Developer Certificate of Origin)

Every commit must include a `Signed-off-by:` line certifying that you have the right to submit the contribution:

```
feat: add fuzzy entity matching

Signed-off-by: Your Name <you@example.com>
```

Add the line manually to your commit message, or use `git commit -s` to append it automatically. By signing off, you attest to the [Developer Certificate of Origin](https://developercertificate.org/) v1.1.
