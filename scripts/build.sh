#!/usr/bin/env bash
# ── HIPPMEM build wrapper: sweep stale artifacts before building ──
#
# Usage:
#   bash scripts/build.sh              # = cleanup + cargo build
#   bash scripts/build.sh test         # = cleanup + cargo test
#   bash scripts/build.sh clippy       # = cleanup + cargo clippy
#   bash scripts/build.sh fmt --check  # = cleanup + cargo fmt --check
#
# Requires: cargo-sweep (install: cargo install cargo-sweep)
#
# Cleanup strategy:
#   1. Remove incremental compilation dirs older than N days (cargo-sweep
#      doesn't touch these — it uses profile-matching, not age, for debug/).
#   2. Run cargo-sweep --time for everything else (deps, examples, etc.).
#   3. If still over MAXSIZE, aggressive size-based sweep.

set -euo pipefail

SWEEP_DAYS="${HIPPMEM_SWEEP_DAYS:-3}"
MAXSIZE="${HIPPMEM_MAXSIZE:-5GB}"

# ── 1. stale incremental compilation artifacts ──
if [ -d target/debug/incremental ]; then
    STALE_INCR=$(find target/debug/incremental -maxdepth 2 -type d -mtime "+${SWEEP_DAYS}" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$STALE_INCR" -gt 0 ]; then
        echo "🧹 removing ${STALE_INCR} stale incremental dirs (>${SWEEP_DAYS}d)..."
        find target/debug/incremental -maxdepth 2 -type d -mtime "+${SWEEP_DAYS}" -exec rm -rf {} + 2>/dev/null || true
    fi
fi

# ── 2. cargo-sweep for regular artifacts ──
if command -v cargo-sweep &>/dev/null; then
    echo "🧹 cargo sweep --time ${SWEEP_DAYS}..."
    cargo sweep --time "$SWEEP_DAYS" 2>&1 | tail -1 || true
else
    echo "⚠️  cargo-sweep not installed. Run: cargo install cargo-sweep"
fi

# ── 3. hard size cap ──
if command -v cargo-sweep &>/dev/null && [ -d target ]; then
    SIZE=$(du -sm target 2>/dev/null | cut -f1)
    if [ "${SIZE:-0}" -gt 10000 ]; then
        echo "🧹 target/ at ${SIZE}MB, capping at ${MAXSIZE}..."
        cargo sweep --maxsize "$MAXSIZE" 2>&1 | tail -1 || true
    fi
fi

if [ $# -eq 0 ]; then
    set -- build
fi

echo "▶ cargo $*"
cargo "$@"
