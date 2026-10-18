#!/usr/bin/env bash
# Local CI: the full release check for this repo (P5).
#
#   bash scripts/ci.sh              # full: examples, release suite, bindings, wrapper tests
#   bash scripts/ci.sh fast         # fast: build + release lib tests + wrapper tests
#   bash scripts/ci.sh rust         # Rust only (no pixi/maturin)
#
# Structure mirrors AGENTS.md: `cargo test --release` is the source of truth
# (~6 min; the batch/macro suites dominate), the Python bindings are checked
# through `pixi run build` (never `cargo build --features python`) and the
# wrapper tests.  The deliberately slow/thorough comparison scripts are *not*
# part of CI; run them explicitly (`pixi run -e comparison <task>`).

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-full}"
step() { printf '\n=== %s ===\n' "$*"; }

step "cargo clippy --release --all-targets -- -D warnings"
cargo clippy --release --all-targets -- -D warnings

step "cargo build --release --examples"
cargo build --release --examples

if [ "$MODE" != "fast" ]; then
    step "cargo test --release (full suite)"
    cargo test --release
else
    step "cargo test --release --lib (fast)"
    cargo test --release --lib
fi

if [ "$MODE" = "rust" ]; then
    step "done (rust mode)"
    exit 0
fi

step "pixi run build (maturin develop into the active env)"
pixi run build

step "python wrapper tests"
pixi run python tests/python/test_wrapper.py

step "python type-stub drift test"
pixi run python tests/python/test_stub.py

if [ "$MODE" != "fast" ]; then
    step "comparison env: bindings rebuilt"
    # Each pixi env carries its own installed fastpiclef (AGENTS gotcha).
    pixi run -e comparison build
fi

step "CI OK ($MODE)"
