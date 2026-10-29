#!/usr/bin/env bash
# Build the distributable artifacts in dist/: one abi3 manylinux wheel plus the
# sdist.  No docker: `zig` (installed from PyPI into a throwaway venv) supplies
# the old-glibc link target, because this host's glibc is newer than any
# manylinux policy and a native build would record GLIBC_2.44 symbols.
#
#   bash scripts/build-dist.sh
#
# Then upload with `maturin upload dist/*` (TestPyPI first:
# `--repository-url https://test.pypi.org/legacy/`) and collaborators install
# with `uv add fastpiclef`.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

# Release artifacts should come from a tagged, clean checkout: maturin packages
# the working tree, so uncommitted files would ship in the sdist.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "warning: working tree is dirty; artifacts will include uncommitted changes" >&2
fi

# pyo3 0.24 predates Python 3.14; the abi3 build is unaffected.
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "==> preparing build tools (maturin + zig) in a throwaway venv"
uv venv -q "$tmp/venv"
uv pip install -q --python "$tmp/venv/bin/python" maturin ziglang

zig_dir=$("$tmp/venv/bin/python" -c \
    'import pathlib, ziglang; print(pathlib.Path(ziglang.__file__).parent)')
echo "    zig $("$zig_dir/zig" version)"

mkdir -p dist
echo "==> building manylinux wheel (abi3, CPython >= 3.12)"
PATH="$zig_dir:$PATH" "$tmp/venv/bin/maturin" build --release --zig \
    --compatibility manylinux_2_17 --out dist

echo "==> building sdist"
"$tmp/venv/bin/maturin" sdist --out dist

echo
echo "==> artifacts in dist/"
ls -1 dist/
echo
echo "Next: bash scripts/release.sh  (tag + index), or attach dist/* to a release."
echo "Collaborators then use the GitHub Pages index (see README)."
