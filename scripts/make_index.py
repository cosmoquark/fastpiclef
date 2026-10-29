#!/usr/bin/env python3
"""Generate a static PEP 503 "simple" index for the built artifacts.

The index is plain HTML (no server): GitHub Pages serves it, and the links
point at the release assets, so no wheel is committed to git.  Collaborators
then need only

    [[tool.uv.index]]
    url = "https://cosmoquark.github.io/fastpiclef/simple"

once, and `uv add fastpiclef` / `uv lock --upgrade` resolve versions normally.

    python scripts/make_index.py --tag v0.1.0

Reads dist/*.whl and dist/*.tar.gz, writes docs/simple/{index.html,
fastpiclef/index.html} with sha256 fragments.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import pathlib
import sys

PROJECT = "fastpiclef"
DEFAULT_BASE = "https://github.com/cosmoquark/fastpiclef/releases/download"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="release tag, e.g. v0.1.0")
    ap.add_argument("--dist", default="dist", type=pathlib.Path)
    ap.add_argument("--out", default="docs/simple", type=pathlib.Path)
    ap.add_argument("--base-url", default=DEFAULT_BASE,
                    help="release download URL prefix (no trailing slash)")
    args = ap.parse_args()

    artifacts = sorted(
        p for p in args.dist.iterdir()
        if p.is_file() and (p.suffix == ".whl" or p.name.endswith(".tar.gz"))
    )
    if not artifacts:
        print(f"error: no wheels or sdists in {args.dist}/", file=sys.stderr)
        return 1

    # PEP 503: normalized project name.
    name = PROJECT.lower().replace("_", "-").replace(".", "-")
    project_dir = args.out / name
    project_dir.mkdir(parents=True, exist_ok=True)

    links = []
    for p in artifacts:
        url = f"{args.base_url}/{args.tag}/{p.name}"
        links.append(f'    <a href="{html.escape(url)}#sha256={sha256(p)}">'
                     f"{html.escape(p.name)}</a><br/>")
        print(f"  {p.name}  sha256={sha256(p)[:16]}…")

    (project_dir / "index.html").write_text(
        "<!DOCTYPE html>\n<html><body>\n"
        f"<h1>{html.escape(name)}</h1>\n" + "\n".join(links) + "\n</body></html>\n"
    )
    (args.out / "index.html").write_text(
        "<!DOCTYPE html>\n<html><body>\n"
        f'<a href="{html.escape(name)}/">{html.escape(name)}</a><br/>\n'
        "</body></html>\n"
    )
    print(f"wrote {args.out}/index.html and {project_dir}/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
