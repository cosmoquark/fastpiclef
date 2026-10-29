#!/usr/bin/env bash
# Cut a release: build the artifacts, regenerate the static index, and (with
# --publish) tag, push and create the GitHub release with the assets attached.
#
#   bash scripts/release.sh            # build + index, print the publish steps
#   bash scripts/release.sh --publish  # also tag, push and create the release
#
# Collaborators then need only, once per project:
#
#   [[tool.uv.index]]
#   url = "https://cosmoquark.github.io/fastpiclef/simple"
#
# (GitHub Pages must serve this repository's docs/ directory; that is a
# one-time repository setting.)
set -euo pipefail
cd "$(dirname "$0")/.."

publish=0
[ "${1:-}" = "--publish" ] && publish=1

version=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')
tag="v$version"

echo "==> fastpiclef $version (tag $tag)"

if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree is dirty; commit first so the artifacts match the tag" >&2
    git status --short >&2
    exit 1
fi
if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    echo "error: tag $tag already exists" >&2
    exit 1
fi

echo "==> building artifacts"
bash scripts/build-dist.sh

echo "==> regenerating the static index"
python3 scripts/make_index.py --tag "$tag"

if [ "$publish" -eq 0 ]; then
    cat <<EOF

Build and index are ready.  To publish:

    git tag -a $tag -m "fastpiclef $version"
    git push origin $tag
    gh release create $tag dist/* --title "fastpiclef $version" --notes "..."

(no gh? create the release in the web UI and attach dist/*, then commit the
regenerated docs/simple/ so the index picks the new version up)

Nothing was pushed or uploaded.
EOF
    exit 0
fi

command -v gh >/dev/null || { echo "error: --publish needs the GitHub CLI (gh)" >&2; exit 1; }
branch=$(git rev-parse --abbrev-ref HEAD)
case "$branch" in
    HEAD|"") echo "error: --publish needs a branch (HEAD is detached)" >&2; exit 1 ;;
esac

echo "==> tagging and pushing $tag"
git tag -a "$tag" -m "fastpiclef $version"
git push origin "$tag"

echo "==> creating the GitHub release with dist/* attached"
gh release create "$tag" dist/* --title "fastpiclef $version" \
    --notes "See the README for install instructions."

echo "==> committing the regenerated index (docs/simple)"
git add docs/simple
git commit -m "docs: index fastpiclef $version"
git push origin "$branch"

echo "done: https://github.com/cosmoquark/fastpiclef/releases/tag/$tag"
