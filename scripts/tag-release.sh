#!/usr/bin/env sh
set -eu

if [ "${1:-}" = "" ]; then
  echo "Usage: scripts/tag-release.sh <semver, e.g. 0.1.0>"
  exit 1
fi

VERSION="$1"
TAG="v$VERSION"

if ! echo "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "Version must follow semantic versioning MAJOR.MINOR.PATCH"
  exit 1
fi

git tag "$TAG"
echo "Created tag $TAG"
echo "Push with: git push origin $TAG"
