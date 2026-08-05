#!/usr/bin/env bash
# Extract a version's changelog section from CHANGELOG.md for GitHub Release notes.
#
# Usage: bash scripts/release-notes.sh <version> [changelog-file]
#   version: e.g. "0.2.0" (matches "## [0.2.0]" header)
#   changelog-file: default CHANGELOG.md
#
# Output goes to stdout; pipe to a file for gh release create --notes-file.

set -euo pipefail

VERSION="${1:?usage: release-notes.sh <version> [changelog-file]}"
CHANGELOG="${2:-CHANGELOG.md}"

awk -v v="$VERSION" '
  $0 ~ "^## \\[" v "\\]" { found = 1; next }
  found && $0 ~ "^## \\[" { exit }
  found && $0 ~ /^\[.*\]:/ { next }   # skip changelog footer link lines like "[0.2.0]: url"
  found { print }
' "$CHANGELOG"
