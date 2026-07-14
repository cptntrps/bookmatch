#!/usr/bin/env bash
# Activate the tracked git hooks (a committed .githooks/ is inert until hooksPath
# points at it — every fresh clone must run this once, or its pushes silently
# bypass the ADR-0020 test gate).
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
echo "hooks active: core.hooksPath=.githooks ($(git rev-parse --show-toplevel))"
