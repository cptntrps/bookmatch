#!/usr/bin/env bash
# deploy.sh — bookmatch deploys IN-PLACE: the systemd unit (bookmatch.service.template)
# runs this checkout directly on aurora. Deploy = restart the unit with the merged code,
# then prove health. Fail-loud everywhere: no sudo ticket, failed restart, or a dead
# /api/health all abort non-zero.
set -euo pipefail
cd "$(dirname "$0")"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || { echo "deploy: refusing — live checkout on '$BRANCH', not main" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "deploy: refusing — dirty tree (deploy only known-good merged state)" >&2; exit 1; }
git fetch --quiet origin main || { echo "deploy: refusing — cannot fetch origin/main to verify HEAD" >&2; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || {
    echo "deploy: refusing — HEAD $(git rev-parse --short HEAD) != origin/main $(git rev-parse --short origin/main) (stale/ahead/diverged checkout)" >&2; exit 1; }

echo "deploy: restarting bookmatch.service @ $(git rev-parse --short HEAD)"
sudo -n systemctl restart bookmatch || { echo "deploy: sudo -n restart failed (need a sudo ticket: run 'sudo -v' first)" >&2; exit 1; }

# Healthy = 200 AND corpus_ready:true — a corpus-less 200 still 503s the primary
# analyze path, so it is NOT a successful deploy (codex finding #2).
for i in $(seq 1 20); do
    sleep 1
    body=$(curl -s --max-time 3 http://localhost:8092/api/health || true)
    if printf '%s' "$body" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("ok") and d.get("corpus_ready") else 1)' 2>/dev/null; then
        echo "deploy: healthy — /api/health ok+corpus_ready after ${i}s"
        exit 0
    fi
done
echo "deploy: FAILED — /api/health never reported ok+corpus_ready within 20s (rollback: sudo systemctl status bookmatch / journalctl -u bookmatch)" >&2
exit 1
