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

echo "deploy: restarting bookmatch.service @ $(git rev-parse --short HEAD)"
sudo -n systemctl restart bookmatch || { echo "deploy: sudo -n restart failed (need a sudo ticket: run 'sudo -v' first)" >&2; exit 1; }

for i in $(seq 1 20); do
    sleep 1
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8092/api/health || true)
    if [ "$code" = "200" ]; then
        echo "deploy: healthy — /api/health 200 after ${i}s"
        exit 0
    fi
done
echo "deploy: FAILED — /api/health never returned 200 within 20s (rollback: sudo systemctl status bookmatch / journalctl -u bookmatch)" >&2
exit 1
