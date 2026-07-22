# Book Match

Add a Goodreads account → run analysis → get a taste profile + grounded book
recommendations → chat with an agent about what that person would like.

Recreates the June-2026 Goodreads "impossible insights" rec-engine as a MyApps app.
Full design: [DESIGN.md](DESIGN.md).

## Layout
- `bookmatch/` — app package (analyze, neighbors, agent, app, storage).
- `scripts/build_index.py` — one-time UCSD Goodreads dataset download + index build.
- `static/` — dark-theme SPA.
- `tests/` — unit + integration (fixtures from `~/livingos/data/scratch/`).
- `deploy.sh` — in-place deploy: restart `bookmatch.service`, prove health, roll back
  to last-good on failure.
- `bookmatch.service.template` — systemd unit (aurora `:8092`, secrets via sops drop-in).
- `.githooks/` + `scripts/install_hooks.sh` — pre-push test gate (ADR-0020); run
  `scripts/install_hooks.sh` once per fresh clone.

## Run (dev)
```
pip install -r requirements.txt
python3 scripts/build_index.py          # one-time, multi-GB, → ~/data/goodreads-ucsd
uvicorn bookmatch.app:app --port 8092
```

Depends on aurora-local services: Ollama (`nomic-embed-text`), livingos-api gateway
(`:8510`), Postgres (`:5435`).

## Deploy (aurora)
Runs live as `bookmatch.service` on aurora `:8092` — the unit executes this checkout
in place. `./deploy.sh` refuses anything but a clean `main` at `origin/main`, restarts
the unit (needs a sudo ticket: `sudo -v`), and requires `/api/health` to report
`ok` + `corpus_ready`; a failed health check rolls back to the last known-good
revision and still exits non-zero. Secrets come from the sops drop-in
(`bookmatch.service.d/sops.conf` → `/run/bookmatch/secrets.env`, ADR-0021); there is
no plaintext `.env`. CI (`.github/workflows/tests.yml`) runs the pytest suite on
push/PR; the pre-push hook runs the same suite locally.
