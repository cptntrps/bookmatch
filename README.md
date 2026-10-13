# Book Match

> **Documentation:** `docs/` carries the generated six-doc set (scope, architecture, workflows, data, quality, internals) — read it before reading code. Do not edit; it regenerates nightly.


Add a Goodreads account → run analysis → get a taste profile + grounded book
recommendations → veto the ones you do not want → chat with an agent about what
that person would like.

Recreates the June-2026 Goodreads "impossible insights" rec-engine as a MyApps app.
Full design: [DESIGN.md](DESIGN.md).

## Layout
- `bookmatch/` — app package (analyze, neighbors, agent, app, storage).
- `scripts/build_index.py` — one-time UCSD Goodreads dataset download + index build.
- `sql/v_bookmatch_correctness.sql` — `books.v_bookmatch_correctness`, the ADR-0071
  correctness surface: per account, asserts the pipeline produced a *grounded* result
  (shelf ingested, taste axes complete, UCSD corpus matched, non-empty recs + profile).
  Health is not modelled there — that stays the live systemd + `/api/health` probe.
- `capability.yaml` — capability manifest (`cap.bookmatch-recommend`, ADR-0095),
  crawled into `registry.capabilities`.
- `static/` — dark-theme SPA.
- `tests/` — unit + integration (fixtures from `~/livingos/data/scratch/`).
- `deploy.sh` — in-place deploy: restart `bookmatch.service`, prove health, roll back
  to last-good on failure.
- `bookmatch.service.template` — systemd unit (aurora `:8092`, secrets via sops drop-in).
- `.githooks/` + `scripts/install_hooks.sh` — pre-push test gate (ADR-0020); run
  `scripts/install_hooks.sh` once per fresh clone.

## Run (dev)
Keep the platform `livingos-api` checkout at `$HOME/src/products/livingos-api`.
The requirements install includes its pinned private client wheel automatically.

```
pip install -r requirements.txt
python3 -m bookmatch.storage            # one-time, bootstrap/upgrade the books.* schema
python3 scripts/build_index.py          # one-time, multi-GB, → ~/data/goodreads-ucsd
uvicorn bookmatch.app:app --port 8092
```

The schema is never created as a side effect of app start or a test run: both call
`storage.ensure_schema_ready()`, which verifies only and raises if `books.*` is missing.
Run `python3 -m bookmatch.storage` explicitly, as a role that holds CREATE on the
database (`livingos`) — the app and test job roles do not hold it.

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
push/PR; the pre-push hook runs the same suite locally. That hook re-execs itself once
under `gate_capture.sh`, so a local red — which aborts the push before the bare remote
sees the ref — still records its verdict in `ops.gate_run`. The wrapper returns the
wrapped exit code unchanged, and degrades to an unwrapped run if the recorder is absent.

## Shared capability dependency

With the platform checkout at `$HOME/src/products/livingos-api`, the normal
`python -m pip install -r requirements.txt` installs `livingos-client` 0.1.0
from its bundled wheel and verifies its SHA-256. Run pip with the interpreter
that runs this consumer. No separate wheel
copy, cache setup or client-install command is needed. A fresh machine or CI
job checks out the private owning repository at that standard path first; the
client is not fetched from public PyPI. The wheel has no third-party dependencies.
