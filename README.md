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

## Run (dev)
```
pip install -r requirements.txt
python3 scripts/build_index.py          # one-time, multi-GB, → ~/data/goodreads-ucsd
uvicorn bookmatch.app:app --port 8092
```

Depends on aurora-local services: Ollama (`nomic-embed-text`), livingos-api gateway
(`:8510`), Postgres (`:5435`).
