---
type: design
id: DESIGN-BOOKMATCH
status: draft
agent: double-gui
created: 2026-07-05
project: bookmatch
tags: [goodreads, recommender, hyperbolic, myapps]
---

# Book Match — design

A MyApps app: add a Goodreads account, run analysis, get a taste profile plus
grounded book recommendations, and chat with an embedded agent about which books
that person would like. Recreates the June-2026 "impossible insights" Goodreads
rec-engine (see session `7d1f695b`, scratch toolkit in
`~/livingos/data/scratch/{reader_a,reader_b,gui}_*`) as a durable product.

## Why it works (the seam that makes this cheap)

Goodreads killed its API in 2020, but every **public** profile still exposes a
per-shelf RSS feed — no key, no auth:

    https://www.goodreads.com/review/list_rss/{user_id}?shelf=read&page={n}

Each item carries title, author, the user's rating, the community average,
custom shelves, and a description. That is enough for a full taste model.

The feed also carries, and the importer keeps (ops.actionables:131533), the
user's read date and date added, their written review, the ISBN, the page count,
the publication year, and four cover-image sizes. The ISBN is the load-bearing
one: it gives downstream consumers (Calibre, `reading.texts.goodreads_id`) a
deterministic crosswalk instead of fuzzy title matching.

**`read_at` is a cataloguing date, not a finish date.** Goodreads stamps it when
the shelf entry is marked read, so one bulk-marking session stamps many books
with a single day. Read it as "read by", never "read on".

One book can appear as several Goodreads *editions*. `rss.dedupe_editions`
collapses them into one entry before storage: the edition with a read date wins
(then the higher rating), and the losers' non-empty fields are merged into it, so
an ISBN held only by the losing edition still survives. The collapse is reported
(log line + `n_duplicate_editions` in the analyze response) — it used to happen
silently in the insert's `ON CONFLICT DO NOTHING`.

## Components (each independently testable)

### 1. Dataset builder — `scripts/build_index.py` (one-time, offline)
Downloads the UCSD Goodreads corpus (Wan & McAuley 2018, McAuley lab):
`goodreads_interactions.csv` (user·book·rating) + a book-id → title/author map.
Persists to `~/data/goodreads-ucsd/` (gitignored, multi-GB):
- `user_item.npz` — sparse user×item CSR matrix (876k users).
- `item_users.npz` — inverted item→users index (for fast candidate gather).
- `book_lookup.sqlite` — normalized `title|author → book_id` + `book_id → title/author`.

### 2. Neighbor-gap engine — `bookmatch/neighbors.py` (query-time)
User-based collaborative filtering. The live target is placed into item space by
matching their shelf to `book_lookup`. Only users who **share** items with the
target are scored (via the inverted index — not all 876k). Rank books the nearest
neighbours rate highly that the target does not own: `score = Σ sim(u)·rating(u,b)`.
This is the "books someone next to you loves but you don't have" from June.

### 3. Fetch + analysis — `bookmatch/analyze.py` (reuses scratch logic verbatim)
RSS-fetch shelves → Ollama `nomic-embed-text` (768d) per `"{title} by {author}. {desc}"`.
Derives: the rating-weighted **taste axis** (loved ↔ tolerated), **contrarian**
picks (`myrating − avg`: "she loves it, the world shrugs" and vice-versa), SVD
content axes, the **antipode** of what she loves, and **arbitrary named-axis
projection** — the "make it more french" trick: build an axis from a handful of
concept embeddings, project every book with one dot product, unsupervised.

### 4. Chat agent — `bookmatch/agent.py` (via livingos-api gateway)
`bare-opus`, `generic` scope, OpenAI schema at `http://127.0.0.1:8510/v1`
(mint a key: `python3 -m app.cli.keys mint --name=bookmatch --schemas=openai --scopes=generic`).
System prompt = the taste fingerprint + top neighbor-gap candidates. One live tool,
`project_axis(name)`, runs the dot-product projection over the shelf on demand, so
"show me her most melancholic books" / "what would she like for summer" work
interactively.

### 5. Storage — Postgres `books.*` (aurora :5435)
- `books.accounts` — goodreads user id/url, label, added_at.
- `books.shelf_books` — per account: title, author, myrating, avg, shelves, desc,
  embedding (`vector(768)`), matched book_id, plus the feed metadata: read_at,
  date_added, review, isbn, num_pages, published, cover_url.
- `books.analyses` — cached taste profile + neighbor recs (JSONB), computed_at.
Re-open is instant; multiple accounts persist.

### 6. Web app — `bookmatch/app.py` (FastAPI) + `static/` SPA
Endpoints: `POST /api/accounts` (add), `POST /api/accounts/{id}/analyze`,
`GET /api/accounts/{id}` (profile + recs), `POST /api/accounts/{id}/chat`.
Single dark-theme page matching MyApps style: add-account form, analysis dashboard
(taste axes, contrarian picks, neighbor-gap recs), chat panel with an axis-request
box. Served on aurora, port `8092` (verify free at deploy).

### 7. Deploy — `bookmatch.service` (systemd, aurora) + one MyApps tile
Tile added to the "Apps" group in `~/platform/services/myapps/index.html`.

## Data flow
add account → store → analyze (RSS → embed → match → neighbor-gap → cache) → UI
renders profile + recs → chat (gateway + `project_axis` tool).

## Error handling (loud — no silent fallback)
- Private/invalid profile or empty `read` shelf → explicit 4xx with a clear message.
- **Low match rate** (few shelf books found in corpus) → visible warning banner
  ("18/50 matched; recs are weak"), never silently proceed as if fine.
- Ollama / gateway / Postgres down → surfaced error, never empty recs.

## Testing
Fixtures already on disk from the June session: `reader_a_books.json`,
`reader_b_books.json`, `gui_books.json`.
- Unit: RSS parse (fixture), match normalizer, neighbor-gap on a synthetic matrix,
  arbitrary-axis projection reproduces the "more french" ranking on the reader_a fixture.
- Integration: analyze path writes to and reloads from `books.*`.
- Smoke: end-to-end on Gui's own account through the browser.

## Decisions (approved 2026-07-05)
Full UCSD neighbor-gap (not LLM-only recs); all-at-once, dataset-first; live
`project_axis` tool in chat; `books.*` Postgres storage; `bare-opus` for chat;
aurora-hosted at `platform/services/bookmatch`; app name "Book Match".
