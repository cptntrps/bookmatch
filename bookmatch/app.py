"""Book Match FastAPI app — add a Goodreads account, analyze, chat.

Loads the UCSD corpus once at startup (lifespan) and holds it for the process.
Embeddings, chat model and Postgres are configured by environment; defaults are local.
"""
from __future__ import annotations
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# load .env (gateway key etc.) if present — before importing modules that read env
_ENV = pathlib.Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from . import analyze, embed, storage       # noqa: E402
from .agent import BookAgent                 # noqa: E402
from .corpus import Corpus                   # noqa: E402
from .match import normkey                   # noqa: E402
from .rss import fetch_shelf, parse_user_id  # noqa: E402

STATIC = pathlib.Path(__file__).resolve().parent.parent / "static"
corpus = Corpus()


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.ensure_schema_ready()
    if corpus.ready:
        corpus.load()  # ~1-2s: mmap npz + tocsc + open sqlite
    yield


app = FastAPI(title="Book Match", lifespan=lifespan)


class AddAccount(BaseModel):
    url_or_id: str
    label: str = ""


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []


class VetoIn(BaseModel):
    title: str
    author: str = ""


class AxisIn(BaseModel):
    label: str = "axis"
    positive: list[str]
    negative: list[str] = []


def _drop_vetoed(recs: list[dict], vetoed: set[str]) -> list[dict]:
    return [r for r in recs if normkey(r["title"], r.get("author") or "") not in vetoed]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "corpus_ready": corpus.ready,
            "corpus": corpus.meta if corpus.ready else None}


@app.get("/api/accounts")
def accounts():
    with storage.connect() as conn:
        return storage.list_accounts(conn)


@app.post("/api/accounts")
def add_account(body: AddAccount):
    try:
        user_id = parse_user_id(body.url_or_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    with storage.connect() as conn:
        aid = storage.upsert_account(conn, user_id, body.label or f"user {user_id}",
                                     body.url_or_id)
    return {"id": aid, "user_id": user_id}


@app.post("/api/accounts/{account_id}/analyze")
def analyze_account(account_id: int):
    if not corpus.ready:
        raise HTTPException(503, "recommendation corpus not built yet")
    with storage.connect() as conn:
        accts = {a["id"]: a for a in storage.list_accounts(conn)}
        if account_id not in accts:
            raise HTTPException(404, "unknown account")
        acct = accts[account_id]
        dropped_editions: list[dict] = []
        try:
            books = fetch_shelf(acct["url"] or acct["user_id"],
                                dropped_out=dropped_editions)
        except ValueError as e:
            raise HTTPException(400, str(e))       # private/empty profile — fail loud

        E = embed.embed_books(books)
        result = corpus.recommend_for_shelf(books)
        matched_by_key = {(m.title, m.author): m.book_id for m in result.matched}
        matched_col = [matched_by_key.get((b["title"], b.get("author")))
                       for b in books]
        profile = analyze.build_profile(books, E)

        storage.save_shelf(conn, account_id, books, E, matched_col)
        storage.save_analysis(conn, account_id, profile, result.recs, result.match_rate)
        # vetoes outlive this re-analysis — a book disliked before must not resurface
        # just because a fresh corpus pass re-derives it (goal of row 95834)
        recs_out = _drop_vetoed(result.recs, storage.vetoed_keys(conn, account_id))

    return {"account_id": account_id, "match_rate": round(result.match_rate, 3),
            "n_matched": len(result.matched), "n_unmatched": len(result.unmatched),
            "n_books": len(books),
            # the shelf is one row per BOOK; duplicate Goodreads editions are merged,
            # never silently dropped (ops.actionables:131533)
            "n_duplicate_editions": len(dropped_editions),
            "profile": profile, "recs": recs_out,
            "warning": (None if result.match_rate >= 0.4 else
                        f"only {len(result.matched)}/{len(books)} books matched the "
                        "corpus — recommendations are weak")}


@app.get("/api/accounts/{account_id}")
def get_account(account_id: int):
    with storage.connect() as conn:
        accts = {a["id"]: a for a in storage.list_accounts(conn)}
        if account_id not in accts:
            raise HTTPException(404, "unknown account")
        analysis = storage.get_analysis(conn, account_id)
        if analysis and analysis.get("recs"):
            vetoed = storage.vetoed_keys(conn, account_id)
            analysis = {**analysis, "recs": _drop_vetoed(analysis["recs"], vetoed)}
    return {"account": accts[account_id], "analysis": analysis}


@app.post("/api/accounts/{account_id}/veto")
def veto_book(account_id: int, body: VetoIn):
    with storage.connect() as conn:
        accts = {a["id"]: a for a in storage.list_accounts(conn)}
        if account_id not in accts:
            raise HTTPException(404, "unknown account")
        storage.add_veto(conn, account_id, body.title, body.author)
    return {"ok": True}


@app.delete("/api/accounts/{account_id}/veto")
def unveto_book(account_id: int, body: VetoIn):
    with storage.connect() as conn:
        accts = {a["id"]: a for a in storage.list_accounts(conn)}
        if account_id not in accts:
            raise HTTPException(404, "unknown account")
        storage.remove_veto(conn, account_id, body.title, body.author)
    return {"ok": True}


@app.post("/api/accounts/{account_id}/axis")
def axis(account_id: int, body: AxisIn):
    """Project the shelf and the current candidates along a named concept axis.
    The same routine the chat agent's tool uses; the UI draws the result."""
    if not [p for p in body.positive if p.strip()]:
        raise HTTPException(400, "give at least one phrase for the positive pole")
    with storage.connect() as conn:
        shelf = storage.load_shelf(conn, account_id)
        analysis = storage.get_analysis(conn, account_id)
        vetoed = storage.vetoed_keys(conn, account_id) if analysis else set()
    if shelf is None or not shelf.embeddings.size:
        raise HTTPException(409, "analyze this account before projecting an axis")
    recs = _drop_vetoed(analysis["recs"], vetoed) if analysis and analysis.get("recs") else []
    pos = embed.embed_terms([p for p in body.positive if p.strip()])
    negs = [n for n in body.negative if n.strip()]
    neg = embed.embed_terms(negs) if negs else None
    C = embed.embed_books(recs) if recs else None
    return analyze.axis_positions(shelf.books, shelf.embeddings, pos, neg, body.label, recs, C)


@app.post("/api/accounts/{account_id}/chat")
def chat(account_id: int, body: ChatIn):
    with storage.connect() as conn:
        analysis = storage.get_analysis(conn, account_id)
        shelf = storage.load_shelf(conn, account_id)
        accts = {a["id"]: a for a in storage.list_accounts(conn)}
        vetoed = storage.vetoed_keys(conn, account_id) if analysis else set()
    if analysis is None or shelf is None:
        raise HTTPException(409, "analyze this account before chatting")
    agent = BookAgent(shelf.books, shelf.embeddings, analysis["profile"],
                      _drop_vetoed(analysis["recs"], vetoed),
                      label=accts.get(account_id, {}).get("label", "this reader"))
    try:
        reply = agent.chat(body.history, body.message)
    except Exception as e:
        raise HTTPException(502, f"chat backend error: {e}")
    return {"reply": reply}
