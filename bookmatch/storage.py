"""Postgres persistence for Book Match: schema `books.*`.

Additive to whatever database it is pointed at: schema + tables are created with
IF NOT EXISTS. Embeddings are stored as raw float32 bytes (numpy) rather than
pgvector — all geometry runs in numpy, so we only need durable blobs, not
in-DB vector search.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass

import numpy as np
import psycopg

DSN = os.environ.get("BOOKMATCH_PG_DSN") or os.environ.get("PG_DSN") \
    or "postgresql://localhost:5432/bookmatch"

_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS books;
CREATE TABLE IF NOT EXISTS books.accounts (
    id        serial PRIMARY KEY,
    user_id   text UNIQUE NOT NULL,
    label     text,
    url       text,
    added_at  timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS books.shelf_books (
    id           serial PRIMARY KEY,
    account_id   int NOT NULL REFERENCES books.accounts(id) ON DELETE CASCADE,
    gr_book_id   text,
    title        text NOT NULL,
    author       text,
    myrating     real,
    avg          real,
    shelves      text,
    descr        text,
    embedding    bytea,
    matched_book_id int,
    UNIQUE (account_id, title, author)
);
-- Fields the Goodreads RSS feed always carried and the importer used to discard
-- (ops.actionables:131533). Additive, so an existing shelf keeps its rows and
-- backfills on the next analyze. read_at is Goodreads' CATALOGUING date, not a
-- proven finish date — a bulk shelf-marking session stamps many books with one
-- day, so read it as "read by", never "read on".
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS read_at    date;
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS date_added timestamptz;
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS review     text;
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS isbn       text;
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS num_pages  int;
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS published  int;
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS cover_url  text;
-- the ISBN crosswalk is the point of capturing it (deterministic Calibre/reading
-- matching instead of fuzzy title matching)
CREATE INDEX IF NOT EXISTS shelf_books_isbn_idx
    ON books.shelf_books (isbn) WHERE isbn IS NOT NULL;
CREATE INDEX IF NOT EXISTS shelf_books_gr_book_id_idx
    ON books.shelf_books (gr_book_id) WHERE gr_book_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS books.analyses (
    account_id   int PRIMARY KEY REFERENCES books.accounts(id) ON DELETE CASCADE,
    profile      jsonb,
    recs         jsonb,
    match_rate   real,
    computed_at  timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS books.vetoes (
    id           serial PRIMARY KEY,
    account_id   int NOT NULL REFERENCES books.accounts(id) ON DELETE CASCADE,
    normkey      text NOT NULL,
    title        text NOT NULL,
    author       text,
    created_at   timestamptz DEFAULT now(),
    UNIQUE (account_id, normkey)
);
-- Where a shelf row came from: the Goodreads feed, or a reader verdict applied at
-- analysis time (see verdicts.py).
ALTER TABLE books.shelf_books ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'goodreads';
-- Reader verdicts on candidates (v0.6.0). One current verdict per book per account,
-- plus an append-only event log: the verdicts are the training signal, and undo is
-- a logged action, never an erased one. books.vetoes is kept read-only for history;
-- its rows are migrated into verdicts as not_interested by migrate_vetoes().
CREATE TABLE IF NOT EXISTS books.verdicts (
    id           serial PRIMARY KEY,
    account_id   int NOT NULL REFERENCES books.accounts(id) ON DELETE CASCADE,
    normkey      text NOT NULL,
    title        text NOT NULL,
    author       text,
    verdict      text NOT NULL CHECK (verdict IN ('want','read','disliked','not_interested')),
    decided_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (account_id, normkey)
);
CREATE TABLE IF NOT EXISTS books.verdict_events (
    id           bigserial PRIMARY KEY,
    account_id   int NOT NULL REFERENCES books.accounts(id) ON DELETE CASCADE,
    normkey      text NOT NULL,
    title        text NOT NULL,
    author       text,
    action       text NOT NULL CHECK (action IN ('want','read','disliked','not_interested','undo')),
    source       text NOT NULL DEFAULT 'ui',
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS verdict_events_account_idx ON books.verdict_events (account_id, created_at);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN)


def schema_present(conn: psycopg.Connection | None = None) -> bool:
    """Read-only check — never issues DDL, so it needs no CREATE grant.

    Postgres checks CREATE-on-database before an IF NOT EXISTS DDL statement's
    own short-circuit, so a job role without CREATE (e.g. ops_wrapper) is
    denied even when the schema already exists (ops.actionables:315969).
    """
    own = conn is None
    conn = conn or connect()
    try:
        row = conn.execute("SELECT to_regclass('books.verdict_events')").fetchone()
        return row is not None and row[0] is not None
    finally:
        if own:
            conn.close()


def init_schema(conn: psycopg.Connection | None = None):
    """Migration-time bootstrap/upgrade — run explicitly (`python -m bookmatch.storage`),
    never as an app-start or test-setup side effect. Requires CREATE on the database
    (the `livingos` role), which app/test job roles do not and should not hold."""
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute(_SCHEMA)
        conn.commit()
        migrate_vetoes(conn)
    finally:
        if own:
            conn.close()


def migrate_vetoes(conn) -> int:
    """Carry legacy 'not interested' vetoes into the verdict store. Idempotent DML (no
    DDL), so it is safe to run on every bootstrap and testable with an app role."""
    cur = conn.execute(
        """INSERT INTO books.verdicts (account_id, normkey, title, author, verdict, decided_at)
           SELECT account_id, normkey, title, author, 'not_interested', created_at
           FROM books.vetoes
           ON CONFLICT (account_id, normkey) DO NOTHING""")
    conn.commit()
    return cur.rowcount


def ensure_schema_ready(conn: psycopg.Connection | None = None) -> None:
    """App-start/test-setup guard: verify only, never mutate. Raises loudly if the
    schema is genuinely missing — bootstrap it once with `python -m bookmatch.storage`."""
    if not schema_present(conn):
        raise RuntimeError(
            "books.* schema is not present. Run `python -m bookmatch.storage` once "
            "(as a role with CREATE on the database) to bootstrap it — it is not "
            "created automatically on app start or test run (ops.actionables:315969)."
        )


if __name__ == "__main__":
    init_schema()
    print("bookmatch: books.* schema initialized/upgraded")


def upsert_account(conn, user_id: str, label: str, url: str) -> int:
    row = conn.execute(
        """INSERT INTO books.accounts (user_id, label, url) VALUES (%s,%s,%s)
           ON CONFLICT (user_id) DO UPDATE SET label=EXCLUDED.label, url=EXCLUDED.url
           RETURNING id""", (user_id, label, url)).fetchone()
    conn.commit()
    return row[0]


def save_shelf(conn, account_id: int, books: list[dict], embeddings: np.ndarray,
               matched: list[int | None]):
    """Replace the stored shelf for an account (idempotent re-analysis)."""
    conn.execute("DELETE FROM books.shelf_books WHERE account_id=%s", (account_id,))
    with conn.cursor() as cur:
        for b, emb, mid in zip(books, embeddings, matched):
            cur.execute(
                """INSERT INTO books.shelf_books
                   (account_id, gr_book_id, title, author, myrating, avg, shelves,
                    descr, embedding, matched_book_id,
                    read_at, date_added, review, isbn, num_pages, published, cover_url,
                    source)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                           %s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (account_id, title, author) DO NOTHING""",
                (account_id, b.get("book_id"), b["title"], b.get("author"),
                 float(b.get("myrating") or 0) or None,
                 float(b.get("avg") or 0) or None, b.get("shelves"),
                 b.get("desc"), np.asarray(emb, np.float32).tobytes(), mid,
                 b.get("read_at"), b.get("date_added"), b.get("review"),
                 b.get("isbn"), b.get("num_pages"), b.get("published"),
                 b.get("cover_url"), b.get("source") or "goodreads"))
    conn.commit()


def save_analysis(conn, account_id: int, profile: dict, recs: list[dict],
                  match_rate: float):
    conn.execute(
        """INSERT INTO books.analyses (account_id, profile, recs, match_rate, computed_at)
           VALUES (%s,%s,%s,%s, now())
           ON CONFLICT (account_id) DO UPDATE
             SET profile=EXCLUDED.profile, recs=EXCLUDED.recs,
                 match_rate=EXCLUDED.match_rate, computed_at=now()""",
        (account_id, json.dumps(profile), json.dumps(recs), match_rate))
    conn.commit()


def list_accounts(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT a.id, a.user_id, a.label, a.url, a.added_at,
                  (an.account_id IS NOT NULL) AS analyzed
           FROM books.accounts a LEFT JOIN books.analyses an ON an.account_id=a.id
           ORDER BY a.added_at DESC""").fetchall()
    return [{"id": r[0], "user_id": r[1], "label": r[2], "url": r[3],
             "added_at": r[4].isoformat() if r[4] else None, "analyzed": r[5]}
            for r in rows]


def get_analysis(conn, account_id: int) -> dict | None:
    row = conn.execute(
        """SELECT profile, recs, match_rate, computed_at
           FROM books.analyses WHERE account_id=%s""", (account_id,)).fetchone()
    if not row:
        return None
    return {"profile": row[0], "recs": row[1], "match_rate": row[2],
            "computed_at": row[3].isoformat() if row[3] else None}


@dataclass
class LoadedShelf:
    books: list[dict]
    embeddings: np.ndarray
    matched: list[int | None]


def load_shelf(conn, account_id: int) -> LoadedShelf | None:
    rows = conn.execute(
        """SELECT gr_book_id, title, author, myrating, avg, shelves, descr,
                  embedding, matched_book_id,
                  read_at, date_added, review, isbn, num_pages, published, cover_url,
                  source
           FROM books.shelf_books WHERE account_id=%s ORDER BY id""",
        (account_id,)).fetchall()
    if not rows:
        return None
    books, embs, matched = [], [], []
    for r in rows:
        books.append({"book_id": r[0], "title": r[1], "author": r[2],
                      "myrating": r[3], "avg": r[4], "shelves": r[5], "desc": r[6],
                      "read_at": r[9].isoformat() if r[9] else None,
                      "date_added": r[10].isoformat() if r[10] else None,
                      "review": r[11], "isbn": r[12], "num_pages": r[13],
                      "published": r[14], "cover_url": r[15], "source": r[16]})
        embs.append(np.frombuffer(r[7], dtype=np.float32) if r[7] else np.zeros(0))
        matched.append(r[8])
    return LoadedShelf(books, np.vstack(embs) if embs and embs[0].size else np.zeros((0, 0)), matched)


def set_verdict(conn, account_id: int, title: str, author: str | None,
                verdict: str, source: str = "ui") -> dict:
    """Record the reader's current verdict on a book and append it to the event log.
    A new verdict on the same book replaces the old one; both stay in the log."""
    from .match import normkey
    from .verdicts import VERDICTS
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; one of {', '.join(VERDICTS)}")
    key = normkey(title, author or "")
    row = conn.execute(
        """INSERT INTO books.verdicts (account_id, normkey, title, author, verdict, decided_at)
           VALUES (%s,%s,%s,%s,%s, now())
           ON CONFLICT (account_id, normkey) DO UPDATE
             SET verdict=EXCLUDED.verdict, title=EXCLUDED.title, author=EXCLUDED.author,
                 decided_at=now()
           RETURNING normkey, title, author, verdict, decided_at""",
        (account_id, key, title, author, verdict)).fetchone()
    conn.execute(
        """INSERT INTO books.verdict_events (account_id, normkey, title, author, action, source)
           VALUES (%s,%s,%s,%s,%s,%s)""", (account_id, key, title, author, verdict, source))
    conn.commit()
    return {"normkey": row[0], "title": row[1], "author": row[2], "verdict": row[3],
            "decided_at": row[4].isoformat()}


def clear_verdict(conn, account_id: int, title: str, author: str | None,
                  source: str = "ui") -> bool:
    """Undo: remove the current verdict (the book returns to the candidates) and log it."""
    from .match import normkey
    key = normkey(title, author or "")
    cur = conn.execute("DELETE FROM books.verdicts WHERE account_id=%s AND normkey=%s",
                       (account_id, key))
    if cur.rowcount:
        conn.execute(
            """INSERT INTO books.verdict_events (account_id, normkey, title, author, action, source)
               VALUES (%s,%s,%s,%s,'undo',%s)""", (account_id, key, title, author, source))
    conn.commit()
    return bool(cur.rowcount)


def list_verdicts(conn, account_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT normkey, title, author, verdict, decided_at FROM books.verdicts
           WHERE account_id=%s ORDER BY decided_at DESC""", (account_id,)).fetchall()
    return [{"normkey": r[0], "title": r[1], "author": r[2], "verdict": r[3],
             "decided_at": r[4].isoformat() if r[4] else None} for r in rows]


def decided_keys(conn, account_id: int) -> set[str]:
    """Every book with a current verdict: none of them is a candidate any more."""
    rows = conn.execute("SELECT normkey FROM books.verdicts WHERE account_id=%s",
                        (account_id,)).fetchall()
    return {r[0] for r in rows}


# ---- legacy names (the 'veto' of v0.4) map onto the not_interested verdict ----
def add_veto(conn, account_id: int, title: str, author: str | None) -> None:
    set_verdict(conn, account_id, title, author, "not_interested")


def remove_veto(conn, account_id: int, title: str, author: str | None) -> None:
    clear_verdict(conn, account_id, title, author)


def vetoed_keys(conn, account_id: int) -> set[str]:
    return decided_keys(conn, account_id)


def list_vetoes(conn, account_id: int) -> list[dict]:
    return [v for v in list_verdicts(conn, account_id) if v["verdict"] == "not_interested"]
