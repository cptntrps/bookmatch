#!/usr/bin/env python3
"""Build the neighbor-gap index from the raw UCSD Goodreads corpus.

Inputs  (~/data/goodreads-ucsd/raw, from download_dataset.sh):
  goodreads_interactions.csv      user_id,book_id,is_read,rating,is_reviewed  (remapped ids)
  book_id_map.csv                 book_id_csv,book_id                          (remapped -> real)
  goodreads_books.json.gz         real book_id -> title, authors[]
  goodreads_book_authors.json.gz  author_id -> name

Outputs (~/data/goodreads-ucsd/index):
  user_item.npz     scipy CSR (n_users x n_items) of explicit ratings (rating > 0)
  book_lookup.sqlite  books(book_id, title, author) + lookup(normkey -> book_id), indexed
  meta.json         {n_users, n_items, n_interactions, built_at}

Only rated interactions (rating in 1..5) are kept — explicit taste is the CF signal.
Idempotent: rewrites the index dir each run.
"""
from __future__ import annotations
import gzip
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bookmatch.match import normkey

RAW = Path(os.environ.get("BOOKMATCH_RAW", Path.home() / "data/goodreads-ucsd/raw"))
OUT = Path(os.environ.get("BOOKMATCH_INDEX", Path.home() / "data/goodreads-ucsd/index"))
CHUNK = 20_000_000


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_matrix() -> sparse.csr_matrix:
    log("reading interactions.csv (rating>0 only, chunked)...")
    us, bs, rs = [], [], []
    n_rows = 0
    reader = pd.read_csv(RAW / "goodreads_interactions.csv",
                         usecols=["user_id", "book_id", "rating"],
                         dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.int8},
                         chunksize=CHUNK)
    for i, chunk in enumerate(reader):
        n_rows += len(chunk)
        rated = chunk[chunk.rating > 0]
        us.append(rated.user_id.to_numpy())
        bs.append(rated.book_id.to_numpy())
        rs.append(rated.rating.to_numpy())
        log(f"  chunk {i}: {n_rows:,} rows scanned, {sum(len(x) for x in us):,} rated kept")
    u = np.concatenate(us); b = np.concatenate(bs); r = np.concatenate(rs)
    n_users = int(u.max()) + 1
    n_items = int(b.max()) + 1
    log(f"building CSR {n_users:,} users x {n_items:,} items, {len(r):,} ratings")
    m = sparse.csr_matrix((r.astype(np.float32), (u, b)), shape=(n_users, n_items))
    m.sum_duplicates()
    return m, n_rows


def build_lookup():
    log("loading author id -> name...")
    authors = {}
    with gzip.open(RAW / "goodreads_book_authors.json.gz", "rt") as fh:
        for line in fh:
            a = json.loads(line)
            authors[a["author_id"]] = a["name"]
    log(f"  {len(authors):,} authors")

    log("loading remapped book_id -> real book_id...")
    m = pd.read_csv(RAW / "book_id_map.csv",
                    dtype={"book_id_csv": np.int32, "book_id": str})
    real_to_remapped = dict(zip(m.book_id, m.book_id_csv.to_numpy()))
    log(f"  {len(real_to_remapped):,} book id mappings")

    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "book_lookup.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT, author TEXT);
        CREATE TABLE lookup (normkey TEXT, book_id INTEGER);
    """)
    log("streaming books.json.gz -> sqlite...")
    n = 0
    batch_b, batch_l = [], []
    with gzip.open(RAW / "goodreads_books.json.gz", "rt") as fh:
        for line in fh:
            bk = json.loads(line)
            rid = real_to_remapped.get(bk.get("book_id"))
            if rid is None:
                continue
            title = (bk.get("title") or "").strip()
            auth_list = bk.get("authors") or []
            author = ""
            if auth_list:
                author = authors.get(auth_list[0].get("author_id"), "")
            if not title:
                continue
            batch_b.append((int(rid), title, author))
            batch_l.append((normkey(title, author), int(rid)))
            n += 1
            if len(batch_b) >= 50_000:
                con.executemany("INSERT OR IGNORE INTO books VALUES (?,?,?)", batch_b)
                con.executemany("INSERT INTO lookup VALUES (?,?)", batch_l)
                batch_b.clear(); batch_l.clear()
                if n % 500_000 == 0:
                    log(f"  {n:,} books indexed")
    con.executemany("INSERT OR IGNORE INTO books VALUES (?,?,?)", batch_b)
    con.executemany("INSERT INTO lookup VALUES (?,?)", batch_l)
    log(f"  {n:,} books total; indexing normkey...")
    con.execute("CREATE INDEX idx_lookup ON lookup(normkey)")
    con.commit(); con.close()
    log("lookup db done")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, n_rows = build_matrix()
    sparse.save_npz(OUT / "user_item.npz", m)
    log(f"saved user_item.npz ({(OUT / 'user_item.npz').stat().st_size / 1e9:.2f} GB)")
    build_lookup()
    meta = {"n_users": m.shape[0], "n_items": m.shape[1],
            "n_ratings": int(m.nnz), "n_interactions": n_rows,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE: {meta}")


if __name__ == "__main__":
    main()
