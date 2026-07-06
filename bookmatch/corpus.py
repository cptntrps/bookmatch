"""Query-time glue: load the neighbor index once, match a live shelf into the
corpus, run the neighbor-gap engine, and map recs back to titles.

The index is large (~1 GB CSR + a sqlite lookup); `Corpus` loads it lazily and
holds it for the process lifetime. The FastAPI app instantiates one `Corpus`.
"""
from __future__ import annotations
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from scipy import sparse

from . import neighbors
from .match import normkey

INDEX_DIR = Path(os.environ.get("BOOKMATCH_INDEX",
                                Path.home() / "data/goodreads-ucsd/index"))


@dataclass
class MatchedBook:
    title: str
    author: str
    myrating: float
    book_id: int  # remapped corpus id


@dataclass
class ShelfRecs:
    matched: list[MatchedBook]
    unmatched: list[dict]
    recs: list[dict]          # {title, author, score, n_supporters}
    match_rate: float


class Corpus:
    def __init__(self, index_dir: Path | None = None):
        self.dir = Path(index_dir) if index_dir else INDEX_DIR
        self._csr: sparse.csr_matrix | None = None
        self._csc: sparse.csc_matrix | None = None
        self._db: sqlite3.Connection | None = None
        self.meta: dict = {}

    # ------- lazy loading -------
    def load(self):
        if self._csr is not None:
            return
        self._csr = sparse.load_npz(self.dir / "user_item.npz").tocsr()
        self._csc = self._csr.tocsc()
        self._db = sqlite3.connect(f"file:{self.dir/'book_lookup.sqlite'}?mode=ro",
                                   uri=True, check_same_thread=False)
        self.meta = json.loads((self.dir / "meta.json").read_text())

    @property
    def ready(self) -> bool:
        return (self.dir / "user_item.npz").exists() and (self.dir / "meta.json").exists()

    # ------- matching -------
    def _match_one(self, title: str, author: str) -> int | None:
        row = self._db.execute(
            "SELECT book_id FROM lookup WHERE normkey=? LIMIT 1",
            (normkey(title, author),)).fetchone()
        return int(row[0]) if row else None

    def title_for(self, book_id: int) -> tuple[str, str]:
        row = self._db.execute("SELECT title, author FROM books WHERE book_id=?",
                               (book_id,)).fetchone()
        return (row[0], row[1]) if row else ("(unknown)", "")

    # ------- the full query -------
    def recommend_for_shelf(self, books: list[dict], *, n_recs: int = 20,
                            k_neighbors: int = 100, min_overlap: int = 2) -> ShelfRecs:
        self.load()
        matched, unmatched, target = [], [], {}
        for b in books:
            bid = self._match_one(b["title"], b["author"])
            rating = float(b.get("myrating") or 0)
            if bid is None:
                unmatched.append(b)
                continue
            matched.append(MatchedBook(b["title"], b["author"], rating, bid))
            # unrated shelf books still count as a positive signal (they read it)
            target[bid] = rating if rating > 0 else 4.0
        rate = len(matched) / max(1, len(books))

        recs_out = []
        if target:
            recs = neighbors.recommend(self._csr, self._csc, target,
                                       k_neighbors=k_neighbors, n_recs=n_recs,
                                       min_overlap=min_overlap)
            for r in recs:
                title, author = self.title_for(r.item)
                recs_out.append({"title": title, "author": author,
                                 "score": round(r.score, 3),
                                 "n_supporters": r.n_supporters})
        return ShelfRecs(matched, unmatched, recs_out, rate)
