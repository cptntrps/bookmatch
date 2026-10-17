"""Reader verdicts on candidate books, and what each one means for the taste model.

want            keep it on the reader's list; hide it from candidates; no taste effect yet
read            the shelf missed it: it joins the shelf as a read, unrated book
disliked        read and not liked: it joins the shelf with a low rating (a negative signal
                for the taste axis and the neighbour search)
not_interested  hide it from candidates; says nothing about taste

Verdicts live in their own table and outlive re-analysis. They are applied to the shelf
at analysis time by `merge_into_shelf`, so the stored shelf and its embeddings stay the
single source the profile, the axis and the chat agent read.
"""
from __future__ import annotations
from .match import normkey

VERDICTS = ("want", "read", "disliked", "not_interested")
LABELS = {"want": "Want to read", "read": "Read it",
          "disliked": "Didn't like it", "not_interested": "Not interested"}
DISLIKED_RATING = 2.0
JOINS_SHELF = {"read": None, "disliked": DISLIKED_RATING}   # verdict -> rating on the shelf


def merge_into_shelf(shelf: list[dict], verdicts: list[dict]) -> list[dict]:
    """Return the shelf with read/disliked verdict books added (or, when the book is
    already on the shelf, with a dislike applied to its rating). The explicit verdict
    beats a stale or missing Goodreads rating. Order is preserved; input is not mutated."""
    out = [dict(b) for b in shelf]
    index = {normkey(b["title"], b.get("author") or ""): i for i, b in enumerate(out)}
    for v in verdicts:
        if v["verdict"] not in JOINS_SHELF:
            continue
        key = normkey(v["title"], v.get("author") or "")
        if key in index:
            if v["verdict"] == "disliked":
                out[index[key]]["myrating"] = DISLIKED_RATING
            continue
        out.append({"title": v["title"], "author": v.get("author") or "",
                    "myrating": JOINS_SHELF[v["verdict"]], "desc": "",
                    "source": f"verdict:{v['verdict']}"})
        index[key] = len(out) - 1
    return out


def grouped(verdicts: list[dict]) -> dict[str, list[dict]]:
    """{verdict: [rows]} with every verdict present, newest first within a group."""
    groups: dict[str, list[dict]] = {v: [] for v in VERDICTS}
    for row in sorted(verdicts, key=lambda r: r.get("decided_at") or "", reverse=True):
        groups[row["verdict"]].append(row)
    return groups
