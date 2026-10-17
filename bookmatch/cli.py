"""Offline demo: taste profile and named-axis projection over a shelf JSON.

    python -m bookmatch.cli SHELF.json
    python -m bookmatch.cli SHELF.json --positive "French literature" "a novel set in Paris" \
        --negative "American literature" --label french [--candidates RECS.json] [--top 10]

Needs only an embedding endpoint (default: local Ollama with nomic-embed-text).
No database, no chat model. SHELF.json is a list of {title, author, desc, myrating,
avg}; RECS.json a list of {title, author}. The pure functions here are reused by
the chat agent's tool and are unit-tested without any service.
"""
from __future__ import annotations
import argparse
import json
import sys
import numpy as np

from . import analyze, embed


def axis_report(books: list[dict], E: np.ndarray, pos: np.ndarray,
                neg: np.ndarray | None, label: str, candidates: list[dict] | None = None,
                C: np.ndarray | None = None, top: int = 10) -> dict:
    """Rank the shelf (and optional candidates) along the axis pos - neg, in the
    shelf-centered frame, and measure how the reader's own ratings follow it."""
    r = analyze.axis_positions(books, E, pos, neg, label, candidates, C)
    name = lambda b: f"{b['title']} — {b.get('author', '')}"
    out = {"label": label,
           "positive_pole": [name(b) for b in r["shelf"][:top]],
           "negative_pole": [name(b) for b in r["shelf"][::-1][:top]],
           "rating_agreement": r["rating_agreement"]}
    if r["candidates"]:
        out["candidates_toward_positive"] = [name(b) for b in r["candidates"][:top]]
        out["candidates_toward_negative"] = [name(b) for b in r["candidates"][::-1][:top]]
    return out


def format_profile(profile: dict, top: int = 6) -> str:
    lines = [f"shelf: {profile['n_books']} books, {profile['n_rated']} rated, "
             f"mean rating {profile['mean_rating']}"]
    for key, title in (("loved", "loves most"), ("tolerated", "merely tolerates"),
                       ("loves_more_than_world", "loves more than the crowd"),
                       ("world_loves_more", "the crowd loves more"),
                       ("blind_spot", "blind spot (least like what they love)")):
        lines.append(f"\n{title}:")
        lines += [f"  {b['title']} — {b['author']}" for b in profile[key][:top]]
    return "\n".join(lines)


def format_report(report: dict) -> str:
    lines = [f"axis '{report['label']}'", "\ntoward the positive pole:"]
    lines += [f"  {t}" for t in report["positive_pole"]]
    lines.append("\ntoward the negative pole:")
    lines += [f"  {t}" for t in report["negative_pole"]]
    lines.append(f"\n{report['rating_agreement']['reading']}")
    if "candidates_toward_positive" in report:
        lines.append("\ncandidates toward the positive pole:")
        lines += [f"  {t}" for t in report["candidates_toward_positive"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shelf", help="JSON list of books: title, author, desc, myrating, avg")
    ap.add_argument("--positive", nargs="+", help="phrases for the positive pole")
    ap.add_argument("--negative", nargs="*", default=None, help="phrases for the opposite pole")
    ap.add_argument("--label", default="axis")
    ap.add_argument("--candidates", help="JSON list of candidate books: title, author")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)

    books = json.load(open(a.shelf))
    E = embed.embed_books(books)
    print(format_profile(analyze.build_profile(books, E)))
    if a.positive:
        pos = embed.embed_terms(a.positive)
        neg = embed.embed_terms(a.negative) if a.negative else None
        cands = json.load(open(a.candidates)) if a.candidates else None
        C = embed.embed_books(cands) if cands else None
        print("\n" + format_report(axis_report(books, E, pos, neg, a.label, cands, C, a.top)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
