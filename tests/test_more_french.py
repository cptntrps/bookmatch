"""Integration gate for card #101: reproduce the June 'make it more french'
result on the real reader_a fixture. Requires Ollama (nomic-embed-text) running.

The claim being defended: an unsupervised semantic axis (French - American,
built from a handful of concept phrases) ranks the French-language authors on
Reader A's 309-book shelf near the top — nobody labelled which books are French.
"""
import json
import os
import pathlib
import numpy as np
import pytest

from bookmatch import analyze, embed

FIX = pathlib.Path(__file__).parent / "reader_a_books.json"

FRENCH_TERMS = ["French literature", "a novel set in France", "French author",
                "Paris, France", "la litterature francaise"]
AMERICAN_TERMS = ["American literature", "a novel set in small-town America",
                  "American author", "the United States", "contemporary American fiction"]

# Ground truth = the exact continental-European literary authors the June session
# itself validated the axis against (reader_a_french.py). The axis is really
# "European translated literary fiction vs American"; these are the recognizably
# European names on Reader A's 309-book shelf.
FRENCH_AUTHORS = ["duras", "süskind", "suskind", "calvino", "ventura", "harwicz"]


def _ollama_up() -> bool:
    try:
        import httpx
        httpx.get(f"{embed.OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable")
def test_more_french_surfaces_french_authors():
    books = json.load(open(FIX))
    E = embed.embed_books(books)
    pos = embed.embed_terms(FRENCH_TERMS)
    neg = embed.embed_terms(AMERICAN_TERMS)
    axis = analyze.axis_from_vectors(pos, neg)
    proj = analyze.project_axis(E, axis)

    order = np.argsort(-proj)
    top = [books[i]["author"].lower() for i in order[:25]]
    hits = sum(any(fa in a for fa in FRENCH_AUTHORS) for a in top)
    # At least two of the validated European authors land in the top 25 of 309 —
    # the geometry found them with zero supervision.
    assert hits >= 2, f"only {hits} European authors in top 25: {top[:8]}"
    # Marguerite Duras (unambiguously French) must rank in the top 10% (~top 30).
    duras_rank = next(r for r, i in enumerate(order)
                      if "duras" in books[i]["author"].lower())
    assert duras_rank < 31, f"Duras ranked #{duras_rank + 1}/{len(books)}"
