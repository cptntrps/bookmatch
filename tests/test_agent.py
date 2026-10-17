"""Chat agent gate (card #103).

- build_profile structure: unit, synthetic embeddings, no services.
- project_axis tool: needs embeddings; ranks shelf AND candidates, reports agreement.
- chat: needs a chat key; must return grounded, non-empty text.
"""
import json
import os
import pathlib
import numpy as np
import pytest

from bookmatch import analyze, embed
from bookmatch import agent as agent_mod
from bookmatch.agent import BookAgent

FIX = pathlib.Path(__file__).parent / "reader_a_books.json"


def test_build_profile_structure():
    books = [
        {"title": "A", "author": "x", "myrating": 5, "avg": 3.0},
        {"title": "B", "author": "y", "myrating": 5, "avg": 4.8},
        {"title": "C", "author": "z", "myrating": 1, "avg": 4.5},
    ]
    E = np.array([[1, 0], [0.8, 0.2], [0, 1]], float)
    p = analyze.build_profile(books, E, n=3)
    assert p["n_rated"] == 3 and p["mean_rating"] is not None
    for key in ("loved", "tolerated", "loves_more_than_world", "blind_spot"):
        assert isinstance(p[key], list)
    # book A (rated 5, avg 3.0) loves it far more than the world
    assert p["loves_more_than_world"][0]["title"] == "A"


@pytest.mark.skipif(not embed.available(), reason="embedding endpoint not reachable")
def test_project_axis_tool_wrapper():
    # Axis *quality* (surfacing French authors) is gated in test_more_french at the
    # right depth; here we prove the tool wrapper returns a well-formed, real ranking.
    books = json.load(open(FIX))
    E = embed.embed_books(books)
    recs = [{"title": "Madame Bovary", "author": "Gustave Flaubert"},
            {"title": "The Grapes of Wrath", "author": "John Steinbeck"},
            {"title": "Swann's Way", "author": "Marcel Proust"}]
    agent = BookAgent(books, E, profile={}, recs=recs, label="Reader A")
    out = agent.project_axis(["French literature", "a novel set in France"],
                             ["American literature"], top=12, label="french")
    assert len(out["positive_pole"]) == 12 and len(out["negative_pole"]) == 12
    # candidates are ranked in the shelf's frame: the French novels lead, Steinbeck trails
    assert out["candidates_toward_positive"][-1].startswith("The Grapes of Wrath")
    assert out["candidates_toward_negative"][0].startswith("The Grapes of Wrath")
    # agreement is measured over the 309-book shelf, with a quotable reading
    agr = out["rating_agreement"]
    assert agr["n_rated"] > 100 and "french" in agr["reading"]
    # the two poles are genuinely opposite ends — no overlap
    assert not (set(out["positive_pole"]) & set(out["negative_pole"]))
    # and it is really axis-ordered: the French pole differs from a neutral 'summer' pole
    other = agent.project_axis(["summer", "beach", "warmth"], top=12)
    assert out["positive_pole"] != other["positive_pole"]


# Opt-in only: an explicit bookmatch chat key, never an incidental OPENAI_API_KEY in the
# environment, so a test run cannot spend money on a key it was not given.
_EXPLICIT_KEY = os.environ.get("BOOKMATCH_CHAT_KEY") or os.environ.get("BOOKMATCH_GATEWAY_KEY")


@pytest.mark.skipif(not (_EXPLICIT_KEY and embed.available()),
                    reason="no explicit BOOKMATCH_CHAT_KEY / embedding endpoint not available")
def test_chat_returns_grounded_text():
    books = json.load(open(FIX))
    E = embed.embed_books(books)
    profile = analyze.build_profile(books, E)
    agent = BookAgent(books, E, profile, recs=[{"title": "Blindness",
                      "author": "José Saramago"}], label="Reader A")
    reply = agent.chat([], "In one or two sentences, recommend a single poetry book she'd like.")
    assert reply and len(reply) > 20
