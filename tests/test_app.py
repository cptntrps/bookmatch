"""Unit test for the rec-filtering helper behind the veto endpoints (row 95834).

Pure function, no DB/corpus needed — importing `bookmatch.app` is safe because
`Corpus()` is lazy (see corpus.py) and the FastAPI lifespan never runs here.
"""
from bookmatch.app import _drop_vetoed
from bookmatch.match import normkey


def test_drop_vetoed_excludes_matching_normkey():
    recs = [
        {"title": "Ready Player One", "author": "Ernest Cline", "score": 5.4},
        {"title": "For We Are Many", "author": "Dennis E. Taylor", "score": 4.1},
    ]
    vetoed = {normkey("Ready Player One", "Ernest Cline")}
    out = _drop_vetoed(recs, vetoed)
    assert [r["title"] for r in out] == ["For We Are Many"]


def test_drop_vetoed_empty_set_is_noop():
    recs = [{"title": "Dune", "author": "Frank Herbert", "score": 1.0}]
    assert _drop_vetoed(recs, set()) == recs
