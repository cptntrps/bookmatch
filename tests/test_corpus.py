"""Integration gate for cards #99/#100: match a real shelf into the built UCSD
index and produce neighbor-gap recs. Skipped unless the index has been built.
"""
import json
import pathlib
import pytest
from bookmatch.corpus import Corpus

FIX = pathlib.Path(__file__).parent / "reader_c_books.json"
_corpus = Corpus()


@pytest.mark.skipif(not _corpus.ready, reason="UCSD index not built")
def test_shelf_matches_and_recommends():
    books = json.load(open(FIX))
    out = _corpus.recommend_for_shelf(books, n_recs=15)

    # match rate: a mainstream English shelf should match a solid fraction of the
    # 2.3M-book corpus. Assert a floor well below the expected ~0.6-0.8.
    assert out.match_rate >= 0.4, f"match rate only {out.match_rate:.2f}"
    assert len(out.matched) + len(out.unmatched) == len(books)

    # recs must exist, carry fields, and exclude owned works at the normkey level
    # (Goodreads has many book_ids per work — a sibling edition must not resurface).
    from bookmatch.match import normkey
    owned_keys = {normkey(b["title"], b.get("author") or "") for b in books}
    rec_keys = [normkey(r["title"], r["author"]) for r in out.recs]
    assert out.recs, "no recommendations produced"
    for r, k in zip(out.recs, rec_keys):
        assert r["title"] and "score" in r and "n_supporters" in r
        assert k not in owned_keys, f"recommended an owned work: {r['title']}"
    assert len(rec_keys) == len(set(rec_keys)), "duplicate works in recs"
