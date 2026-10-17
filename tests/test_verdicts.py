"""Verdict semantics: pure, no services."""
from bookmatch import verdicts as V


SHELF = [{"title": "Dune", "author": "Frank Herbert", "myrating": 5, "desc": "sand"},
         {"title": "Blindness", "author": "José Saramago", "myrating": None, "desc": "eyes"}]


def test_read_and_disliked_join_the_shelf_but_want_and_not_interested_do_not():
    rows = [{"title": "Hyperion", "author": "Dan Simmons", "verdict": "read"},
            {"title": "Ready Player One", "author": "Ernest Cline", "verdict": "disliked"},
            {"title": "Snow Crash", "author": "Neal Stephenson", "verdict": "want"},
            {"title": "Twilight", "author": "Stephenie Meyer", "verdict": "not_interested"}]
    merged = V.merge_into_shelf(SHELF, rows)
    titles = [b["title"] for b in merged]
    assert titles == ["Dune", "Blindness", "Hyperion", "Ready Player One"]
    by = {b["title"]: b for b in merged}
    assert by["Hyperion"]["myrating"] is None and by["Hyperion"]["source"] == "verdict:read"
    assert by["Ready Player One"]["myrating"] == V.DISLIKED_RATING
    assert SHELF[0].get("source") is None            # input untouched


def test_dislike_overrides_rating_of_a_book_already_on_the_shelf():
    merged = V.merge_into_shelf(SHELF, [{"title": "Blindness", "author": "José Saramago", "verdict": "disliked"}])
    assert len(merged) == 2 and merged[1]["myrating"] == V.DISLIKED_RATING
    # a 'read' verdict for a shelf book changes nothing
    again = V.merge_into_shelf(SHELF, [{"title": "Dune", "author": "Frank Herbert", "verdict": "read"}])
    assert again == [dict(b) for b in SHELF]


def test_grouped_has_every_verdict_and_sorts_newest_first():
    g = V.grouped([{"title": "a", "verdict": "want", "decided_at": "2026-01-01"},
                   {"title": "b", "verdict": "want", "decided_at": "2026-02-01"}])
    assert set(g) == set(V.VERDICTS)
    assert [r["title"] for r in g["want"]] == ["b", "a"] and g["read"] == []
