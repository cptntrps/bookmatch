"""Write-path test for books.* storage. Skipped if the Postgres DSN is unreachable.
Uses a throwaway account (unique user_id) and cleans up via ON DELETE CASCADE.
"""
from datetime import datetime

import numpy as np
import pytest

from bookmatch import storage


def _db_up() -> bool:
    try:
        c = storage.connect(); c.close(); return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not reachable")

TEST_USER = "bookmatch-test-9999999"


@pytest.fixture
def conn():
    storage.ensure_schema_ready()
    c = storage.connect()
    c.execute("DELETE FROM books.accounts WHERE user_id=%s", (TEST_USER,)); c.commit()
    yield c
    c.execute("DELETE FROM books.accounts WHERE user_id=%s", (TEST_USER,)); c.commit()
    c.close()


def test_save_and_reload_shelf_without_recompute(conn):
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x/y/9999999")
    books = [
        {"book_id": "1", "title": "The Lover", "author": "Marguerite Duras",
         "myrating": 5, "avg": 3.79, "shelves": "fav", "desc": "d1"},
        {"book_id": "2", "title": "Perfume", "author": "Patrick Suskind",
         "myrating": 4, "avg": 4.02, "shelves": "read", "desc": "d2"},
    ]
    embs = np.random.default_rng(0).normal(size=(2, 8)).astype(np.float32)
    storage.save_shelf(conn, aid, books, embs, matched=[101, None])

    loaded = storage.load_shelf(conn, aid)
    assert loaded is not None
    assert [b["title"] for b in loaded.books] == ["The Lover", "Perfume"]
    assert loaded.matched == [101, None]
    # embeddings round-trip byte-exact (no recompute needed on reload)
    assert np.allclose(loaded.embeddings, embs)


def test_shelf_roundtrips_the_goodreads_metadata(conn):
    """ops.actionables:131533 — read date, ISBN, review, pages, publication year and
    cover URL must survive the write path, not stop at the parser."""
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x")
    book = {"book_id": "42", "title": "The Lover", "author": "Marguerite Duras",
            "myrating": 5, "avg": 3.79, "shelves": "fav", "desc": "d1",
            "read_at": "2026-07-20", "date_added": "2026-07-20T22:07:10-07:00",
            "review": "Spare and devastating.", "isbn": "0060977825",
            "num_pages": 117, "published": 1984,
            "cover_url": "https://i.gr-assets.com/large.jpg"}
    storage.save_shelf(conn, aid, [book], np.zeros((1, 8), np.float32), matched=[None])

    got = storage.load_shelf(conn, aid).books[0]
    assert got["read_at"] == "2026-07-20"
    # timestamptz reloads in the session time zone — compare the instant, not the text
    assert (datetime.fromisoformat(got["date_added"])
            == datetime.fromisoformat("2026-07-20T22:07:10-07:00"))
    assert got["review"] == "Spare and devastating."
    assert got["isbn"] == "0060977825"
    assert got["num_pages"] == 117
    assert got["published"] == 1984
    assert got["cover_url"] == "https://i.gr-assets.com/large.jpg"


def test_shelf_accepts_a_book_with_no_metadata(conn):
    """A feed item carrying none of the optional fields must still persist — the
    columns are nullable, not a new required contract."""
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x")
    storage.save_shelf(conn, aid, [{"title": "Dune", "author": "Frank Herbert"}],
                       np.zeros((1, 8), np.float32), matched=[None])
    got = storage.load_shelf(conn, aid).books[0]
    assert got["title"] == "Dune"
    assert got["read_at"] is None and got["isbn"] is None and got["num_pages"] is None


def test_save_and_get_analysis(conn):
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x")
    storage.save_analysis(conn, aid, {"taste": ["a", "b"]},
                          [{"title": "X", "author": "Y", "score": 1.0, "n_supporters": 3}],
                          match_rate=0.72)
    got = storage.get_analysis(conn, aid)
    assert got["match_rate"] == pytest.approx(0.72)
    assert got["profile"]["taste"] == ["a", "b"]
    assert got["recs"][0]["title"] == "X"

    accts = storage.list_accounts(conn)
    mine = [a for a in accts if a["user_id"] == TEST_USER]
    assert mine and mine[0]["analyzed"] is True


def test_veto_persists_across_reanalysis_and_ingest(conn):
    """Row 95834: a veto must survive both a fresh save_analysis (re-analyze) and a
    fresh save_shelf (Goodreads re-ingest) — it lives in its own table, untouched by
    either's delete-and-replace write path."""
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x")

    storage.add_veto(conn, aid, "Ready Player One", "Ernest Cline")
    assert storage.vetoed_keys(conn, aid) == {"ready player one|cline"}

    # simulate a Goodreads re-ingest (save_shelf deletes and reinserts shelf_books)
    embs = np.zeros((1, 8), np.float32)
    storage.save_shelf(conn, aid, [{"title": "Dune", "author": "Frank Herbert"}],
                       embs, matched=[None])
    # and a re-analyze (save_analysis is an upsert over books.analyses)
    storage.save_analysis(conn, aid, {"taste": []},
                          [{"title": "Ready Player One", "author": "Ernest Cline",
                            "score": 5.4, "n_supporters": 38}],
                          match_rate=0.5)

    assert storage.vetoed_keys(conn, aid) == {"ready player one|cline"}

    storage.remove_veto(conn, aid, "Ready Player One", "Ernest Cline")
    assert storage.vetoed_keys(conn, aid) == set()


def test_verdicts_replace_each_other_undo_and_log(conn):
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x")
    first = storage.set_verdict(conn, aid, "Hyperion", "Dan Simmons", "want")
    assert first["verdict"] == "want"
    # a later verdict on the same book replaces the current one
    storage.set_verdict(conn, aid, "Hyperion", "Dan Simmons", "disliked")
    current = {v["normkey"]: v["verdict"] for v in storage.list_verdicts(conn, aid)}
    assert current == {"hyperion|simmons": "disliked"}
    assert storage.decided_keys(conn, aid) == {"hyperion|simmons"}
    # undo returns the book to the candidates
    assert storage.clear_verdict(conn, aid, "Hyperion", "Dan Simmons") is True
    assert storage.clear_verdict(conn, aid, "Hyperion", "Dan Simmons") is False
    assert storage.decided_keys(conn, aid) == set()
    # every step is in the append-only log, undo included
    actions = [r[0] for r in conn.execute(
        "SELECT action FROM books.verdict_events WHERE account_id=%s ORDER BY id", (aid,)).fetchall()]
    assert actions == ["want", "disliked", "undo"]
    with pytest.raises(ValueError):
        storage.set_verdict(conn, aid, "X", "Y", "maybe")


def test_legacy_vetoes_migrate_to_not_interested(conn):
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x")
    conn.execute("INSERT INTO books.vetoes (account_id, normkey, title, author) VALUES (%s,%s,%s,%s)",
                 (aid, "ready player one|cline", "Ready Player One", "Ernest Cline")); conn.commit()
    assert storage.migrate_vetoes(conn) >= 1
    assert storage.migrate_vetoes(conn) == 0              # idempotent
    got = storage.list_vetoes(conn, aid)
    assert got and got[0]["verdict"] == "not_interested"


def test_list_vetoes(conn):
    aid = storage.upsert_account(conn, TEST_USER, "Test Reader", "http://x")
    storage.add_veto(conn, aid, "Ready Player One", "Ernest Cline")
    got = storage.list_vetoes(conn, aid)
    assert len(got) == 1
    assert got[0]["title"] == "Ready Player One"
    assert got[0]["normkey"] == "ready player one|cline"

    # idempotent — vetoing twice does not duplicate
    storage.add_veto(conn, aid, "Ready Player One", "Ernest Cline")
    assert len(storage.list_vetoes(conn, aid)) == 1
