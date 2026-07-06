"""Write-path test for books.* storage. Skipped if the Postgres DSN is unreachable.
Uses a throwaway account (unique user_id) and cleans up via ON DELETE CASCADE.
"""
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
    storage.init_schema()
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
