"""Unit tests for the Goodreads RSS fetch/parse — no network needed."""
import pytest
from bookmatch import rss

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <book_id>12345</book_id>
    <title>The Lover</title>
    <author_name>Marguerite Duras</author_name>
    <user_rating>5</user_rating>
    <average_rating>3.79</average_rating>
    <user_shelves>favorites</user_shelves>
    <user_read_at>Mon, 20 Jul 2026 00:00:00 +0000</user_read_at>
    <user_date_added>Mon, 20 Jul 2026 22:07:10 -0700</user_date_added>
    <user_review>&lt;p&gt;Spare and &lt;b&gt;devastating&lt;/b&gt;.&lt;/p&gt;</user_review>
    <isbn>0060977825</isbn>
    <book_published>1984</book_published>
    <book_image_url>https://i.gr-assets.com/small.jpg</book_image_url>
    <book_large_image_url>https://i.gr-assets.com/large.jpg</book_large_image_url>
    <book><num_pages>117</num_pages></book>
    <book_description>&lt;p&gt;A story set in &lt;b&gt;French&lt;/b&gt; Indochina.&lt;/p&gt;</book_description>
  </item>
  <item>
    <book_id>67890</book_id>
    <title>Perfume</title>
    <author_name>Patrick Süskind</author_name>
    <user_rating>4</user_rating>
    <average_rating>4.02</average_rating>
    <user_shelves>read</user_shelves>
    <user_read_at></user_read_at>
    <isbn></isbn>
    <book_description>A murderer with an extraordinary sense of smell.</book_description>
  </item>
</channel></rss>""".encode("utf-8")


def test_parse_shelf_rss_extracts_fields():
    books = rss.parse_shelf_rss(SAMPLE)
    assert len(books) == 2
    b = books[0]
    assert b["title"] == "The Lover"
    assert b["author"] == "Marguerite Duras"
    assert b["myrating"] == "5"
    assert b["avg"] == "3.79"
    # HTML tags stripped, entities unescaped
    assert "<" not in b["desc"] and "French" in b["desc"] and "Indochina" in b["desc"]


def test_parse_shelf_rss_captures_every_feed_field():
    """ops.actionables:131533 — read dates, ISBNs, reviews and covers were parsed
    away even though the feed always carried them."""
    first, second = rss.parse_shelf_rss(SAMPLE)
    assert first["read_at"] == "2026-07-20"
    assert first["date_added"].startswith("2026-07-20T22:07:10")
    assert first["review"] == "Spare and devastating ."
    assert first["isbn"] == "0060977825"
    assert first["num_pages"] == 117
    assert first["published"] == 1984
    assert first["cover_url"] == "https://i.gr-assets.com/large.jpg"   # large preferred
    # absent/empty elements become None, never "" — they are unknown, not blank
    assert second["read_at"] is None
    assert second["isbn"] is None
    assert second["review"] is None
    assert second["num_pages"] is None
    assert second["cover_url"] is None


def _edition(book_id, **over):
    b = {"book_id": book_id, "title": "The Three-Body Problem", "author": "Liu Cixin",
         "myrating": "0", "avg": "4.1", "shelves": "", "desc": "",
         "read_at": None, "date_added": None, "review": None, "isbn": None,
         "num_pages": None, "published": None, "cover_url": None}
    b.update(over)
    return b


def test_dedupe_editions_keeps_the_read_edition_and_merges_the_other():
    """The live case: books.shelf_books is UNIQUE (account_id,title,author), so one
    of Gui's 75 shelf items never persisted. The collapse must be explicit and
    lossless, not an ON CONFLICT DO NOTHING."""
    read_one = _edition("18245960", myrating="5", read_at="2026-06-11")
    isbn_one = _edition("20518872", myrating="4", isbn="1400110483", num_pages=400)
    kept, dropped = rss.dedupe_editions([read_one, isbn_one])
    assert len(kept) == 1 and len(dropped) == 1
    assert kept[0]["book_id"] == "18245960"      # the edition with the read date wins
    assert kept[0]["myrating"] == "5"
    assert kept[0]["isbn"] == "1400110483"       # ...but the loser's ISBN survives
    assert kept[0]["num_pages"] == 400
    assert dropped[0]["book_id"] == "20518872"


def test_dedupe_editions_falls_back_to_rating_and_preserves_order():
    a = _edition("1", myrating="3")
    b = _edition("2", myrating="5", isbn="999")
    other = dict(_edition("3"), title="Perfume", author="Patrick Süskind")
    kept, dropped = rss.dedupe_editions([a, other, b])
    assert [k["title"] for k in kept] == ["The Three-Body Problem", "Perfume"]
    assert kept[0]["book_id"] == "2" and kept[0]["isbn"] == "999"
    assert [d["book_id"] for d in dropped] == ["1"]


def test_dedupe_editions_noop_on_distinct_books():
    books = rss.parse_shelf_rss(SAMPLE)
    kept, dropped = rss.dedupe_editions(books)
    assert len(kept) == 2 and dropped == []


def test_parse_empty_rss():
    empty = b"<?xml version='1.0'?><rss><channel></channel></rss>"
    assert rss.parse_shelf_rss(empty) == []


@pytest.mark.parametrize("inp,expected", [
    ("12345678", "12345678"),
    ("https://www.goodreads.com/user/show/12345678-reader_a", "12345678"),
    ("https://www.goodreads.com/review/list/23456789-reader_b", "23456789"),
    ("https://www.goodreads.com/review/list_rss/34567890?shelf=read", "34567890"),
    ("https://www.goodreads.com/review/list/45678901?ref=nav_mybooks", "45678901"),
])
def test_parse_user_id(inp, expected):
    assert rss.parse_user_id(inp) == expected


def test_parse_user_id_rejects_garbage():
    with pytest.raises(ValueError):
        rss.parse_user_id("not-a-goodreads-url")
