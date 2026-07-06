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
    <book_description>&lt;p&gt;A story set in &lt;b&gt;French&lt;/b&gt; Indochina.&lt;/p&gt;</book_description>
  </item>
  <item>
    <book_id>67890</book_id>
    <title>Perfume</title>
    <author_name>Patrick Süskind</author_name>
    <user_rating>4</user_rating>
    <average_rating>4.02</average_rating>
    <user_shelves>read</user_shelves>
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
