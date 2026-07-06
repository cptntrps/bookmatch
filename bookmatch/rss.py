"""Fetch a public Goodreads shelf via its RSS feed — no API key, no auth.

Goodreads killed its JSON API in 2020 but still serves a per-shelf RSS feed for
public profiles:

    https://www.goodreads.com/review/list_rss/{user_id}?shelf={shelf}&page={n}

Ported from the June-2026 scratch fetchers (`reader_a_fetch.py` et al).
"""
from __future__ import annotations
import html
import re
import time
from xml.etree import ElementTree as ET
import httpx

_TAG = re.compile(r"<[^>]+>")
_UA = "Mozilla/5.0 (compatible; BookMatch/0.1)"


def parse_user_id(url_or_id: str) -> str:
    """Accept a raw numeric id or any Goodreads profile/shelf URL and return the
    numeric user id. Goodreads user URLs look like
    /user/show/12345678-name or /review/list/12345678?shelf=read."""
    s = url_or_id.strip()
    if s.isdigit():
        return s
    m = re.search(r"/(?:user/show|review/list(?:_rss)?)/(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{3,})", s)  # last resort: first long number
    if m:
        return m.group(1)
    raise ValueError(f"could not extract a Goodreads user id from {url_or_id!r}")


def _text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "") if el is not None else ""


def parse_shelf_rss(xml_bytes: bytes) -> list[dict]:
    """Parse one RSS page into a list of book dicts. Raises on malformed XML."""
    root = ET.fromstring(xml_bytes)
    books = []
    for it in root.findall(".//item"):
        desc = _TAG.sub(" ", html.unescape(_text(it, "book_description"))).strip()
        books.append({
            "book_id": _text(it, "book_id").strip(),
            "title": _text(it, "title").strip(),
            "author": _text(it, "author_name").strip(),
            "myrating": _text(it, "user_rating").strip(),
            "avg": _text(it, "average_rating").strip(),
            "shelves": _text(it, "user_shelves").strip(),
            "desc": desc[:700],
        })
    return books


def fetch_shelf(url_or_id: str, shelf: str = "read", max_pages: int = 20,
                client: httpx.Client | None = None) -> list[dict]:
    """Fetch and dedupe every book on a public Goodreads shelf.

    Raises ValueError if the profile is private / empty (no items on page 1) so
    the caller fails loudly rather than returning silent empty recommendations.
    """
    user_id = parse_user_id(url_or_id)
    owns = client is None
    client = client or httpx.Client(timeout=30, headers={"User-Agent": _UA})
    books, seen = [], set()
    try:
        for page in range(1, max_pages + 1):
            url = (f"https://www.goodreads.com/review/list_rss/{user_id}"
                   f"?shelf={shelf}&page={page}")
            r = client.get(url)
            r.raise_for_status()
            page_books = parse_shelf_rss(r.content)
            if not page_books:
                break
            new = 0
            for b in page_books:
                key = b["book_id"] or b["title"]
                if key in seen:
                    continue
                seen.add(key)
                books.append(b)
                new += 1
            if new == 0:
                break
            time.sleep(0.4)
    finally:
        if owns:
            client.close()
    if not books:
        raise ValueError(
            f"no books found on shelf '{shelf}' for user {user_id} — "
            "the profile may be private, empty, or the id wrong")
    return books
