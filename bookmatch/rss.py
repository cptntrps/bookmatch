"""Fetch a public Goodreads shelf via its RSS feed — no API key, no auth.

Goodreads killed its JSON API in 2020 but still serves a per-shelf RSS feed for
public profiles:

    https://www.goodreads.com/review/list_rss/{user_id}?shelf={shelf}&page={n}

Ported from the June-2026 scratch fetchers.
"""
from __future__ import annotations
import html
import logging
import re
import time
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET
import httpx

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_UA = "Mozilla/5.0 (compatible; BookMatch/0.1)"

log = logging.getLogger(__name__)


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


def _rfc822_date(raw: str) -> str | None:
    """'Mon, 20 Jul 2026 00:00:00 +0000' -> '2026-07-20'. None on empty/garbage."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        return None


def _rfc822_ts(raw: str) -> str | None:
    """Same input, full ISO-8601 timestamp (Goodreads keeps the user's offset)."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).isoformat()
    except (TypeError, ValueError):
        return None


def _int(raw: str) -> int | None:
    m = re.search(r"\d+", raw or "")
    return int(m.group(0)) if m else None


def parse_shelf_rss(xml_bytes: bytes) -> list[dict]:
    """Parse one RSS page into a list of book dicts. Raises on malformed XML.

    Every field the feed carries is captured. `read_at` is Goodreads' *cataloguing*
    date, not a proven finish date — a bulk shelf-marking session stamps many books
    with one day. Consumers must treat it as "read by", never as "read on".
    """
    root = ET.fromstring(xml_bytes)
    books = []
    for it in root.findall(".//item"):
        desc = _TAG.sub(" ", html.unescape(_text(it, "book_description"))).strip()
        # reviews are shown to a human, so collapse the whitespace the tag strip
        # leaves behind (desc is left byte-identical — it feeds the embeddings)
        review = _WS.sub(" ", _TAG.sub(" ", html.unescape(
            _text(it, "user_review")))).strip()
        cover = (_text(it, "book_large_image_url").strip()
                 or _text(it, "book_image_url").strip()
                 or _text(it, "book_medium_image_url").strip())
        books.append({
            "book_id": _text(it, "book_id").strip(),
            "title": _text(it, "title").strip(),
            "author": _text(it, "author_name").strip(),
            "myrating": _text(it, "user_rating").strip(),
            "avg": _text(it, "average_rating").strip(),
            "shelves": _text(it, "user_shelves").strip(),
            "desc": desc[:700],
            "read_at": _rfc822_date(_text(it, "user_read_at")),
            "date_added": _rfc822_ts(_text(it, "user_date_added")),
            "review": review[:4000] or None,
            "isbn": _text(it, "isbn").strip() or None,
            "num_pages": _int(_text(it, "book/num_pages")),
            "published": _int(_text(it, "book_published")),
            "cover_url": cover or None,
        })
    return books


# Fields worth carrying over from a losing duplicate edition: the two Goodreads
# editions of one book hold COMPLEMENTARY metadata (one has the ISBN, the other
# the read date), so a merge keeps more than a pick.
_MERGEABLE = ("read_at", "isbn", "review", "num_pages", "published", "cover_url",
              "shelves", "desc")


def _edition_rank(b: dict) -> tuple:
    """Higher sorts first: a read date beats none, then the higher user rating."""
    return (1 if b.get("read_at") else 0, float(b.get("myrating") or 0))


def dedupe_editions(books: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse multiple Goodreads editions of one book into a single entry.

    books.shelf_books is UNIQUE (account_id, title, author), so two editions of the
    same book can never both persist. That collapse used to happen SILENTLY in the
    INSERT's ON CONFLICT DO NOTHING, which threw away whichever edition arrived
    second along with any field only it carried. Do it here instead, explicitly:
    keep the best-ranked edition, fill its empty fields from the losers, and return
    the dropped ones so the caller can report them.
    """
    by_key: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    dropped: list[dict] = []
    for b in books:
        key = (b["title"], b.get("author") or "")
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = b
            order.append(key)
            continue
        winner, loser = ((prev, b) if _edition_rank(prev) >= _edition_rank(b)
                         else (b, prev))
        for f in _MERGEABLE:
            if not winner.get(f) and loser.get(f):
                winner[f] = loser[f]
        by_key[key] = winner
        dropped.append(loser)
    return [by_key[k] for k in order], dropped


def fetch_shelf(url_or_id: str, shelf: str = "read", max_pages: int = 20,
                client: httpx.Client | None = None,
                dropped_out: list | None = None) -> list[dict]:
    """Fetch and dedupe every book on a public Goodreads shelf.

    Raises ValueError if the profile is private / empty (no items on page 1) so
    the caller fails loudly rather than returning silent empty recommendations.

    Pass `dropped_out` to receive the duplicate editions collapsed by
    `dedupe_editions` — the caller can then report the collapse instead of
    presenting a shorter shelf than the feed with no explanation.
    """
    user_id = parse_user_id(url_or_id)
    owns = client is None
    client = client or httpx.Client(timeout=30, headers={"User-Agent": _UA})
    books: list[dict] = []
    seen: set[str] = set()
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
    books, dropped = dedupe_editions(books)
    if dropped_out is not None:
        dropped_out.extend(dropped)
    if dropped:
        log.warning("shelf %s/%s: collapsed %d duplicate edition(s) into their "
                    "primary entry: %s", user_id, shelf, len(dropped),
                    ", ".join(f"{d['title']} (book_id {d['book_id']})" for d in dropped))
    return books
