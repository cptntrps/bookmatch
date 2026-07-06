"""Normalize (title, author) into a match key.

The live target comes from Goodreads RSS as free text; the corpus stores its own
titles/authors. Matching is the one lossy joint in the system, so the same
normalization must be used on both sides — build_index.py and query time both
call `normkey`.

Strategy: lowercase, drop a trailing subtitle / series parenthetical, strip
punctuation, collapse whitespace, and key on `title|author-surname`. Keying on
the author *surname* (not the full name) absorbs "J.R.R. Tolkien" vs "Tolkien,
J.R.R." vs "John Ronald Reuel Tolkien" style variation.
"""
from __future__ import annotations
import re
import unicodedata

_APOS = re.compile(r"['’]")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
# a subtitle after ':' or a trailing "(Series #n)" parenthetical
_SUBTITLE = re.compile(r"\s*[:(].*$")
_ARTICLE = re.compile(r"^(the|a|an)\s+")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def norm_title(title: str) -> str:
    t = _fold(title)
    t = _SUBTITLE.sub("", t)          # drop ": subtitle" and "(Series #1)"
    t = _APOS.sub("", t)              # sorcerer's -> sorcerers (join, don't split)
    t = _PUNCT.sub(" ", t)
    t = _ARTICLE.sub("", t)           # leading article rarely disambiguates
    return _WS.sub(" ", t).strip()


def author_surname(author: str) -> str:
    a = _fold(author)
    if "," in a:                       # "Tolkien, J.R.R." -> surname first
        a = a.split(",")[0]
    else:
        a = a.split("&")[0].split(" and ")[0]  # first of multiple authors
        parts = [p for p in _PUNCT.sub(" ", a).split() if len(p) > 1]
        a = parts[-1] if parts else a  # last token = surname
    a = _PUNCT.sub(" ", a)
    return _WS.sub(" ", a).strip()


def normkey(title: str, author: str) -> str:
    return f"{norm_title(title)}|{author_surname(author)}"
