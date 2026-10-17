"""Embeddings through any OpenAI-compatible `/v1/embeddings` endpoint.

Default: a local Ollama (`http://127.0.0.1:11434/v1`) serving `nomic-embed-text`
(768-d). Texts are sent raw: no prefix, no lowercasing, so vectors stored by
earlier versions stay comparable. Kept separate from analyze.py so the geometry
stays pure and unit-testable.

Environment:
  BOOKMATCH_EMBED_URL    base URL (default Ollama); any OpenAI-schema server works
  BOOKMATCH_EMBED_KEY    bearer key (default "ollama"; Ollama ignores it)
  BOOKMATCH_EMBED_MODEL  model name (default nomic-embed-text)
"""
from __future__ import annotations
import os
import numpy as np
from openai import OpenAI

EMBED_URL = os.environ.get("BOOKMATCH_EMBED_URL", "http://127.0.0.1:11434/v1")
EMBED_KEY = (os.environ.get("BOOKMATCH_EMBED_KEY")
             or os.environ.get("BOOKMATCH_GATEWAY_KEY") or "ollama")
EMBED_MODEL = os.environ.get("BOOKMATCH_EMBED_MODEL", "nomic-embed-text")
BATCH = 64

_client_: OpenAI | None = None


def _client() -> OpenAI:
    global _client_
    if _client_ is None:
        _client_ = OpenAI(base_url=EMBED_URL, api_key=EMBED_KEY, timeout=120)
    return _client_


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts in order into an (n, d) float matrix. A short or empty answer
    raises: a missing vector must never become a silent zero row."""
    rows: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        resp = _client().embeddings.create(model=EMBED_MODEL, input=chunk)
        data = sorted(resp.data, key=lambda d: d.index)
        if len(data) != len(chunk):
            raise RuntimeError(f"embeddings: asked for {len(chunk)}, got {len(data)}")
        rows.extend(d.embedding for d in data)
    return np.asarray(rows, dtype=float)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def book_text(book: dict) -> str:
    return f"{book['title']} by {book.get('author', '')}. {(book.get('desc') or '')[:500]}"


def embed_books(books: list[dict]) -> np.ndarray:
    """Embed a list of book dicts into an (n, 768) matrix, one row per book."""
    if not books:
        return np.zeros((0, 0))
    return embed_texts([book_text(b) for b in books])


def embed_terms(terms: list[str]) -> np.ndarray:
    """Mean embedding of a list of concept phrases: one pole of a semantic axis."""
    return embed_texts(list(terms)).mean(axis=0)


def available() -> bool:
    """True when the embedding endpoint answers a real request. Tests use this to
    skip honestly instead of guessing from a URL."""
    try:
        return embed_texts(["ping"]).shape[0] == 1
    except Exception:
        return False
