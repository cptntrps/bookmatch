"""Ollama embedding wrapper (nomic-embed-text, 768d).

Kept separate from analyze.py so the geometry stays pure and unit-testable.
"""
from __future__ import annotations
import os
import numpy as np
import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("BOOKMATCH_EMBED_MODEL", "nomic-embed-text")
_TIMEOUT = 60.0


def embed_text(text: str, client: httpx.Client | None = None) -> np.ndarray:
    owns = client is None
    client = client or httpx.Client(timeout=_TIMEOUT)
    try:
        r = client.post(f"{OLLAMA_URL}/api/embeddings",
                        json={"model": EMBED_MODEL, "prompt": text})
        r.raise_for_status()
        return np.asarray(r.json()["embedding"], dtype=float)
    finally:
        if owns:
            client.close()


def book_text(book: dict) -> str:
    return f"{book['title']} by {book['author']}. {(book.get('desc') or '')[:500]}"


def embed_books(books: list[dict]) -> np.ndarray:
    """Embed a list of book dicts into an (n, 768) matrix, one row per book."""
    with httpx.Client(timeout=_TIMEOUT) as client:
        rows = [embed_text(book_text(b), client) for b in books]
    return np.vstack(rows)


def embed_terms(terms: list[str]) -> np.ndarray:
    """Mean embedding of a list of concept phrases — one pole of a semantic axis."""
    with httpx.Client(timeout=_TIMEOUT) as client:
        rows = [embed_text(t, client) for t in terms]
    return np.mean(np.vstack(rows), axis=0)
