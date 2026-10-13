"""LivingOS embedding client (nomic-embed-text, raw-v1, 768d).

Kept separate from analyze.py so the geometry stays pure and unit-testable.
"""
from __future__ import annotations
import os
import numpy as np
from livingos_client import embed as gateway_embed

EMBED_MODEL = os.environ.get("BOOKMATCH_EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str) -> np.ndarray:
    """Raw legacy profile preserves existing book vectors."""
    return np.asarray(gateway_embed(text, profile="raw-v1", model=EMBED_MODEL), dtype=float)


def book_text(book: dict) -> str:
    return f"{book['title']} by {book['author']}. {(book.get('desc') or '')[:500]}"


def embed_books(books: list[dict]) -> np.ndarray:
    """Embed a list of book dicts into an (n, 768) matrix, one row per book."""
    rows = [embed_text(book_text(b)) for b in books]
    return np.vstack(rows)


def embed_terms(terms: list[str]) -> np.ndarray:
    """Mean embedding of a list of concept phrases — one pole of a semantic axis."""
    rows = [embed_text(t) for t in terms]
    return np.mean(np.vstack(rows), axis=0)
