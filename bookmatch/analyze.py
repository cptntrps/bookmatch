"""Taste analysis over an embedded shelf — pure geometry, no inference.

The math is a direct port of the June-2026 scratch toolkit
(`reader_a_analysis.py`, `reader_a_french.py`). Every function here takes a
pre-computed embedding matrix so it is fast and unit-testable without Ollama;
`embed.py` owns the actual embedding calls.

Conventions: `E` is an (n, d) float array of L2-comparable book embeddings,
one row per book in the same order as the `books` metadata list.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-9


def _norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + _EPS)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + _EPS)


def ratings(books: list[dict]) -> np.ndarray:
    return np.array([float(b.get("myrating") or 0) for b in books], float)


def averages(books: list[dict]) -> np.ndarray:
    return np.array([float(b.get("avg") or 0) for b in books], float)


def taste_axis(E: np.ndarray, my: np.ndarray) -> np.ndarray:
    """The rating-weighted taste direction in centered, row-normalized space.

    Books the reader rated above their own mean pull the axis toward what they
    love; below-mean books push the other way. Returns the projection of every
    book onto that unit axis (higher = more like what they love). Books with no
    rating still get a projection (useful for ranking unrated/ to-read shelves).
    """
    En = _norm_rows(E - E.mean(0))
    rated = my > 0
    if rated.sum() < 2:
        return np.zeros(len(E))
    centered = (my[rated] - my[rated].mean())[:, None]
    td = _unit((centered * En[rated]).sum(0))
    return En @ td


def contrarian(books: list[dict]) -> np.ndarray:
    """Per-book (myrating - community average). Positive = they love it more
    than the world; negative = the world loves it more than they do."""
    return ratings(books) - averages(books)


def axis_from_vectors(pos: np.ndarray, neg: np.ndarray | None = None) -> np.ndarray:
    """Build a unit semantic axis from concept embeddings. With a negative pole
    the axis is the contrast (pos - neg); without one it is the concept itself
    (the caller centers the book cloud in `project_axis`)."""
    axis = pos if neg is None else (pos - neg)
    return _unit(axis)


def project_axis(E: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project every book onto a semantic axis in mean-centered space.

    This is the "make my shelf more french" operation: one dot product per book,
    unsupervised — nobody labels which books are french. Returns the projection
    (higher = more toward the positive pole of the axis)."""
    return (E - E.mean(0)) @ axis


def content_axes(E: np.ndarray, k: int = 3) -> np.ndarray:
    """Top-k principal content axes (SVD) of the centered, row-normalized cloud.
    Returns a (k, d) array of unit row-vectors — the shelf's own latent themes."""
    En = _norm_rows(E - E.mean(0))
    _, _, Vt = np.linalg.svd(En, full_matrices=False)
    return Vt[:k]


def antipode(E: np.ndarray, my: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    """Projection of every book onto the (negated) centroid of loved books.
    Highest values = least like what they love — the taste blind spot."""
    En = _norm_rows(E - E.mean(0))
    loved_mask = my >= threshold
    if loved_mask.sum() == 0:
        return np.zeros(len(E))
    loved = _unit(En[loved_mask].mean(0))
    return -(En @ loved)
