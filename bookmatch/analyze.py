"""Taste analysis over an embedded shelf — pure geometry, no inference.

The math is a direct port of the June-2026 scratch toolkit. Every function here takes a
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


def project_axis(E: np.ndarray, axis: np.ndarray,
                 center: np.ndarray | None = None) -> np.ndarray:
    """Project every book onto a semantic axis in mean-centered space.

    This is the "make my shelf more french" operation: one dot product per book,
    unsupervised — nobody labels which books are french. Returns the projection
    (higher = more toward the positive pole of the axis).

    `center` lets a second set of books (candidate recommendations) be projected
    in the same frame as the shelf: pass the shelf mean. Without it the set is
    centered on itself."""
    c = E.mean(0) if center is None else center
    return (E - c) @ axis


def axis_rating_agreement(proj: np.ndarray, my: np.ndarray, min_rated: int = 5) -> dict:
    """How much the reader's own ratings follow an axis.

    Spearman rank correlation between the axis projection and the reader's rating,
    over rated books only. Positive: they rate books nearer the positive pole
    higher. Returns {"spearman": float | None, "n_rated": int}. `None` means
    "cannot tell" (too few rated books, or no variance), never a fake zero."""
    rated = my > 0
    n = int(rated.sum())
    if n < min_rated:
        return {"spearman": None, "n_rated": n}
    x, y = proj[rated], my[rated]
    if np.ptp(x) < _EPS or np.ptp(y) < _EPS:
        return {"spearman": None, "n_rated": n}
    from scipy.stats import spearmanr
    res = spearmanr(x, y)
    rho = float(getattr(res, "statistic", res[0]))
    return {"spearman": round(rho, 3), "n_rated": n}


def describe_agreement(agreement: dict, label: str) -> str:
    """One plain sentence a chat model can quote about an axis-rating agreement."""
    rho, n = agreement.get("spearman"), agreement.get("n_rated", 0)
    if rho is None:
        return f"Too few rated books ({n}) to tell whether ratings follow the '{label}' axis."
    strength = "no" if abs(rho) < 0.1 else "a weak" if abs(rho) < 0.3 else "a clear"
    if strength == "no":
        return f"Ratings show no relation to the '{label}' axis (Spearman {rho:+.2f}, n={n})."
    side = "toward" if rho > 0 else "away from"
    return (f"The reader rates books {side} the '{label}' pole higher: "
            f"{strength} relation (Spearman {rho:+.2f}, n={n}).")


def axis_positions(books: list[dict], E: np.ndarray, pos: np.ndarray,
                   neg: np.ndarray | None, label: str,
                   candidates: list[dict] | None = None,
                   C: np.ndarray | None = None) -> dict:
    """The one projection routine behind the chat tool, the CLI and the UI.

    Builds the axis from the pole embeddings, projects the shelf and (optionally)
    the candidates in the shelf's frame, scales positions to [-1, 1] by the largest
    absolute projection across both sets, and attaches the rating agreement.
    Lists come sorted from the positive pole down."""
    axis = axis_from_vectors(pos, neg)
    center = E.mean(0)
    proj = project_axis(E, axis, center)
    cproj = (project_axis(C, axis, center)
             if candidates and C is not None and len(C) else np.zeros(0))
    scale = max(float(np.abs(proj).max()) if len(proj) else 0.0,
                float(np.abs(cproj).max()) if len(cproj) else 0.0, _EPS)
    agreement = axis_rating_agreement(proj, ratings(books))
    shelf = [{"title": books[i]["title"], "author": books[i].get("author", ""),
              "myrating": float(books[i].get("myrating") or 0),
              "x": round(float(proj[i] / scale), 4)} for i in np.argsort(-proj)]
    cands = [{"title": candidates[i]["title"], "author": candidates[i].get("author", ""),
              "x": round(float(cproj[i] / scale), 4)} for i in np.argsort(-cproj)]
    return {"label": label, "shelf": shelf, "candidates": cands,
            "rating_agreement": {**agreement, "reading": describe_agreement(agreement, label)}}


def content_axes(E: np.ndarray, k: int = 3) -> np.ndarray:
    """Top-k principal content axes (SVD) of the centered, row-normalized cloud.
    Returns a (k, d) array of unit row-vectors — the shelf's own latent themes."""
    En = _norm_rows(E - E.mean(0))
    _, _, Vt = np.linalg.svd(En, full_matrices=False)
    return Vt[:k]


def _titles(books: list[dict], idx) -> list[dict]:
    return [{"title": books[i]["title"], "author": books[i].get("author", ""),
             "myrating": float(books[i].get("myrating") or 0),
             "avg": float(books[i].get("avg") or 0)} for i in idx]


def build_profile(books: list[dict], E: np.ndarray, n: int = 8) -> dict:
    """Assemble the serializable taste fingerprint the UI and chat agent consume."""
    my = ratings(books)
    rated = np.where(my > 0)[0]
    proj = taste_axis(E, my)
    order = np.argsort(-proj)
    loved = [i for i in order if my[i] > 0][:n]
    tolerated = [i for i in order[::-1] if my[i] > 0][:n]

    d = contrarian(books)
    do = np.argsort(-d)
    ap = antipode(E, my)

    return {
        "n_books": len(books),
        "n_rated": int(len(rated)),
        "mean_rating": round(float(my[rated].mean()), 2) if len(rated) else None,
        "loved": _titles(books, loved),
        "tolerated": _titles(books, tolerated),
        "loves_more_than_world": _titles(books, [i for i in do if my[i] > 0][:n]),
        "world_loves_more": _titles(books, [i for i in do[::-1] if my[i] > 0][:n]),
        "blind_spot": _titles(books, list(np.argsort(-ap)[:n])),
    }


def antipode(E: np.ndarray, my: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    """Projection of every book onto the (negated) centroid of loved books.
    Highest values = least like what they love — the taste blind spot."""
    En = _norm_rows(E - E.mean(0))
    loved_mask = my >= threshold
    if loved_mask.sum() == 0:
        return np.zeros(len(E))
    loved = _unit(En[loved_mask].mean(0))
    return -(En @ loved)
