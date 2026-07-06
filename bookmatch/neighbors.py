"""Neighbor-gap recommender — user-based collaborative filtering over the UCSD
Goodreads corpus (876k users, 2.3M books).

The target reader (fetched live from Goodreads) is placed into the corpus's item
space by matching their shelf to book ids. We then find the corpus users nearest
them and surface books those neighbours love that the target does not own — the
"books someone next to you loves but you don't have" insight from June.

Only users who *share* at least one book with the target are ever scored (via the
item-major CSC view), so a query touches a few thousand users, not all 876k.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse

_EPS = 1e-9


@dataclass
class Rec:
    item: int
    score: float
    n_supporters: int  # how many neighbours rated it highly


def _candidate_users(csc: sparse.csc_matrix, target_items: list[int]) -> np.ndarray:
    """Union of users who rated any of the target's items."""
    users = set()
    for it in target_items:
        if 0 <= it < csc.shape[1]:
            users.update(csc.indices[csc.indptr[it]:csc.indptr[it + 1]].tolist())
    return np.fromiter(users, dtype=np.int64, count=len(users))


def recommend(user_item: sparse.csr_matrix, csc: sparse.csc_matrix,
              target_ratings: dict[int, float], *, k_neighbors: int = 50,
              n_recs: int = 20, min_overlap: int = 2, like_threshold: float = 4.0
              ) -> list[Rec]:
    """Rank books for a target reader.

    `user_item` : (n_users, n_items) CSR of ratings.
    `csc`       : the same matrix as CSC (item-major) for candidate lookup.
    `target_ratings` : {item_index: rating} for the matched shelf.
    Returns recs excluding books the target already owns, best first.
    """
    n_items = user_item.shape[1]
    target_items = [i for i in target_ratings if 0 <= i < n_items]
    if not target_items:
        return []

    tvec = sparse.csr_matrix(
        (list(target_ratings[i] for i in target_items),
         ([0] * len(target_items), target_items)),
        shape=(1, n_items), dtype=np.float64)
    tnorm = float(np.sqrt(tvec.multiply(tvec).sum()))
    if tnorm < _EPS:
        return []

    cand = _candidate_users(csc, target_items)
    if cand.size == 0:
        return []

    rows = user_item[cand].astype(np.float64)          # (C, n_items)
    rows_bin = rows.copy(); rows_bin.data[:] = 1.0
    tvec_bin = tvec.copy(); tvec_bin.data[:] = 1.0

    dots = (rows @ tvec.T).toarray().ravel()           # shared-item dot products
    row_norms = np.sqrt(np.asarray(rows.multiply(rows).sum(axis=1)).ravel())
    overlap = (rows_bin @ tvec_bin.T).toarray().ravel()  # shared-item counts

    sim = dots / (row_norms * tnorm + _EPS)
    keep = overlap >= min_overlap
    if not keep.any():
        return []
    sim = np.where(keep, sim, 0.0)

    k = min(k_neighbors, int(keep.sum()))
    top = np.argsort(-sim)[:k]
    nbr_rows = rows[top]                                # (k, n_items)
    weights = sim[top]                                  # (k,)

    # weighted rating sum per item, and count of neighbours who *liked* it
    scores = (sparse.csr_matrix(weights.reshape(1, -1)) @ nbr_rows).toarray().ravel()
    liked = nbr_rows.copy()
    liked.data = (liked.data >= like_threshold).astype(np.float64)
    supporters = np.asarray(liked.sum(axis=0)).ravel()

    scores[list(target_ratings)] = -np.inf          # never recommend owned books
    order = np.argsort(-scores)[: n_recs * 3]
    recs = [Rec(int(i), float(scores[i]), int(supporters[i]))
            for i in order if scores[i] > 0][:n_recs]
    return recs
