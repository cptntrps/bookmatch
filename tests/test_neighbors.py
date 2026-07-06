"""Unit tests for the neighbor-gap engine on a synthetic corpus."""
import numpy as np
import pytest
from scipy import sparse
from bookmatch import neighbors


def _corpus():
    # 5 users x 6 books. Books 0-2 = "cluster A", books 3-5 = "cluster B".
    #  u0,u1 love A and also love book 2 & book 5 respectively.
    #  u2,u3 love B.  u4 is noise.
    R = np.array([
        [5, 5, 5, 0, 0, 0],   # u0: pure A lover
        [5, 5, 0, 0, 0, 5],   # u1: A lover who also loves book 5
        [0, 0, 0, 5, 5, 5],   # u2: pure B lover
        [1, 0, 0, 5, 5, 4],   # u3: B lover
        [0, 0, 0, 0, 0, 1],   # u4: noise
    ], float)
    csr = sparse.csr_matrix(R)
    return csr, csr.tocsc()


def test_recommends_neighbor_loved_book_not_owned():
    csr, csc = _corpus()
    # target loves cluster A books 0 and 1 (owns them), lacks book 2.
    target = {0: 5.0, 1: 5.0}
    recs = neighbors.recommend(csr, csc, target, k_neighbors=3, n_recs=5, min_overlap=2)
    rec_items = [r.item for r in recs]
    # nearest neighbour u0 loves book 2 -> it should be the top rec
    assert 2 in rec_items
    assert recs[0].item == 2


def test_excludes_owned_books():
    csr, csc = _corpus()
    target = {0: 5.0, 1: 5.0}
    recs = neighbors.recommend(csr, csc, target, min_overlap=2)
    rec_items = {r.item for r in recs}
    assert 0 not in rec_items and 1 not in rec_items


def test_min_overlap_filters_thin_matches():
    csr, csc = _corpus()
    # target shares only ONE book with anyone -> min_overlap=2 yields nothing
    target = {0: 5.0}
    recs = neighbors.recommend(csr, csc, target, min_overlap=2)
    assert recs == []


def test_empty_target_returns_empty():
    csr, csc = _corpus()
    assert neighbors.recommend(csr, csc, {}, min_overlap=1) == []
    # items outside the corpus are ignored
    assert neighbors.recommend(csr, csc, {999: 5.0}, min_overlap=1) == []


def test_supporter_count_populated():
    csr, csc = _corpus()
    target = {3: 5.0, 4: 5.0}   # a B-cluster reader
    recs = neighbors.recommend(csr, csc, target, min_overlap=2)
    assert recs and all(r.n_supporters >= 0 for r in recs)
    # book 5 is loved across B users -> should surface with supporters
    five = [r for r in recs if r.item == 5]
    assert five and five[0].n_supporters >= 1
