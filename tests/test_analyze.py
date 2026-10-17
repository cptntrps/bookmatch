"""Unit tests for the taste math — synthetic geometry, no Ollama needed."""
import numpy as np
import pytest
from bookmatch import analyze


def test_taste_axis_points_toward_loved():
    # Two clusters: axis 0 = "loved" pole, axis 1 = "disliked" pole.
    E = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], float)
    my = np.array([5, 5, 1, 1], float)  # loves the first cluster
    proj = analyze.taste_axis(E, my)
    # loved books must project higher than disliked ones
    assert proj[0] > proj[2] and proj[1] > proj[3]


def test_taste_axis_needs_two_ratings():
    E = np.random.default_rng(0).normal(size=(5, 8))
    my = np.array([5, 0, 0, 0, 0], float)
    assert np.allclose(analyze.taste_axis(E, my), 0)


def test_contrarian_sign():
    books = [
        {"myrating": 5, "avg": 3.0},  # she >> world
        {"myrating": 2, "avg": 4.5},  # world >> her
    ]
    d = analyze.contrarian(books)
    assert d[0] > 0 and d[1] < 0


def test_project_axis_ranks_positive_pole_first():
    # axis pointing along dim 0; books further along dim 0 rank higher
    E = np.array([[2, 0], [1, 0], [-1, 0], [-2, 0]], float)
    axis = analyze.axis_from_vectors(np.array([1.0, 0.0]))
    proj = analyze.project_axis(E, axis)
    assert list(np.argsort(-proj)) == [0, 1, 2, 3]


def test_axis_from_vectors_contrast_and_unit():
    pos = np.array([1.0, 0.0]); neg = np.array([0.0, 1.0])
    axis = analyze.axis_from_vectors(pos, neg)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    # contrast axis points from neg toward pos
    assert axis[0] > 0 and axis[1] < 0


def test_antipode_flags_unloved_region():
    E = np.array([[1, 0], [0.9, 0.1], [0, 1]], float)
    my = np.array([5, 5, 0], float)  # loves cluster near dim 0
    ap = analyze.antipode(E, my)
    # the book near dim 1 is least like what she loves -> highest antipode score
    assert np.argmax(ap) == 2


def test_project_axis_with_shared_center_keeps_frames_aligned():
    E = np.array([[2, 0], [0, 0]], float)             # shelf mean = (1, 0)
    C = np.array([[1.5, 0], [0.5, 0]], float)         # candidates
    axis = analyze.axis_from_vectors(np.array([1.0, 0.0]))
    cproj = analyze.project_axis(C, axis, center=E.mean(0))
    # measured from the SHELF mean: 1.5 is +0.5, 0.5 is -0.5 (self-centering would give ±0.5 too,
    # but shifted candidates must not be re-centered on themselves)
    assert np.allclose(cproj, [0.5, -0.5])
    assert np.allclose(analyze.project_axis(C + 10, axis, center=E.mean(0)), [10.5, 9.5])


def test_axis_rating_agreement_sign_and_honest_unknown():
    proj = np.arange(8, dtype=float)
    my = np.array([1, 2, 2, 3, 4, 4, 5, 5], float)
    assert analyze.axis_rating_agreement(proj, my)["spearman"] > 0.9
    assert analyze.axis_rating_agreement(-proj, my)["spearman"] < -0.9
    # unrated books are ignored; too few rated -> None, never a fake 0
    few = analyze.axis_rating_agreement(proj, np.array([5, 1, 0, 0, 0, 0, 0, 0], float))
    assert few == {"spearman": None, "n_rated": 2}
    flat = analyze.axis_rating_agreement(proj, np.full(8, 4.0))
    assert flat["spearman"] is None and flat["n_rated"] == 8
    assert "Too few" in analyze.describe_agreement(few, "x")
    assert "toward the 'x' pole" in analyze.describe_agreement({"spearman": 0.5, "n_rated": 9}, "x")
    assert "away from" in analyze.describe_agreement({"spearman": -0.5, "n_rated": 9}, "x")
    assert "no relation" in analyze.describe_agreement({"spearman": 0.02, "n_rated": 9}, "x")


def test_axis_positions_shared_scale_and_order():
    books = [{"title": "far", "author": "a", "myrating": 5}, {"title": "mid", "author": "a", "myrating": 3},
             {"title": "near", "author": "a", "myrating": 1}]
    E = np.array([[2, 0], [0, 0], [-2, 0]], float)
    cands = [{"title": "cand", "author": "c"}]
    C = np.array([[4.0, 0.0]])                      # further out than any shelf book
    r = analyze.axis_positions(books, E, np.array([1.0, 0.0]), None, "x", cands, C)
    assert [b["title"] for b in r["shelf"]] == ["far", "mid", "near"]
    assert r["candidates"][0]["x"] == 1.0           # the scale is the largest of BOTH sets
    assert r["shelf"][0]["x"] == 0.5 and r["shelf"][-1]["x"] == -0.5
    assert r["rating_agreement"]["n_rated"] == 3 and "reading" in r["rating_agreement"]
    assert analyze.axis_positions(books, E, np.array([1.0, 0.0]), None, "x")["candidates"] == []


def test_content_axes_shape():
    E = np.random.default_rng(1).normal(size=(20, 16))
    V = analyze.content_axes(E, k=3)
    assert V.shape == (3, 16)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-6)
