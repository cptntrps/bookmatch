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


def test_content_axes_shape():
    E = np.random.default_rng(1).normal(size=(20, 16))
    V = analyze.content_axes(E, k=3)
    assert V.shape == (3, 16)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-6)
