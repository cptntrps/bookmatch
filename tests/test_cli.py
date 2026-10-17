"""Axis report and its formatting on synthetic geometry: no services."""
import numpy as np

from bookmatch import analyze
from bookmatch.cli import axis_report, format_profile, format_report


def _shelf():
    # dim 0 = "french-ness"; ratings rise with it, so the agreement must be positive
    books = [{"title": f"B{i}", "author": "a", "myrating": r, "avg": 3.5}
             for i, r in enumerate([1, 2, 3, 4, 5, 5])]
    E = np.array([[-2, 0.1], [-1, 0.0], [0, 0.2], [1, 0.1], [2, 0.0], [2.5, 0.1]], float)
    return books, E


def test_axis_report_ranks_candidates_in_shelf_frame_and_measures_agreement():
    books, E = _shelf()
    pos, neg = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    cands = [{"title": "C-far", "author": "c"}, {"title": "C-near", "author": "c"}]
    C = np.array([[5.0, 0.0], [-5.0, 0.0]])
    r = axis_report(books, E, pos, neg, "french", cands, C, top=3)
    assert r["positive_pole"][0].startswith("B5") and r["negative_pole"][0].startswith("B0")
    assert r["candidates_toward_positive"][0].startswith("C-far")
    assert r["candidates_toward_negative"][0].startswith("C-near")
    assert r["rating_agreement"]["spearman"] > 0.9 and r["rating_agreement"]["n_rated"] == 6
    assert "toward the 'french' pole higher" in r["rating_agreement"]["reading"]


def test_report_without_candidates_has_no_candidate_keys():
    books, E = _shelf()
    r = axis_report(books, E, np.array([1.0, 0.0]), None, "x")
    assert "candidates_toward_positive" not in r


def test_formatting_mentions_titles_and_reading():
    books, E = _shelf()
    r = axis_report(books, E, np.array([1.0, 0.0]), None, "french", top=2)
    text = format_report(r)
    assert "axis 'french'" in text and "B5" in text and "Spearman" in text
    prof = format_profile(analyze.build_profile(books, E, n=2))
    assert "loves most" in prof and "blind spot" in prof
