"""The single-page UI: the French preset is gone and the How it works drawer ships.

Static checks on static/index.html, plus the page as FastAPI serves it."""
from pathlib import Path

from fastapi.testclient import TestClient

from bookmatch.app import app

PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"


def test_no_french_axis_control():
    html = PAGE.read_text()
    assert "french" not in html.lower()


def test_presets_still_offer_axes():
    html = PAGE.read_text()
    assert "Melancholy vs comic" in html


def test_how_it_works_drawer_has_every_section():
    html = PAGE.read_text()
    assert 'id="howBtn"' in html and 'id="how"' in html
    for heading in ("What it reads", "Where the candidates come from", "What an axis is",
                    "The check under the axis", "How your verdicts change results", "Limits"):
        assert f"<h3>{heading}</h3>" in html, heading


def test_root_serves_the_page_with_the_drawer():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "How it works" in r.text and "French" not in r.text
