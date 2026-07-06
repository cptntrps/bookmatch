"""Tests for the match-key normalization — the lossy corpus join."""
import pytest
from bookmatch import match


@pytest.mark.parametrize("title,expected", [
    ("The Lover", "lover"),
    ("Perfume: The Story of a Murderer", "perfume"),
    ("If on a Winter's Night a Traveler", "if on a winters night a traveler"),
    ("Harry Potter and the Sorcerer's Stone (Harry Potter, #1)", "harry potter and the sorcerers stone"),
    ("A Little Life", "little life"),
])
def test_norm_title(title, expected):
    assert match.norm_title(title) == expected


@pytest.mark.parametrize("author,expected", [
    ("Marguerite Duras", "duras"),
    ("Tolkien, J.R.R.", "tolkien"),
    ("J.R.R. Tolkien", "tolkien"),
    ("Gabriel García Márquez", "marquez"),
    ("Neil Gaiman & Terry Pratchett", "gaiman"),
])
def test_author_surname(author, expected):
    assert match.author_surname(author) == expected


def test_normkey_stable_across_variants():
    # same book, different renderings, must collide
    a = match.normkey("The Lord of the Rings: The Fellowship of the Ring", "J.R.R. Tolkien")
    b = match.normkey("Lord of the Rings (The Fellowship of the Ring)", "Tolkien, J.R.R.")
    assert a == b


def test_normkey_accent_insensitive():
    assert (match.normkey("Cien años de soledad", "Gabriel García Márquez")
            == match.normkey("Cien anos de soledad", "Gabriel Garcia Marquez"))
