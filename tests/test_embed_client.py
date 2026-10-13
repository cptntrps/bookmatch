"""Preserve embedding inputs, ordering and aggregation without provider calls."""
from unittest.mock import call, patch

import numpy as np
import pytest

from bookmatch import embed


def test_books_keep_text_and_row_order():
    books = [
        {"title": "First", "author": "A", "desc": "x" * 501},
        {"title": "Second", "author": "B", "desc": None},
    ]
    with patch.object(embed, "gateway_embed", side_effect=[[1, 2], [3, 4]]) as gateway:
        result = embed.embed_books(books)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])
    assert result.dtype == float
    assert gateway.call_args_list == [
        call("First by A. " + "x" * 500, profile="raw-v1", model=embed.EMBED_MODEL),
        call("Second by B. ", profile="raw-v1", model=embed.EMBED_MODEL),
    ]


def test_terms_keep_sequential_calls_and_mean():
    with patch.object(embed, "gateway_embed", side_effect=[[1, 3], [3, 5]]) as gateway:
        result = embed.embed_terms(["MiXeD", "second"])
    np.testing.assert_array_equal(result, [2, 4])
    assert gateway.call_args_list == [
        call("MiXeD", profile="raw-v1", model=embed.EMBED_MODEL),
        call("second", profile="raw-v1", model=embed.EMBED_MODEL),
    ]


def test_failure_stops_before_next_request():
    with patch.object(embed, "gateway_embed", side_effect=RuntimeError("fixture failure")) as gateway:
        with pytest.raises(RuntimeError, match="fixture failure"):
            embed.embed_terms(["first", "second"])
    assert gateway.call_count == 1
