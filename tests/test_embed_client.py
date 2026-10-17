"""Embedding client: inputs, ordering, batching and failure, without a provider."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bookmatch import embed


def _fake_client(vectors_by_text: dict, calls: list):
    def create(model, input):
        calls.append(list(input))
        data = [SimpleNamespace(index=i, embedding=vectors_by_text[t])
                for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))   # out of order on purpose
    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def test_books_keep_text_and_row_order():
    books = [
        {"title": "First", "author": "A", "desc": "x" * 501},
        {"title": "Second", "author": "B", "desc": None},
    ]
    calls: list = []
    fake = _fake_client({"First by A. " + "x" * 500: [1, 2], "Second by B. ": [3, 4]}, calls)
    with patch.object(embed, "_client", return_value=fake):
        result = embed.embed_books(books)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])   # re-sorted by index
    assert result.dtype == float
    assert calls == [["First by A. " + "x" * 500, "Second by B. "]]


def test_terms_are_sent_raw_and_averaged():
    calls: list = []
    fake = _fake_client({"MiXeD": [1, 3], "second": [3, 5]}, calls)
    with patch.object(embed, "_client", return_value=fake):
        result = embed.embed_terms(["MiXeD", "second"])
    np.testing.assert_array_equal(result, [2, 4])
    assert calls == [["MiXeD", "second"]]              # no lowercasing, no prefix


def test_batches_respect_limit():
    texts = [f"t{i}" for i in range(embed.BATCH + 1)]
    calls: list = []
    fake = _fake_client({t: [i] for i, t in enumerate(texts)}, calls)
    with patch.object(embed, "_client", return_value=fake):
        out = embed.embed_texts(texts)
    assert [len(c) for c in calls] == [embed.BATCH, 1]
    np.testing.assert_array_equal(out.ravel(), np.arange(len(texts)))


def test_short_answer_fails_loud():
    def create(model, input):
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])
    fake = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with patch.object(embed, "_client", return_value=fake):
        with pytest.raises(RuntimeError, match="asked for 2"):
            embed.embed_texts(["a", "b"])


def test_empty_shelf_embeds_to_empty_matrix():
    assert embed.embed_books([]).shape == (0, 0)
