from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from redis.exceptions import ResponseError

from rag import indexer as indexer_module
from rag.exceptions import EmbeddingModelMismatchError
from rag.indexer import RedisIndexer


def _page(*ids: str) -> SimpleNamespace:
    """A fake RediSearch Result carrying docs with the given ids."""
    return SimpleNamespace(docs=[SimpleNamespace(id=i) for i in ids])


def test_upsert_empty_list_returns_zero_without_writing(
    index: RedisIndexer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from_documents = MagicMock()
    monkeypatch.setattr(indexer_module.RedisVectorStore, "from_documents", from_documents)

    result = index.upsert([], "global")

    assert result.index_name == "global"
    assert result.document_count == 0
    from_documents.assert_not_called()


def test_upsert_missing_metadata_raises(index: RedisIndexer) -> None:
    doc = Document(page_content="x", metadata={"document_id": "doc"})  # no chunk_index
    with pytest.raises(ValueError, match="document_id.*chunk_index"):
        index.upsert([doc], "global")


def test_upsert_builds_stable_keys_and_returns_count(
    index: RedisIndexer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from_documents = MagicMock()
    monkeypatch.setattr(indexer_module.RedisVectorStore, "from_documents", from_documents)
    docs = [
        Document(page_content="a", metadata={"document_id": "u/report.md", "chunk_index": 0}),
        Document(page_content="b", metadata={"document_id": "u/report.md", "chunk_index": 1}),
    ]

    result = index.upsert(docs, "user_42")

    assert result.index_name == "user_42"
    assert result.document_count == 2
    assert from_documents.call_args.kwargs["keys"] == ["u/report.md:0", "u/report.md:1"]
    assert from_documents.call_args.kwargs["config"].index_name == "user_42"


def test_delete_empty_where_raises(index: RedisIndexer, client: MagicMock) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        index.delete_by_metadata("user_42", {})
    client.delete.assert_not_called()


def test_delete_combines_filters_with_and(index: RedisIndexer, client: MagicMock) -> None:
    client.ft.return_value.search.return_value = _page()  # no matches, single page

    index.delete_by_metadata("user_42", {"document_id": "u/report.md", "lang": "fr"})

    query = client.ft.return_value.search.call_args.args[0]
    # RedisTag escapes special chars (/, .) in values; AND wraps the terms in parens.
    assert query.query_string() == "(@document_id:{u\\/report\\.md} @lang:{fr})"


def test_delete_calls_delete_with_matching_ids(index: RedisIndexer, client: MagicMock) -> None:
    client.ft.return_value.search.return_value = _page("u/report.md:0", "u/report.md:1")

    index.delete_by_metadata("user_42", {"document_id": "u/report.md"})

    client.delete.assert_called_once_with("u/report.md:0", "u/report.md:1")


def test_delete_no_matches_does_not_delete(index: RedisIndexer, client: MagicMock) -> None:
    client.ft.return_value.search.return_value = _page()

    index.delete_by_metadata("user_42", {"document_id": "missing"})

    client.delete.assert_not_called()


def test_delete_missing_index_returns_silently(index: RedisIndexer, client: MagicMock) -> None:
    client.ft.return_value.search.side_effect = ResponseError("no such index")

    index.delete_by_metadata("user_42", {"document_id": "x"})  # must not raise

    client.delete.assert_not_called()


def test_delete_paginates_until_short_batch(
    index: RedisIndexer, client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(indexer_module, "_DELETE_PAGE_SIZE", 2)
    # Two full pages of 2, then a short page of 1 -> loop stops after the short page.
    client.ft.return_value.search.side_effect = [
        _page("a:0", "a:1"),
        _page("b:0", "b:1"),
        _page("c:0"),
    ]

    index.delete_by_metadata("user_42", {"document_id": "x"})

    assert client.ft.return_value.search.call_count == 3
    offsets = [c.args[0]._offset for c in client.ft.return_value.search.call_args_list]
    assert offsets == [0, 2, 4]
    client.delete.assert_called_once_with("a:0", "a:1", "b:0", "b:1", "c:0")


def test_upsert_refuses_model_mismatch(index: RedisIndexer, client: MagicMock) -> None:
    client.hget.return_value = b"other-model"  # index built with a different embedding model
    doc = Document(page_content="a", metadata={"document_id": "d", "chunk_index": 0})
    with pytest.raises(EmbeddingModelMismatchError):
        index.upsert([doc], "user_42")
