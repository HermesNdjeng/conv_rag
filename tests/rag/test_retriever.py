from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from redisvl.exceptions import RedisSearchError

from rag import retriever as retriever_module
from rag.retriever import DocumentRetriever


def _store(*scored_docs: tuple[str, float]) -> MagicMock:
    """A fake vector store; args are (text, relevance) — stored as the cosine distance
    (1 - relevance) that langchain_redis actually returns from similarity_search_with_score."""
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (Document(page_content=text), 1.0 - relevance) for text, relevance in scored_docs
    ]
    return store


def test_retrieve_drops_docs_below_threshold(
    retriever: DocumentRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    # threshold is 0.5 (set in the fixture)
    monkeypatch.setattr(
        retriever_module.RedisVectorStore,
        "from_existing_index",
        lambda **_: _store(("keep", 0.9), ("drop", 0.3), ("edge", 0.5)),
    )

    docs = retriever.retrieve("q", ["global"])

    assert [d.page_content for d in docs] == ["keep", "edge"]


def test_retrieve_sorts_by_score_descending(
    retriever: DocumentRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        retriever_module.RedisVectorStore,
        "from_existing_index",
        lambda **_: _store(("mid", 0.7), ("top", 0.95), ("low", 0.6)),
    )

    docs = retriever.retrieve("q", ["global"])

    assert [d.page_content for d in docs] == ["top", "mid", "low"]


def test_retrieve_skips_unavailable_index(
    retriever: DocumentRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    def from_existing_index(**kwargs: object) -> MagicMock:
        if kwargs["index_name"] == "missing":
            raise RedisSearchError("Unknown index name")
        return _store(("hit", 0.8))

    monkeypatch.setattr(
        retriever_module.RedisVectorStore, "from_existing_index", from_existing_index
    )

    docs = retriever.retrieve("q", ["missing", "global"])

    assert [d.page_content for d in docs] == ["hit"]
