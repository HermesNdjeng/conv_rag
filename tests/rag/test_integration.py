"""Round-trip integration tests against a real Redis Stack.

Skipped automatically unless a Redis Stack (RediSearch) is reachable at REDIS_URL.
Run explicitly with:  pytest -m integration
"""

# TODO: add an integration test for the ingestion pipeline (rag.pipeline.ingest_bucket):
# ingest a real MinIO bucket into Redis, assert the index, remove an object, re-sync,
# and assert the orphan is pruned. The unit tests only cover the control-flow branches.


import contextlib
import os

import pytest
import redis as redis_lib
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.indexer import RedisIndexer
from rag.retriever import DocumentRetriever
from rag.schemas import RetrieverConfig, VectorStoreConfig


pytestmark = pytest.mark.integration

_INDEX = "test_rag_integration"


@pytest.fixture(scope="module")
def redis_url() -> str:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        client = redis_lib.from_url(url)
        client.ping()
        client.execute_command("FT._LIST")  # RediSearch present?
    except Exception:
        pytest.skip(f"Redis Stack not reachable at {url}")
    return url


def test_index_retrieve_delete_roundtrip(redis_url: str) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    indexer = RedisIndexer(
        VectorStoreConfig(redis_url=redis_url, embedding_model_name="unused"),
        embeddings=embeddings,
    )
    retriever = DocumentRetriever(
        RetrieverConfig(redis_url=redis_url, embedding_model_name="unused"),
        embeddings=embeddings,
    )

    query = "Ernest Ouandié led the UPC."
    docs = [
        Document(page_content=query, metadata={"document_id": "doc_a", "chunk_index": 0}),
        Document(
            page_content="Cameroon gained independence in 1960.",
            metadata={"document_id": "doc_b", "chunk_index": 0},
        ),
    ]

    try:
        indexer.upsert(docs, _INDEX)

        found = {d.metadata["document_id"] for d in retriever.retrieve(query, [_INDEX])}
        assert "doc_a" in found

        indexer.delete_by_metadata(_INDEX, {"document_id": "doc_a"})

        after = {d.metadata["document_id"] for d in retriever.retrieve(query, [_INDEX])}
        assert "doc_a" not in after
    finally:
        with contextlib.suppress(Exception):
            indexer._client.ft(_INDEX).dropindex(delete_documents=True)
