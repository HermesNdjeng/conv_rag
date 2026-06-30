from unittest.mock import MagicMock

import pytest

from rag.indexer import RedisIndexer
from rag.retriever import DocumentRetriever
from rag.schemas import RetrieverConfig, VectorStoreConfig


@pytest.fixture
def config() -> VectorStoreConfig:
    return VectorStoreConfig(redis_url="redis://test", embedding_model_name="fake-model")


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def index(config: VectorStoreConfig, client: MagicMock) -> RedisIndexer:
    return RedisIndexer(config=config, client=client, embeddings=MagicMock())


@pytest.fixture
def retriever() -> DocumentRetriever:
    config = RetrieverConfig(
        redis_url="redis://test", embedding_model_name="fake-model", score_threshold=0.5
    )
    return DocumentRetriever(config=config, embeddings=MagicMock())
