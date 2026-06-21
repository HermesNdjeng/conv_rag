import os

from pydantic import BaseModel, Field

from rag.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SCORE_THRESHOLD,
)


class LoaderConfig(BaseModel):
    max_chunk_size: int = Field(default=DEFAULT_MAX_CHUNK_SIZE)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP)
    strip_headers: bool = Field(default=False)


class VectorStoreConfig(BaseModel):
    redis_url: str = Field(default_factory=lambda: os.environ["REDIS_URL"])
    embedding_model_name: str = Field(default_factory=lambda: os.environ["EMBEDDING_MODEL"])


class RetrieverConfig(VectorStoreConfig):
    score_threshold: float = Field(default=DEFAULT_SCORE_THRESHOLD)


class IndexResult(BaseModel):
    index_name: str
    document_count: int
