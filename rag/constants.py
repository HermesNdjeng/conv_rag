HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

DEFAULT_MAX_CHUNK_SIZE: int = 2000
DEFAULT_CHUNK_OVERLAP: int = 200

DEFAULT_TOP_K: int = 5
DEFAULT_SCORE_THRESHOLD: float = 0.3

GLOBAL_INDEX: str = "global"

# Redis hash holding per-index metadata; the embedding model it was built with is stored here
# so retrieval can refuse to query an index with a mismatched model (silently wrong otherwise).
INDEX_META_KEY: str = "rag:meta:{}"

# Metadata fields indexed by the indexer (langchain_redis `metadata_schema`), mirroring
# the metadata the loader stamps on each chunk. Fields not listed here cannot be
# filtered on. Keep in sync with HEADERS_TO_SPLIT_ON + loader metadata.
RAG_INDEX_SCHEMA: list[dict[str, str]] = [
    {"name": "document_id", "type": "tag"},
    {"name": "source", "type": "tag"},
    {"name": "h1", "type": "tag"},
    {"name": "h2", "type": "tag"},
    {"name": "h3", "type": "tag"},
    {"name": "h4", "type": "tag"},
    {"name": "chunk_index", "type": "numeric"},
]
