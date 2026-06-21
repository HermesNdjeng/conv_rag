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

# Explicit index schema shared by the indexer (at index creation) and the retriever
# (when reconnecting via from_existing_index). Must mirror the metadata the loader
# stamps on each chunk. Keep in sync with HEADERS_TO_SPLIT_ON + loader metadata.
RAG_INDEX_SCHEMA: dict[str, list[dict[str, str]]] = {
    "tag": [
        {"name": "document_id"},
        {"name": "source"},
        {"name": "h1"},
        {"name": "h2"},
        {"name": "h3"},
        {"name": "h4"},
    ],
    "numeric": [
        {"name": "chunk_index"},
    ],
}
