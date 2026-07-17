class EmbeddingModelMismatchError(ValueError):
    """Raised when an index's embedding model differs from the one in use."""


class MissingChunkMetadataError(ValueError):
    """Raised when a document lacks the document_id/chunk_index needed to build a stable key."""


class EmptyDeleteFilterError(ValueError):
    """Raised when a delete filter is empty (would otherwise match the entire index)."""
