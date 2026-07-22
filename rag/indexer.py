from typing import cast

import redis as redis_lib
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from redis.commands.search.query import Query
from redis.commands.search.result import Result
from redis.exceptions import ResponseError
from redisvl.query.filter import FilterExpression, Tag

from rag.constants import INDEX_META_KEY, RAG_INDEX_SCHEMA
from rag.exceptions import (
    EmbeddingModelMismatchError,
    EmptyDeleteFilterError,
    MissingChunkMetadataError,
)
from rag.schemas import IndexResult, VectorStoreConfig
from rag.utils.logging_utils import setup_logger


logger = setup_logger("indexer")

_DELETE_PAGE_SIZE = 500


class RedisIndexer:
    """Indexes and manages documents in Redis vector store indexes."""

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        *,
        client: redis_lib.Redis | None = None,
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> None:
        self.config = config or VectorStoreConfig()
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name
        )
        self._client = client or redis_lib.from_url(self.config.redis_url)
        logger.info(f"RedisIndexer connected — model={self.config.embedding_model_name}")

    @staticmethod
    def user_index(user_id: str) -> str:
        """Return the Redis index name for a given user."""
        return f"user_{user_id}"

    def upsert(self, documents: list[Document], index_name: str) -> IndexResult:
        """Embed and store documents into the given Redis index.

        Creates the index if it does not exist. Each chunk is written under a
        deterministic key ``{document_id}:{chunk_index}``, so re-indexing the same
        document overwrites its chunks in place instead of creating duplicates.

        Args:
            documents: LangChain Document objects. Each must carry ``document_id`` and
                ``chunk_index`` in its metadata (the loader stamps both).
            index_name: Target Redis index (e.g. 'global' or 'user_42').

        Returns:
            IndexResult with the index name and number of documents stored.

        Raises:
            MissingChunkMetadataError: If any document lacks ``document_id`` or ``chunk_index``.
        """
        if not documents:
            logger.warning(f"upsert called with empty list for '{index_name}'")
            return IndexResult(index_name=index_name, document_count=0)

        keys: list[str] = []
        for doc in documents:
            document_id = doc.metadata.get("document_id")
            chunk_index = doc.metadata.get("chunk_index")
            if document_id is None or chunk_index is None:
                raise MissingChunkMetadataError(
                    "Each document needs 'document_id' and 'chunk_index' in metadata to "
                    f"build a stable key; got metadata keys {list(doc.metadata)}"
                )
            keys.append(f"{document_id}:{chunk_index}")

        stored_model = self._client.hget(INDEX_META_KEY.format(index_name), "embedding_model")
        if stored_model is not None:
            stored_model = (
                stored_model.decode() if isinstance(stored_model, bytes) else stored_model
            )
            if stored_model != self.config.embedding_model_name:
                raise EmbeddingModelMismatchError(
                    f"Index '{index_name}' was built with embedding model '{stored_model}', "
                    f"refusing to re-index with '{self.config.embedding_model_name}'."
                )

        config = RedisConfig(
            index_name=index_name,
            redis_url=self.config.redis_url,
            metadata_schema=RAG_INDEX_SCHEMA,
        )
        RedisVectorStore.from_documents(
            documents,
            self.embeddings,
            config=config,
            keys=keys,
        )
        self._client.hset(
            INDEX_META_KEY.format(index_name), "embedding_model", self.config.embedding_model_name
        )
        logger.info(f"Upserted {len(documents)} docs into '{index_name}'")
        return IndexResult(index_name=index_name, document_count=len(documents))

    def delete_by_metadata(self, index_name: str, where: dict[str, str]) -> None:
        """Delete all documents in index_name whose metadata matches every key-value in where.

        Runs a filtered RediSearch query against the index's secondary metadata fields
        (each string metadata key is indexed as a TAG), then deletes the matching keys.
        This uses the index rather than scanning every key in the keyspace.

        Args:
            index_name: Redis index to target (e.g. 'user_42').
            where: Metadata filter, e.g. {'document_id': 'user_42/report.md'}.

        Raises:
            EmptyDeleteFilterError: If *where* is empty (would otherwise match every document).
        """
        if not where:
            raise EmptyDeleteFilterError(
                "`where` cannot be empty — refusing to delete the entire index"
            )

        # Combine the filters: @k1:{v1} @k2:{v2} ... (Tag handles value escaping).
        filter_expr: FilterExpression | None = None
        for k, v in where.items():
            expr = Tag(k) == v
            filter_expr = expr if filter_expr is None else filter_expr & expr

        ids: list[str] = []
        offset = 0
        while True:
            query = Query(str(filter_expr)).no_content().paging(offset, _DELETE_PAGE_SIZE)
            try:
                results = cast(Result, self._client.ft(index_name).search(query))
            except ResponseError as exc:
                logger.warning(f"Search on index '{index_name}' failed: {exc}")
                return
            batch = [doc.id for doc in results.docs]
            ids.extend(batch)
            if len(batch) < _DELETE_PAGE_SIZE:
                break
            offset += _DELETE_PAGE_SIZE

        if ids:
            self._client.delete(*ids)
            logger.info(f"Deleted {len(ids)} docs from '{index_name}' matching {where}")
        else:
            logger.info(f"No docs found in '{index_name}' matching {where}")

    def list_document_ids(self, index_name: str) -> set[str]:
        """Return the distinct ``document_id`` values currently indexed in index_name.

        Returns an empty set if the index does not exist.
        """
        ids: set[str] = set()
        offset = 0
        while True:
            query = Query("*").return_fields("document_id").paging(offset, _DELETE_PAGE_SIZE)
            try:
                results = cast(Result, self._client.ft(index_name).search(query))
            except ResponseError:
                return ids
            for doc in results.docs:
                document_id = getattr(doc, "document_id", None)
                if document_id is not None:
                    ids.add(document_id.decode() if isinstance(document_id, bytes) else document_id)
            if len(results.docs) < _DELETE_PAGE_SIZE:
                break
            offset += _DELETE_PAGE_SIZE
        return ids
