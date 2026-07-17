import redis as redis_lib
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from redis.exceptions import RedisError
from redisvl.exceptions import RedisVLError

from rag.constants import INDEX_META_KEY
from rag.exceptions import EmbeddingModelMismatchError
from rag.schemas import RetrieverConfig
from rag.utils.logging_utils import setup_logger


logger = setup_logger("retriever")


class DocumentRetriever:
    def __init__(
        self,
        config: RetrieverConfig | None = None,
        *,
        embeddings: HuggingFaceEmbeddings | None = None,
        client: redis_lib.Redis | None = None,
    ) -> None:
        self.config = config or RetrieverConfig()
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name
        )
        self._client = client or redis_lib.from_url(self.config.redis_url)
        logger.info(f"DocumentRetriever ready — threshold={self.config.score_threshold}")

    def retrieve(self, query: str, indexes: list[str]) -> list[Document]:
        results: list[tuple[Document, float]] = []

        for index_name in indexes:
            stored_model = self._client.hget(INDEX_META_KEY.format(index_name), "embedding_model")
            if stored_model is not None:
                if isinstance(stored_model, bytes):
                    stored_model = stored_model.decode()
                if stored_model != self.config.embedding_model_name:
                    raise EmbeddingModelMismatchError(
                        f"Index '{index_name}' was built with embedding model '{stored_model}', "
                        f"but '{self.config.embedding_model_name}' is in use."
                    )
            try:
                store = RedisVectorStore.from_existing_index(
                    index_name=index_name,
                    embedding=self.embeddings,
                    redis_url=self.config.redis_url,
                )
                # langchain_redis returns cosine distance (0 = identical); convert to a
                # relevance score (higher = more similar) for the score_threshold below.
                results.extend(
                    (doc, 1.0 - distance)
                    for doc, distance in store.similarity_search_with_score(query, k=50)
                )
            except (RedisVLError, RedisError) as exc:
                logger.info(f"Index '{index_name}' unavailable — skipping: {exc}")

        filtered = sorted(
            [(doc, score) for doc, score in results if score >= self.config.score_threshold],
            key=lambda t: t[1],
            reverse=True,
        )

        logger.info(f"Retrieved {len(filtered)} docs from {indexes}")
        return [doc for doc, _ in filtered]
