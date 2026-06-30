from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Redis as RedisVectorStore

from rag.constants import RAG_INDEX_SCHEMA
from rag.schemas import RetrieverConfig
from rag.utils.logging_utils import setup_logger


logger = setup_logger("retriever")


class DocumentRetriever:
    def __init__(
        self,
        config: RetrieverConfig | None = None,
        *,
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> None:
        self.config = config or RetrieverConfig()
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name
        )
        logger.info(f"DocumentRetriever ready — threshold={self.config.score_threshold}")

    def retrieve(self, query: str, indexes: list[str]) -> list[Document]:
        results: list[tuple[Document, float]] = []

        for index_name in indexes:
            try:
                store = RedisVectorStore.from_existing_index(
                    embedding=self.embeddings,
                    redis_url=self.config.redis_url,
                    index_name=index_name,
                    schema=RAG_INDEX_SCHEMA,
                )
                results.extend(store.similarity_search_with_relevance_scores(query, k=50))
            except Exception as exc:
                logger.info(f"Index '{index_name}' unavailable — skipping: {exc}")

        filtered = sorted(
            [(doc, score) for doc, score in results if score >= self.config.score_threshold],
            key=lambda t: t[1],
            reverse=True,
        )

        logger.info(f"Retrieved {len(filtered)} docs from {indexes}")
        return [doc for doc, _ in filtered]
