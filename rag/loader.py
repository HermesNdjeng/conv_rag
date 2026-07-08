"""
Document Loader

Loads Markdown content (as bytes, from MinIO or disk) and splits it into
LangChain Document chunks for downstream indexing.

Chunking strategy: two-pass Markdown-aware split.
  Pass 1 — MarkdownHeaderTextSplitter: respects header hierarchy (H1→H2→H3→H4)
            and injects header breadcrumbs into each chunk's metadata.
  Pass 2 — RecursiveCharacterTextSplitter: further splits sections that still
            exceed max_chunk_size using standard text separators.

This module has no dependency on FastAPI, ARQ, or the worker layer.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.constants import HEADERS_TO_SPLIT_ON
from rag.schemas import LoaderConfig
from rag.utils.logging_utils import setup_logger


logger = setup_logger("loader")


class DocumentLoader:
    """Loads Markdown content and splits it into chunks for indexing."""

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=self.config.strip_headers,
        )
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        logger.info(
            f"DocumentLoader ready — max_chunk_size={self.config.max_chunk_size}, "
            f"overlap={self.config.chunk_overlap}, strip_headers={self.config.strip_headers}"
        )

    def _split(self, text: str, metadata: dict[str, Any]) -> list[Document]:
        """Split by Markdown headers; fall back to character split only for oversized sections.

        Each chunk is stamped with a 0-based ``chunk_index`` so the indexer can build a
        stable ``{document_id}:{chunk_index}`` key, making re-indexing idempotent.
        """
        chunks: list[Document] = []
        for chunk in self._md_splitter.split_text(text):
            if len(chunk.page_content) > self.config.max_chunk_size:
                chunks.extend(self._char_splitter.split_documents([chunk]))
            else:
                chunks.append(chunk)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_index"] = i
        return chunks

    def load_from_bytes(
        self, content: bytes, document_id: str, extra_metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        """Primary entry point for the worker pipeline (content streamed from MinIO).

        Args:
            content: Raw Markdown bytes.
            document_id: Stable identifier for the source document (the MinIO object
                path). The indexer uses it to build idempotent per-chunk keys.
            extra_metadata: Optional extra metadata merged into every chunk.
        """
        text = content.decode("utf-8")
        metadata: dict[str, Any] = {"document_id": document_id, "source": document_id}
        if extra_metadata:
            metadata.update(extra_metadata)
        chunks = self._split(text, metadata)
        logger.info(f"{len(chunks)} chunks — document_id: {document_id}")
        return chunks

    def load_from_file(
        self, file_path: str, extra_metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        """Load and chunk a Markdown file from the local filesystem.

        The file path doubles as ``document_id`` so local re-indexing is idempotent too.
        """
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        metadata: dict[str, Any] = {"document_id": file_path, "source": file_path}
        if extra_metadata:
            metadata.update(extra_metadata)
        chunks = self._split(text, metadata)
        logger.info(f"Loaded '{file_path}' → {len(chunks)} chunks")
        return chunks
