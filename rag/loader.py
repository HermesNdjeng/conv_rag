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

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from conv_rag.utils.logging_utils import setup_logger


logger = setup_logger("rag.loader")

_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


class LoaderConfig(BaseModel):
    """Configuration for the document loader."""

    max_chunk_size: int = Field(
        default=2000,
        description="Maximum chunk size in characters; sections below this are kept as-is",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap used only when a section exceeds max_chunk_size and must be split",
    )
    strip_headers: bool = Field(
        default=False,
        description="Strip header lines from chunk content (headers are still kept in metadata)",
    )


class DocumentLoader:
    """Loads Markdown content and splits it into chunks for indexing."""

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
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
        """Split by Markdown headers; fall back to character split only for oversized sections."""
        chunks: list[Document] = []
        for chunk in self._md_splitter.split_text(text):
            if len(chunk.page_content) > self.config.max_chunk_size:
                chunks.extend(self._char_splitter.split_documents([chunk]))
            else:
                chunks.append(chunk)
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks

    def load_from_bytes(self, content: bytes, metadata: dict[str, Any]) -> list[Document]:
        """Primary entry point for the worker pipeline (content streamed from MinIO)."""
        text = content.decode("utf-8")
        chunks = self._split(text, metadata)
        logger.info(f"{len(chunks)} chunks — source: {metadata.get('source', 'unknown')}")
        return chunks

    def load_from_file(
        self, file_path: str, extra_metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        """Load and chunk a Markdown file from the local filesystem."""
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        metadata: dict[str, Any] = {"source": file_path}
        if extra_metadata:
            metadata.update(extra_metadata)
        chunks = self._split(text, metadata)
        logger.info(f"Loaded '{file_path}' → {len(chunks)} chunks")
        return chunks
