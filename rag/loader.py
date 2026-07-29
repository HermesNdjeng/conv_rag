"""
Document Loader

Loads Markdown content (as bytes, from MinIO or disk) and splits it into
LangChain Document chunks for downstream indexing.

Chunking strategy: two-pass Markdown-aware split.
  Pass 1 — MarkdownHeaderTextSplitter: respects header hierarchy (H1→H2→H3→H4)
            and injects header breadcrumbs into each chunk's metadata.
  Pass 2 — RecursiveCharacterTextSplitter: further splits oversized sections on line
            breaks only (never mid-line), so single-line HTML tables stay whole.

This module has no dependency on FastAPI, ARQ, or the worker layer.
"""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.constants import HEADERS_TO_SPLIT_ON
from rag.schemas import LoaderConfig
from rag.utils.logging_utils import setup_logger


logger = setup_logger("loader")

# OCR grounding artifacts. DeepSeek emits <|ref|>label<|/ref|><|det|>[[bbox]]<|/det|> blocks;
# we drop each block WHOLE (label included), plus stray coord lists, stray <|...|> tags, and the
# page-split marker our OCR step inserts between pages. A degenerate table has no real cell text.
_GROUNDING = re.compile(r"<\|ref\|>.*?<\|/ref\|>|<\|det\|>.*?<\|/det\|>")
_OCR_TAGS = re.compile(r"<\|[^|]*\|>")
_OCR_COORDS = re.compile(r"\[\[[\d\s,.\[\]]+\]\]")
_PAGE_SPLIT = re.compile(r"<--- Page Split --->")
_HTML_TAG = re.compile(r"<[^>]+>")
# Tables are single-line in our OCR output; match a closed block or, if unclosed, to line end.
_TABLE_BLOCK = re.compile(r"<table\b.*?</table>|<table\b[^\n]*")


def _clean_markdown(text: str) -> str:
    """Strip OCR grounding artifacts/markers and drop degenerate (empty) HTML tables.

    Args:
        text: Raw Markdown text (as produced by the OCR step).

    Returns:
        The cleaned text: whole ``<|ref|>…<|/ref|>`` / ``<|det|>…<|/det|>`` blocks removed
        (their inner label/coords included), stray tags/coords and the ``<--- Page Split --->``
        marker removed, and any HTML table whose cells contain no real text deleted (real
        tables are kept intact).
    """
    text = _GROUNDING.sub("", text)
    text = _OCR_TAGS.sub("", text)
    text = _OCR_COORDS.sub("", text)
    text = _PAGE_SPLIT.sub("", text)

    def _keep_or_drop(match: re.Match[str]) -> str:
        block = match.group(0)
        return "" if not _HTML_TAG.sub("", block).strip() else block

    return _TABLE_BLOCK.sub(_keep_or_drop, text)


# TODO(perf): if hierarchy-based context proves too weak for retrieval, switch to an
# LLM-generated context blurb per chunk (who/when/where) instead of this heading sentence.
def _context_sentence(metadata: dict[str, Any]) -> str:
    """Build a French sentence locating a chunk within the document's heading hierarchy.

    Args:
        metadata: Chunk metadata; may carry ``h1``..``h4`` header breadcrumbs.

    Returns:
        A sentence like "Cet extrait provient du document « … », section « … », sous-section
        « … ».", or an empty string if no headers are present.
    """
    doc = metadata.get("h1")
    subs = [metadata[k] for k in ("h2", "h3", "h4") if k in metadata]
    if not doc and not subs:
        return ""
    lead = f"Cet extrait provient du document « {doc} »" if doc else "Cet extrait provient"
    labels = ["section", "sous-section", "sous-partie"]
    tail = "".join(f", {labels[i]} « {sub} »" for i, sub in enumerate(subs))
    return f"{lead}{tail}."


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
            separators=["\n"],  # only split on line breaks — never mid-line (keeps tables whole)
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
        text = _clean_markdown(text)
        chunks: list[Document] = []
        for chunk in self._md_splitter.split_text(text):
            if len(chunk.page_content) > self.config.max_chunk_size:
                chunks.extend(self._char_splitter.split_documents([chunk]))
            else:
                chunks.append(chunk)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_index"] = i
            context = _context_sentence(chunk.metadata)
            if context:
                chunk.page_content = f"{context}\n\n{chunk.page_content}"
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
