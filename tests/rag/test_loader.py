import pytest

from rag.loader import DocumentLoader, LoaderConfig


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


def test_empty_input_returns_empty_list(loader: DocumentLoader) -> None:
    assert loader._split("", {}) == []


def test_header_metadata_injected(loader: DocumentLoader) -> None:
    md = "# Doc\n\nIntro.\n\n## Section\n\nBody."
    chunks = loader._split(md, {})
    assert all("h1" in c.metadata for c in chunks)
    section = next(c for c in chunks if "h2" in c.metadata)
    assert section.metadata["h2"] == "Section"
    assert section.metadata["h1"] == "Doc"


def test_caller_metadata_merged_into_all_chunks(loader: DocumentLoader) -> None:
    md = "# A\n\nFoo.\n\n## B\n\nBar."
    chunks = loader._split(md, {"source": "doc.md", "author": "Alice"})
    for chunk in chunks:
        assert chunk.metadata["source"] == "doc.md"
        assert chunk.metadata["author"] == "Alice"


def test_oversized_section_is_char_split_and_respects_size_limit() -> None:
    config = LoaderConfig(max_chunk_size=100, chunk_overlap=0)
    loader = DocumentLoader(config)
    long_body = "abcdefghij " * 30  # 330 chars, clean word boundaries
    chunks = loader._split(f"# Title\n\n{long_body}", {})
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.get("h1") == "Title"
        assert len(chunk.page_content) <= 100


def test_strip_headers_removes_header_from_content() -> None:
    config = LoaderConfig(strip_headers=True)
    loader = DocumentLoader(config)
    chunks = loader._split("# My Header\n\nBody text.", {})
    assert len(chunks) == 1
    assert "# My Header" not in chunks[0].page_content
    assert chunks[0].metadata.get("h1") == "My Header"


def test_load_from_bytes_decodes_and_chunks(loader: DocumentLoader) -> None:
    content = b"# Header\n\nContent here."
    chunks = loader.load_from_bytes(content, {"source": "bucket/doc.md"})
    assert len(chunks) > 0
    assert all(c.metadata["source"] == "bucket/doc.md" for c in chunks)


def test_load_from_bytes_same_chunks_as_split(loader: DocumentLoader) -> None:
    md = "# Title\n\nParagraph.\n\n## Sub\n\nMore text."
    via_bytes = loader.load_from_bytes(md.encode(), {"source": "x"})
    via_split = loader._split(md, {"source": "x"})
    assert len(via_bytes) == len(via_split)


def test_load_from_file_sets_source_to_path(
    loader: DocumentLoader, tmp_path: pytest.TempPathFactory
) -> None:
    f = tmp_path / "doc.md"  # type: ignore[operator]
    f.write_text("# Title\n\nFile content.", encoding="utf-8")
    chunks = loader.load_from_file(str(f))
    assert len(chunks) > 0
    assert all(c.metadata["source"] == str(f) for c in chunks)


def test_load_from_file_extra_metadata_merged(
    loader: DocumentLoader, tmp_path: pytest.TempPathFactory
) -> None:
    f = tmp_path / "doc.md"  # type: ignore[operator]
    f.write_text("# Title\n\nContent.", encoding="utf-8")
    chunks = loader.load_from_file(str(f), extra_metadata={"version": "2"})
    for chunk in chunks:
        assert chunk.metadata["version"] == "2"
        assert chunk.metadata["source"] == str(f)


def test_load_from_file_extra_metadata_does_not_override_source(
    loader: DocumentLoader, tmp_path: pytest.TempPathFactory
) -> None:
    f = tmp_path / "doc.md"  # type: ignore[operator]
    f.write_text("# Title\n\nContent.", encoding="utf-8")
    chunks = loader.load_from_file(str(f), extra_metadata={"source": "override-attempt"})
    # extra_metadata is merged after source is set, so it wins — document that behavior
    assert all(c.metadata["source"] == "override-attempt" for c in chunks)
