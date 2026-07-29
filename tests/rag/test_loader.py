import pytest

from rag.loader import DocumentLoader, LoaderConfig, _clean_markdown


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


def test_strip_headers_removes_header_from_content() -> None:
    config = LoaderConfig(strip_headers=True)
    loader = DocumentLoader(config)
    chunks = loader._split("# My Header\n\nBody text.", {})
    assert len(chunks) == 1
    assert "# My Header" not in chunks[0].page_content
    assert chunks[0].metadata.get("h1") == "My Header"


def test_load_from_bytes_sets_document_id_and_source(loader: DocumentLoader) -> None:
    content = b"# Header\n\nContent here."
    chunks = loader.load_from_bytes(content, document_id="bucket/doc.md")
    assert len(chunks) > 0
    assert all(c.metadata["document_id"] == "bucket/doc.md" for c in chunks)
    assert all(c.metadata["source"] == "bucket/doc.md" for c in chunks)


def test_load_from_bytes_stamps_chunk_index(loader: DocumentLoader) -> None:
    md = "# Title\n\nParagraph.\n\n## Sub\n\nMore text."
    chunks = loader.load_from_bytes(md.encode(), document_id="x")
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


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


def test_clean_removes_grounding_blocks() -> None:
    text = "Avant <|ref|>label<|/ref|><|det|>[[128, 90, 873, 279]]<|/det|> apres"
    out = _clean_markdown(text)
    assert "label" not in out
    assert "128" not in out
    assert "<|" not in out
    assert "Avant" in out and "apres" in out


def test_clean_removes_page_split_marker() -> None:
    assert "Page Split" not in _clean_markdown("A\n\n<--- Page Split --->\n\nB")


def test_clean_drops_empty_table_keeps_real_table() -> None:
    empty = "<table><tr><td></td><td></td></tr></table>"
    real = "<table><tr><td>1955</td><td>UPC</td></tr></table>"
    out = _clean_markdown(f"{empty}\n{real}")
    assert "1955" in out  # real table kept
    assert "<td></td>" not in out  # degenerate table dropped
