from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag.exceptions import EmbeddingModelMismatchError
from rag.pipeline import ingest_bucket


def test_failing_document_is_skipped_not_fatal(
    make_minio: Callable[..., MagicMock], loader: MagicMock
) -> None:
    minio = make_minio("ok.md", "bad.md")

    def load(content: bytes, document_id: str) -> list[SimpleNamespace]:
        if document_id == "bad.md":
            raise ValueError("corrupt markdown")
        return [SimpleNamespace()]

    loader.load_from_bytes.side_effect = load
    indexer = MagicMock()
    # Index matches what was actually ingested (only ok.md), so no orphan pruning.
    indexer.list_document_ids.return_value = {"ok.md"}

    report = ingest_bucket(minio, "bucket", loader, indexer, index_name="global")

    assert report["ingested"] == 1
    assert report["failed"] == 1


def test_model_mismatch_is_fatal(make_minio: Callable[..., MagicMock], loader: MagicMock) -> None:
    minio = make_minio("a.md")
    indexer = MagicMock()
    indexer.upsert.side_effect = EmbeddingModelMismatchError("mismatch")

    with pytest.raises(EmbeddingModelMismatchError):
        ingest_bucket(minio, "bucket", loader, indexer, index_name="global")


def test_empty_bucket_skips_cleanup(
    make_minio: Callable[..., MagicMock], loader: MagicMock
) -> None:
    minio = make_minio()  # empty — must NOT wipe the index
    indexer = MagicMock()

    report = ingest_bucket(minio, "bucket", loader, indexer, index_name="global")

    assert report == {"ingested": 0, "failed": 0, "deleted": 0}
    indexer.list_document_ids.assert_not_called()
    indexer.delete_by_metadata.assert_not_called()
