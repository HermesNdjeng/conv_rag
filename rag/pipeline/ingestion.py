from minio import Minio

from rag.exceptions import EmbeddingModelMismatchError
from rag.indexer import RedisIndexer
from rag.loader import DocumentLoader
from rag.utils.logging_utils import setup_logger


logger = setup_logger("ingestion")


def ingest_bucket(
    minio_client: Minio,
    bucket: str,
    loader: DocumentLoader,
    indexer: RedisIndexer,
    *,
    index_name: str,
) -> dict[str, int]:
    """Synchronize a Redis index with the Markdown objects in a MinIO bucket.

    The bucket is the source of truth: every ``.md`` object is (re)indexed, and documents no
    longer present in the bucket are removed from the index. Reused by the bootstrap script
    (global corpus) and by the worker (user uploads).

    Robustness: idempotent (deterministic per-chunk keys), a single failing document is logged
    and skipped, an embedding-model mismatch is fatal, and an empty bucket does NOT wipe the
    index (treated as a likely source error rather than an intentional empty corpus).

    Args:
        minio_client: MinIO client used to list and download objects.
        bucket: Name of the bucket holding the Markdown (``.md``) objects to ingest.
        loader: DocumentLoader that splits each Markdown file into chunks.
        indexer: RedisIndexer that writes chunks and prunes orphaned documents.
        index_name: Target Redis index the bucket is synced into (e.g. 'global' or 'user_42').

    Returns:
        A report dict with counts: ``ingested`` (documents successfully indexed), ``failed``
        (documents skipped after an error), ``deleted`` (orphaned documents removed).
    """
    names = [
        obj.object_name
        for obj in minio_client.list_objects(bucket, recursive=True)
        if obj.object_name and obj.object_name.endswith(".md")
    ]
    logger.info(f"Found {len(names)} markdown object(s) in bucket '{bucket}'")

    ingested = 0
    failed = 0
    for name in names:
        try:
            response = minio_client.get_object(bucket, name)
            try:
                content = response.read()
            finally:
                # MinIO requires both close() and release_conn(); a plain `with` would skip
                # release_conn() and leak pooled connections under load.
                response.close()
                response.release_conn()
            chunks = loader.load_from_bytes(content, document_id=name)
            indexer.upsert(chunks, index_name)
            ingested += 1
            logger.info(f"Ingested '{name}' ({len(chunks)} chunks) → '{index_name}'")
        except EmbeddingModelMismatchError:
            raise  # fatal config error — don't swallow, don't keep retrying every document
        except Exception as exc:
            failed += 1
            logger.error(f"Failed to ingest '{name}': {exc}")

    deleted = 0
    if names:
        orphans = indexer.list_document_ids(index_name) - set(names)
        for document_id in orphans:
            indexer.delete_by_metadata(index_name, {"document_id": document_id})
        deleted = len(orphans)
        if deleted:
            logger.info(f"Deleted {deleted} orphaned document(s) from '{index_name}'")
    else:
        logger.warning(
            f"Bucket '{bucket}' is empty — skipping orphan cleanup to avoid wiping the index"
        )

    return {"ingested": ingested, "failed": failed, "deleted": deleted}
