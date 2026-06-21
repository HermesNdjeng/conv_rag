"""
PDF to Markdown Processor

Fetches PDF files from a MinIO bucket and converts them to Markdown
using the DeepSeek OCR API. Resulting Markdown is uploaded to an output bucket.

This module has no dependency on FastAPI or the worker layer — it is a pure
domain library that can be called from a worker task, a CLI script, or tests.
"""

# TODO: Now we are waiting for the entire PDF to be processed before uploading
# the markdown file. We can optimize this by streaming the PDF content to the
# OCR API and uploading the markdown content as it is generated. This would
# reduce the overall processing time and allow for faster access to the
# converted files.

import io
import json
import os
import tempfile
import time
from urllib.parse import urlparse

import requests
from minio import Minio
from minio.error import S3Error

from rag.utils.logging_utils import setup_logger


logger = setup_logger("ocr.processor")


class PDFToMarkdownProcessor:
    """Converts PDF files stored in MinIO to Markdown via the DeepSeek OCR API."""

    def __init__(
        self,
        api_base_url: str,
        api_key: str | None,
        minio_endpoint: str,
        access_key: str,
        secret_key: str,
        source_bucket: str,
        output_bucket: str,
    ) -> None:
        """
        Args:
            api_base_url: Base URL of the OCR API (e.g. "https://ocr.example.com").
            api_key: Optional API key sent as the ``x-api-key`` header.
            minio_endpoint: Full URL of the MinIO endpoint (scheme + host + optional port).
            access_key: MinIO access key.
            secret_key: MinIO secret key.
            source_bucket: Bucket name containing source PDF files.
            output_bucket: Bucket name where converted Markdown files will be uploaded.
        """
        self.api_base_url = api_base_url
        self.api_headers = {"x-api-key": api_key} if api_key else {}
        self.source_bucket = source_bucket
        self.output_bucket = output_bucket

        parsed = urlparse(minio_endpoint)
        self.minio_client = Minio(
            str(parsed.netloc),
            access_key=access_key,
            secret_key=secret_key,
            secure=parsed.scheme == "https",
        )

        self._test_minio_connection()

        if not self._test_api_connection():
            raise ConnectionError(f"Cannot connect to OCR API at {api_base_url}")

    def _test_minio_connection(self) -> None:
        """Verify that both source and output buckets are accessible."""
        try:
            for bucket in (self.source_bucket, self.output_bucket):
                if not self.minio_client.bucket_exists(bucket):
                    raise ConnectionError(f"MinIO bucket '{bucket}' does not exist.")
            logger.info("MinIO connection successful — both buckets are accessible.")
        except S3Error as exc:
            raise ConnectionError(f"MinIO connection failed: {exc}") from exc

    def _test_api_connection(self) -> bool:
        """Return True if the OCR API /docs endpoint is reachable."""
        try:
            response = requests.get(
                f"{self.api_base_url}/docs",
                headers=self.api_headers,
                timeout=5,
                verify=False,  # noqa: S501
            )
            if response.status_code == 200:
                logger.info("OCR API connection successful.")
                return True
            logger.error(f"OCR API returned status {response.status_code}.")
            return False
        except requests.exceptions.RequestException as exc:
            logger.error(f"OCR API connection failed: {exc}")
            return False

    def _call_ocr_api(self, pdf_path: str) -> str | None:
        """
        Send a local PDF file to the OCR API and return the Markdown content.

        Args:
            pdf_path: Absolute path to the PDF file on disk.

        Returns:
            Markdown string, or ``None`` if the API call failed.
        """
        try:
            url = f"{self.api_base_url}/ocr/pdf"
            logger.info(f"Sending PDF to OCR API: {url}")

            with open(pdf_path, "rb") as pdf_file:
                files = {"file": (os.path.basename(pdf_path), pdf_file, "application/pdf")}
                data = {"prompt": "<image>\n<|grounding|>Convert the document to markdown."}
                response = requests.post(
                    url,
                    headers=self.api_headers,
                    files=files,
                    data=data,
                    timeout=4800,
                    verify=False,  # noqa: S501
                )

            if response.status_code != 200:
                logger.error(f"OCR API error {response.status_code}: {response.text}")
                return None

            result = response.json()

            if isinstance(result, dict):
                if "results" in result and isinstance(result["results"], list):
                    pages = [
                        page["result"]
                        for page in result["results"]
                        if isinstance(page, dict) and page.get("result")
                    ]
                    return "\n\n<--- Page Split --->\n\n".join(pages)

                for field in ("markdown", "content", "text", "result", "output"):
                    if field in result:
                        return result[field]  # type: ignore[return-value]

                return json.dumps(result, indent=2)

            return str(result)

        except Exception as exc:
            logger.error(f"Error calling OCR API for {pdf_path}: {exc}")
            return None

    def convert_and_upload(self, object_name: str) -> bool:
        """
        Download a PDF from MinIO, convert it to Markdown, and upload the result.

        The output object name mirrors the source key with ``.pdf`` replaced by ``.md``.

        Args:
            object_name: Object key of the PDF in the source bucket.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            logger.info(f"Downloading '{object_name}' from bucket '{self.source_bucket}'")
            self.minio_client.fget_object(self.source_bucket, object_name, tmp_path)

            markdown_content = self._call_ocr_api(tmp_path)
            os.unlink(tmp_path)

            if not markdown_content:
                logger.error(f"OCR returned no content for '{object_name}'")
                return False

            stem = object_name.rsplit(".", 1)[0]
            md_object_name = f"{stem}.md"
            encoded = markdown_content.encode("utf-8")

            self.minio_client.put_object(
                self.output_bucket,
                md_object_name,
                io.BytesIO(encoded),
                length=len(encoded),
                content_type="text/markdown",
            )
            logger.info(
                f"Uploaded '{md_object_name}' ({len(encoded)} bytes) → '{self.output_bucket}'"
            )
            return True

        except S3Error as exc:
            logger.error(f"MinIO error processing '{object_name}': {exc}")
            return False
        except Exception as exc:
            logger.error(f"Unexpected error processing '{object_name}': {exc}")
            return False

    def scan_and_process_all_pdfs(self) -> list[str]:
        """
        List every PDF in the source bucket and convert each one to Markdown.

        Returns:
            List of Markdown object keys that were successfully uploaded.
        """
        objects = list(self.minio_client.list_objects(self.source_bucket, recursive=True))
        pdf_objects = [
            obj.object_name
            for obj in objects
            if obj.object_name and obj.object_name.lower().endswith(".pdf")
        ]

        if not pdf_objects:
            logger.info(f"No PDF files found in bucket '{self.source_bucket}'.")
            return []

        logger.info(f"Found {len(pdf_objects)} PDF(s) in '{self.source_bucket}'.")

        results: list[str] = []
        for name in pdf_objects:
            logger.info(f"Processing: {name}")
            start = time.time()
            if self.convert_and_upload(name):
                stem = name.rsplit(".", 1)[0]
                results.append(f"{stem}.md")
            logger.info(f"Done in {time.time() - start:.2f}s")

        return results
