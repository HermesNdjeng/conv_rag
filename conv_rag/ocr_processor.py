"""
PDF to Markdown Processor

This application fetches PDF files from a remote MinIO bucket and converts them to Markdown format
using the DeepSeek OCR API deployed on a VM. Each PDF file is then saved as a Markdown file in a
different MinIO bucket.
"""

# TODO: Now we are waiting for the entire PDF to be processed before uploading
# the markdown file. We can optimize this by streaming the PDF content to the
# OCR API and uploading the markdown content as it is generated. This would
# reduce the overall processing time and allow for faster access to the
# converted files.

import io
import json
import os
import sys
import tempfile
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from utils.logging_utils import setup_logger


# Set up module-specific logger
logger = setup_logger("ocr_processor")

# Load environment variables
load_dotenv()
api_base_url = os.getenv("OCR_URL")
ocr_api_key = os.getenv("OCR_API_KEY")


class Colors:
    """ANSI color codes for terminal output"""

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


class PDFToMarkdownProcessor:
    """Processor for converting PDF files to Markdown using DeepSeek OCR API"""

    def __init__(
        self,
        api_base_url: str | None = api_base_url,
        api_key: str | None = ocr_api_key,
        minio_endpoint: str | None = os.getenv("MINIO_ENDPOINT"),
        access_key: str | None = os.getenv("MINIO_ACCESS_KEY"),
        secret_key: str | None = os.getenv("MINIO_SECRET_KEY"),
        source_bucket: str | None = os.getenv("MINIO_SOURCE_BUCKET"),
        output_bucket: str | None = os.getenv("MINIO_OUTPUT_BUCKET"),
    ):
        if not all([minio_endpoint, access_key, secret_key, source_bucket, output_bucket]):
            raise ValueError("Missing required MinIO configuration in environment variables.")

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
        """Verify that the source and output buckets are accessible."""
        try:
            for bucket in (self.source_bucket, self.output_bucket):
                if bucket and not self.minio_client.bucket_exists(bucket):
                    raise ConnectionError(f"MinIO bucket '{bucket}' does not exist.")
            logger.info("MinIO connection successful — both buckets are accessible.")
        except S3Error as e:
            raise ConnectionError(f"MinIO connection failed: {e}") from e

    def _test_api_connection(self) -> bool:
        """Test if the OCR API is accessible."""
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
            logger.error(f"OCR API returned status code: {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"OCR API connection failed: {e}")
            return False

    def _call_ocr_api(self, pdf_path: str) -> str | None:
        """
        Call the OCR API to process a PDF file.

        Args:
            pdf_path: Local path to the PDF file.

        Returns:
            Markdown content string, or None if processing failed.
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
                    pages = []
                    for page in result["results"]:
                        if isinstance(page, dict) and page.get("result"):
                            pages.append(page["result"])
                    return "\n\n<--- Page Split --->\n\n".join(pages)

                for field in ("markdown", "content", "text", "result", "output"):
                    if field in result:
                        return result[field]

                return json.dumps(result, indent=2)

            return str(result)

        except Exception as e:
            logger.error(f"Error calling OCR API for {pdf_path}: {e}")
            return None

    def convert_and_upload(self, object_name: str) -> bool:
        """
        Download a PDF from MinIO, convert it via OCR, and upload the Markdown result.

        Args:
            object_name: Object key of the PDF in the source bucket.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            logger.info(f"Downloading '{object_name}' from bucket '{self.source_bucket}'")
            if self.source_bucket:
                self.minio_client.fget_object(self.source_bucket, object_name, tmp_path)
            else:
                logger.error("Source bucket is not configured.")
                return False

            markdown_content = self._call_ocr_api(tmp_path)
            os.unlink(tmp_path)

            if not markdown_content:
                logger.error(f"OCR returned no content for '{object_name}'")
                return False

            stem = object_name.rsplit(".", 1)[0]
            md_object_name = f"{stem}.md"
            encoded = markdown_content.encode("utf-8")
            if not self.output_bucket:
                logger.error("Output bucket is not configured.")
                return False
            self.minio_client.put_object(
                self.output_bucket,
                md_object_name,
                io.BytesIO(encoded),
                length=len(encoded),
                content_type="text/markdown",
            )
            logger.info(
                f"The len of the markdown content for '{object_name}' is {len(encoded)} bytes"
            )
            logger.info(f"Uploaded '{md_object_name}' to bucket '{self.output_bucket}'")
            return True

        except S3Error as e:
            logger.error(f"MinIO error processing '{object_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error processing '{object_name}': {e}")
            return False

    def scan_and_process_all_pdfs(self) -> list[str]:
        """
        List all PDFs in the source bucket and convert each one to Markdown.

        Returns:
            List of Markdown object names that were successfully uploaded.
        """
        if not self.source_bucket:
            logger.error("Source bucket is not configured.")
            return []
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

        results = []
        for name in pdf_objects:
            logger.info(f"Processing PDF: {name}")
            start_time = time.time()
            if self.convert_and_upload(name):
                stem = name.rsplit(".", 1)[0]
                results.append(f"{stem}.md")
            end_time = time.time()
            logger.info(f"Processing time for {name}: {end_time - start_time:.2f} seconds")

        return results


def main() -> None:
    """Main function to run the PDF processor."""
    source_bucket = os.getenv("MINIO_SOURCE_BUCKET")
    print(f"{Colors.BLUE}PDF to Markdown Processor{Colors.RESET}")
    print(f"{Colors.YELLOW}Scanning MinIO bucket '{source_bucket}'...{Colors.RESET}")

    try:
        processor = PDFToMarkdownProcessor()
        markdown_files = processor.scan_and_process_all_pdfs()

        if markdown_files:
            print(
                f"\n{Colors.GREEN}Successfully converted "
                f"{len(markdown_files)} PDF(s) to Markdown:{Colors.RESET}"
            )
            for md_file in markdown_files:
                print(f"  - {md_file}")
        else:
            print(f"{Colors.YELLOW}No PDF files were processed.{Colors.RESET}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
