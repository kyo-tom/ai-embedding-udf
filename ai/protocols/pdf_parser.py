"""
PDF Parser protocol for parsing PDF documents via MinIO-based parsing service.

This module provides a protocol for batch parsing PDF files stored in MinIO,
with support for async job polling, retry strategies, and error handling.
"""

import os
import time
import random
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import requests

from ..utils import calculate_delay, should_retry, RetryStrategy, ErrorHandlingStrategy, sanitize_for_logging


logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Exception raised when PDF parsing fails."""
    pass


@dataclass
class FileParseResult:
    """Result of parsing a single file"""
    source_path: str
    output_path: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    job_id: Optional[str] = None


@dataclass
class BatchParseResult:
    """Result of batch parsing operation"""
    successful: List[str]  # List of successfully parsed output paths
    failed: List[FileParseResult]  # List of failed file parse results

    @property
    def success_count(self) -> int:
        """Number of successful parses"""
        return len(self.successful)

    @property
    def failed_count(self) -> int:
        """Number of failed parses"""
        return len(self.failed)

    @property
    def total_count(self) -> int:
        """Total number of files"""
        return self.success_count + self.failed_count


@dataclass
class PDFParserDescriptor:
    """
    PDF Parser Descriptor (Serializable Configuration)

    This stores all configuration needed to create a PDFParser instance.
    It's designed to be serializable for distributed computing environments.

    Design Pattern (following TextEmbedderDescriptor):
    - Descriptor stores configuration (serializable)
    - instantiate() creates the actual parser (non-serializable, has HTTP clients)
    """
    provider_name: str
    provider_options: Dict[str, Any]  # Contains base_url, api_key, etc.
    document_type: str = "pdf"
    parser_type: str = "mineru"
    parser_mode: str = "pipeline"
    poll_interval: int = 2
    poll_timeout: int = 300
    custom_options: Optional[Dict[str, Any]] = None

    # Retry and error handling configuration
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST

    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        # Convert string strategies to enums if needed
        if isinstance(self.retry_strategy, str):
            try:
                object.__setattr__(self, 'retry_strategy', RetryStrategy(self.retry_strategy))
            except ValueError:
                raise ValueError(
                    f"Invalid retry strategy: '{self.retry_strategy}'. "
                    f"Valid options: {[s.value for s in RetryStrategy]}"
                )

        if isinstance(self.error_handling, str):
            try:
                object.__setattr__(self, 'error_handling', ErrorHandlingStrategy(self.error_handling))
            except ValueError:
                raise ValueError(
                    f"Invalid error handling strategy: '{self.error_handling}'. "
                    f"Valid options: {[s.value for s in ErrorHandlingStrategy]}"
                )

        # Validate numeric parameters
        if self.poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive, got {self.poll_interval}")

        if self.poll_timeout <= 0:
            raise ValueError(f"poll_timeout must be positive, got {self.poll_timeout}")

        if self.poll_timeout < self.poll_interval:
            raise ValueError(
                f"poll_timeout ({self.poll_timeout}) must be >= poll_interval ({self.poll_interval})"
            )

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be non-negative, got {self.initial_delay}")

        if self.max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {self.max_delay}")

        if self.max_delay < self.initial_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= initial_delay ({self.initial_delay})"
            )

        if self.exponential_base <= 0:
            raise ValueError(f"exponential_base must be positive, got {self.exponential_base}")

        # Ensure custom_options is a dict
        if self.custom_options is None:
            object.__setattr__(self, 'custom_options', {})

    def get_provider(self) -> str:
        """Get provider name"""
        return self.provider_name

    def get_base_url(self) -> str:
        """Get base URL from provider options"""
        return self.provider_options.get("base_url", "http://localhost:8000")

    def instantiate(self) -> "PDFParser":
        """Create actual PDFParser instance (non-serializable)"""
        logger.info(
            f"Instantiating PDFParser: {self.parser_type} via {self.provider_name}"
        )
        return PDFParser(
            base_url=self.get_base_url(),
            document_type=self.document_type,
            parser_type=self.parser_type,
            parser_mode=self.parser_mode,
            poll_interval=self.poll_interval,
            poll_timeout=self.poll_timeout,
            custom_options=self.custom_options,
            retry_strategy=self.retry_strategy,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
            error_handling=self.error_handling,
        )


class PDFParser:
    """
    PDF Parser client for interacting with the document parsing service.

    Args:
        base_url: Base URL of the parsing service (e.g., "http://localhost:8000")
        document_type: Type of document to parse (default: "pdf")
        parser_type: Parser backend to use (default: "mineru")
        parser_mode: Parsing mode (default: "pipeline")
        poll_interval: Interval in seconds between status polls (default: 2)
        poll_timeout: Maximum time in seconds to poll for results (default: 300)
        custom_options: Custom parsing options (default: {})
        retry_strategy: Retry strategy for API calls (default: EXPONENTIAL_BACKOFF_LIMITED)
        max_retries: Maximum number of retries (default: 3)
        initial_delay: Initial delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
        error_handling: Error handling strategy (default: FAIL_FAST)
    """

    def __init__(
        self,
        base_url: str,
        document_type: str = "pdf",
        parser_type: str = "mineru",
        parser_mode: str = "pipeline",
        poll_interval: int = 2,
        poll_timeout: int = 300,
        custom_options: Optional[Dict[str, Any]] = None,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST,
    ):
        self.base_url = base_url.rstrip("/")
        self.document_type = document_type
        self.parser_type = parser_type
        self.parser_mode = parser_mode
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.custom_options = custom_options or {}

        # Retry configuration
        self.retry_strategy = retry_strategy
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

        # Error handling configuration
        self.error_handling = error_handling

        logger.info(
            f"PDFParser initialized: base_url={base_url}, "
            f"retry_strategy={retry_strategy.value}, "
            f"max_retries={max_retries}, "
            f"error_handling={error_handling.value}, "
            f"custom_options={sanitize_for_logging(custom_options or {})}"
        )

    def _submit_parse_job(self, source_path: str, main_output_path: str) -> Dict[str, Any]:
        """
        Submit a PDF parsing job to the service with retry logic.

        Args:
            source_path: MinIO path to the source PDF (without s3:// prefix)
            main_output_path: Output path for the parsed markdown file

        Returns:
            API response dictionary

        Raises:
            PDFParseError: If the API request fails after all retries
        """
        url = f"{self.base_url}/api/v1/parse_from_oss"

        payload = {
            "source_path": source_path,
            "document_type": self.document_type,
            "parser_type": self.parser_type,
            "parser_mode": self.parser_mode,
            "custom_options": self.custom_options,
            "main_output_path": main_output_path,
        }

        attempt = 0
        last_exception = None

        while True:
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=35,  # Slightly more than 30s server timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                last_exception = e
                logger.error(f"Submit parse job failed (attempt {attempt + 1}): {e}")

                # Check if we should retry
                if should_retry(attempt, e, self.retry_strategy, self.max_retries):
                    delay = calculate_delay(
                        attempt,
                        self.initial_delay,
                        self.max_delay,
                        self.exponential_base,
                        self.jitter
                    )
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                else:
                    # No more retries
                    break

        # All retries exhausted
        raise PDFParseError(f"Failed to submit parse job for {source_path} after {attempt + 1} attempts: {last_exception}")

    def _poll_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Poll the status of a parsing job with retry logic.

        Args:
            job_id: Job ID to query

        Returns:
            Job status dictionary

        Raises:
            PDFParseError: If the API request fails after all retries
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}"

        attempt = 0
        last_exception = None

        while True:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                last_exception = e
                logger.error(f"Poll job status failed (attempt {attempt + 1}): {e}")

                # Check if we should retry
                if should_retry(attempt, e, self.retry_strategy, self.max_retries):
                    delay = calculate_delay(
                        attempt,
                        self.initial_delay,
                        self.max_delay,
                        self.exponential_base,
                        self.jitter
                    )
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                else:
                    # No more retries
                    break

        # All retries exhausted
        raise PDFParseError(f"Failed to poll job status for {job_id} after {attempt + 1} attempts: {last_exception}")

    def _wait_for_completion(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for a parsing job to complete by polling its status.

        Args:
            job_id: Job ID to wait for

        Returns:
            Final job status dictionary

        Raises:
            PDFParseError: If job fails or timeout is reached
        """
        start_time = time.time()

        while True:
            # Check timeout
            if time.time() - start_time > self.poll_timeout:
                raise PDFParseError(
                    f"Job {job_id} did not complete within {self.poll_timeout}s timeout"
                )

            # Poll status
            status = self._poll_job_status(job_id)

            # Check if completed
            if status.get("status") == "completed":
                if status.get("error_message"):
                    raise PDFParseError(
                        f"Job {job_id} completed with error: {status['error_message']}"
                    )
                return status

            # Check if failed
            if status.get("status") == "failed":
                error_msg = status.get("error_message", "Unknown error")
                raise PDFParseError(f"Job {job_id} failed: {error_msg}")

            # Wait before next poll
            time.sleep(self.poll_interval)

    def parse_file(self, source_path: str, output_path: str) -> str:
        """
        Parse a single PDF file.

        Args:
            source_path: MinIO path to the source PDF (without s3:// prefix)
            output_path: Full output path for the parsed markdown file

        Returns:
            Path to the parsed markdown file

        Raises:
            PDFParseError: If parsing fails
        """
        # Submit job
        result = self._submit_parse_job(source_path, output_path)

        # If completed immediately, return result
        if result.get("completed"):
            if result.get("error_message"):
                raise PDFParseError(
                    f"Parse job for {source_path} failed: {result['error_message']}"
                )
            return result["main_output_path"]

        # Otherwise, poll until completion
        job_id = result.get("job_id")
        if not job_id:
            raise PDFParseError(f"No job_id returned for {source_path}")

        final_status = self._wait_for_completion(job_id)
        return final_status["main_output_path"]

    def parse_files(
        self,
        files: List[str],
        source_parent_path: str,
    ) -> BatchParseResult:
        """
        Parse multiple PDF files with error handling support.

        Args:
            files: List of file paths (MinIO paths without s3:// prefix)
            source_parent_path: Parent directory path for output markdown files

        Returns:
            BatchParseResult with successful and failed file results

        Raises:
            PDFParseError: If error_handling is FAIL_FAST and any file fails
        """
        successful = []
        failed = []

        for file_path in files:
            # Extract filename and change extension to .md
            filename = Path(file_path).stem + ".md"
            output_path = os.path.join(source_parent_path, filename)

            try:
                # Parse file
                result_path = self.parse_file(file_path, output_path)
                successful.append(result_path)
                logger.info(f"Successfully parsed: {file_path} -> {result_path}")

            except PDFParseError as e:
                # Handle error based on strategy
                if self.error_handling == ErrorHandlingStrategy.FAIL_FAST:
                    logger.error(f"Parse failed for {file_path}, failing fast: {e}")
                    raise
                else:
                    # Record failed file and continue (graceful degradation)
                    failed.append(FileParseResult(
                        source_path=file_path,
                        output_path=output_path,
                        success=False,
                        error_message=str(e),
                    ))
                    logger.warning(f"Failed to parse {file_path}, continuing with remaining files: {e}")

        return BatchParseResult(successful=successful, failed=failed)


def parse_pdfs(
    files: List[str],
    source_parent_path: str,
    base_url: str,
    document_type: str = "pdf",
    parser_type: str = "mineru",
    parser_mode: str = "pipeline",
    poll_interval: int = 2,
    poll_timeout: int = 300,
    custom_options: Optional[Dict[str, Any]] = None,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST,
) -> BatchParseResult:
    """
    UDF function to parse multiple PDF files via the parsing service with retry support.

    Args:
        files: List of file paths in MinIO (without s3:// prefix)
               Example: ["/uni-parse-documents/documents/file1.pdf", "/uni-parse-documents/documents/file2.pdf"]
        source_parent_path: Parent directory for output markdown files
                           Example: "/uni-parse-documents/documents/output"
        base_url: Base URL of the parsing service
                 Example: "http://localhost:8000"
        document_type: Type of document (default: "pdf")
        parser_type: Parser backend (default: "mineru")
        parser_mode: Parsing mode (default: "pipeline")
        poll_interval: Seconds between status polls (default: 2)
        poll_timeout: Maximum seconds to wait for job completion (default: 300)
        custom_options: Additional parser options (default: {})
        retry_strategy: Retry strategy for API calls (default: EXPONENTIAL_BACKOFF_LIMITED)
        max_retries: Maximum number of retries (default: 3)
        initial_delay: Initial delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
        error_handling: Error handling strategy (default: FAIL_FAST)

    Returns:
        BatchParseResult with successful paths and failed file results

    Example:
        >>> files = [
        ...     "/uni-parse-documents/documents/report1.pdf",
        ...     "/uni-parse-documents/documents/report2.pdf",
        ... ]
        >>> result = parse_pdfs(
        ...     files=files,
        ...     source_parent_path="/uni-parse-documents/output",
        ...     base_url="http://localhost:8000",
        ...     retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
        ...     max_retries=3,
        ...     error_handling=ErrorHandlingStrategy.FAIL_FAST
        ... )
        >>> print(f"Success: {result.success_count}, Failed: {result.failed_count}")
        >>> print(result.successful)
        ['/uni-parse-documents/output/report1.md', '/uni-parse-documents/output/report2.md']
    """
    parser = PDFParser(
        base_url=base_url,
        document_type=document_type,
        parser_type=parser_type,
        parser_mode=parser_mode,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
        custom_options=custom_options,
        retry_strategy=retry_strategy,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        error_handling=error_handling,
    )

    return parser.parse_files(files, source_parent_path)
