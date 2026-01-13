"""
Example: Using AIWorksProvider to parse PDF documents.

This example demonstrates the recommended pattern for PDF parsing:
1. Create Provider (stores API configuration)
2. Get Descriptor from Provider
3. Instantiate Parser from Descriptor
4. Parse files
"""

from ai.providers import AIWorksProvider
from ai.protocols import PDFParseError, RetryStrategy


def main():
    """
    Main example using AIWorksProvider (Recommended Pattern).

    This follows the same architecture as text_embedder.
    """
    print("=" * 60)
    print("PDF Parsing via AIWorksProvider")
    print("=" * 60)

    # 1. Create Provider with API configuration
    provider = AIWorksProvider(
        name="AIWorks",
        base_url="http://172.16.99.68:8011",
        max_batch_tokens=100_000
    )

    # 2. Get PDF parser descriptor
    descriptor = provider.get_pdf_parser(
        parser_type="mineru",  # Uses DEFAULT_PDF_PARSER if not specified
        document_type="pdf",
        parser_mode="pipeline",
        poll_interval=2,
        poll_timeout=600,  # 10 minutes for large files
        custom_options={
            "language": "zh-CN",
            "enable_ocr": True,
        },
        # Retry configuration (same as text_embedder defaults)
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
        max_retries=5,
        initial_delay=2.0,
        max_delay=120.0,
    )

    print(f"\nConfiguration:")
    print(f"  Provider:    {descriptor.get_provider()}")
    print(f"  Base URL:    {descriptor.get_base_url()}")
    print(f"  Parser Type: {descriptor.parser_type}")
    print(f"  Retry:       {descriptor.retry_strategy.value}")
    print(f"  Max Retries: {descriptor.max_retries}")
    print("-" * 60)

    # 3. Instantiate parser
    parser = descriptor.instantiate()

    # 4. Parse files
    files = [
        "/uni-parse-documents/documents/test01.pdf",
        "/uni-parse-documents/documents/test02.pdf",
    ]

    try:
        print(f"\nParsing {len(files)} file(s)...")
        result = parser.parse_files(
            files=files,
            source_parent_path="/uni-parse-documents/documents/parsed"
        )

        # Display results
        print(f"\n{'='*60}")
        print(f"Parsing Summary:")
        print(f"  Total:   {result.total_count}")
        print(f"  Success: {result.success_count}")
        print(f"  Failed:  {result.failed_count}")
        print(f"{'='*60}\n")

        if result.successful:
            print("✓ Successfully parsed files:")
            for i, path in enumerate(result.successful, 1):
                print(f"  {i}. {path}")

        if result.failed:
            print(f"\n✗ Failed to parse {len(result.failed)} file(s):")
            for i, failed in enumerate(result.failed, 1):
                print(f"  {i}. {failed.source_path}")
                print(f"     Error: {failed.error_message}")

        # Return appropriate exit code
        return 1 if result.failed_count > 0 else 0

    except PDFParseError as e:
        print(f"\n✗ Error: {e}")
        return 1


def example_single_file():
    """Example: Parse a single file."""
    provider = AIWorksProvider(
        name="AIWorks",
        base_url="http://172.16.99.68:8011",
    )

    descriptor = provider.get_pdf_parser(
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
        max_retries=3,
    )

    parser = descriptor.instantiate()

    try:
        output_path = parser.parse_file(
            source_path="/uni-parse-documents/documents/report.pdf",
            output_path="/uni-parse-documents/documents/parsed/report.md"
        )
        print(f"✓ File parsed successfully: {output_path}")
    except PDFParseError as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    exit(main())

    # Uncomment to run single file example:
    # example_single_file()

