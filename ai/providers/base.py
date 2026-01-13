"""
Base Provider class
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict


class Provider(ABC):
    """
    Base class for AI service providers.

    A Provider represents an API endpoint/service that hosts AI models.
    Different providers may have different API limits, authentication methods, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass

    @abstractmethod
    def get_text_embedder(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        **options: Any
    ):
        """
        Create a text embedder descriptor.

        Args:
            model: Model name (uses provider's default if not specified)
            dimensions: Custom dimensions (if model supports it)
            **options: Additional model-specific options

        Returns:
            TextEmbedderDescriptor instance
        """
        pass

    @abstractmethod
    def get_pdf_parser(
        self,
        parser_type: Optional[str] = None,
        document_type: str = "pdf",
        parser_mode: str = "pipeline",
        poll_interval: int = 2,
        poll_timeout: int = 300,
        custom_options: Optional[Dict[str, Any]] = None,
        **options: Any
    ):
        """
        Create a PDF parser descriptor.

        Args:
            parser_type: Parser backend (uses provider's default if not specified)
            document_type: Type of document (default: "pdf")
            parser_mode: Parsing mode (default: "pipeline")
            poll_interval: Seconds between status polls (default: 2)
            poll_timeout: Maximum seconds to wait for job completion (default: 300)
            custom_options: Additional parser options (default: {})
            **options: Additional options including retry configuration

        Returns:
            PDFParserDescriptor instance
        """
        pass
