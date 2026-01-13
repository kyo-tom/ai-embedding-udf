"""
OpenAI Provider implementation
"""
import sys
from typing import Optional, Any, Dict

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

from .base import Provider
from ..typing import OpenAIProviderOptions


class AIWorksProvider(Provider):
    """
    OpenAI (or OpenAI-compatible) API provider.

    This class stores API-LEVEL configurations that are specific to the
    API endpoint, not the model itself.

    Key Design:
    - max_batch_tokens: This is an API LIMIT, not a model limit
      * OpenAI official: 300,000 tokens per request
      * Azure OpenAI: May have different limits
      * Custom endpoints: Could be completely different

    Example:
        # Official OpenAI with default limits
        provider = OpenAIProvider(
            api_key="sk-...",
            max_batch_tokens=300_000  # OpenAI's limit
        )

        # Custom endpoint with different limits
        custom = OpenAIProvider(
            name="CustomAPI",
            base_url="http://localhost:9997/v1",
            api_key="empty",
            max_batch_tokens=100_000  # Custom server's limit
        )
    """

    _name: str
    _options: OpenAIProviderOptions
    _max_batch_tokens: int

    DEFAULT_TEXT_EMBEDDER = "conan-embedding-v1"
    SUPPORT_EMBEDDER = ["conan-embedding-v1"]

    DEFAULT_PDF_PARSER = "mineru"
    SUPPORT_PDF_PARSER = ["mineru"]

    def __init__(
        self,
        name: Optional[str] = None,
        max_batch_tokens: int = 10_000,  # OpenAI official API limit
        **options: Unpack[OpenAIProviderOptions]
    ):
        """
        Initialize OpenAI Provider.

        Args:
            name: Provider name (default: "openai")
            max_batch_tokens: Maximum tokens per batch request (API limit)
            **options: OpenAI client options (api_key, base_url, etc.)
        """
        self._name = name if name else "aiworks"
        self._options = options
        self._max_batch_tokens = max_batch_tokens

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_batch_tokens(self) -> int:
        """Maximum tokens allowed in a single batch request (API limit)"""
        return self._max_batch_tokens

    def get_text_embedder(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        retry_strategy: Optional[Any] = None,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        exponential_base: Optional[float] = None,
        jitter: Optional[bool] = None,
        error_handling: Optional[Any] = None,
        **options: Any
    ):
        """
        Create a text embedder descriptor.

        Args:
            model: Model name (uses DEFAULT_TEXT_EMBEDDER if not specified)
            dimensions: Custom dimensions (if model supports it)
            retry_strategy: Retry strategy (RetryStrategy enum or string)
            max_retries: Maximum retry attempts (for LIMITED strategy)
            initial_delay: Initial delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds for exponential backoff
            exponential_base: Exponential base for backoff calculation
            jitter: Whether to add random jitter to delays
            error_handling: Error handling strategy (ErrorHandlingStrategy enum or string)
            **options: Additional model options

        Returns:
            TextEmbedderDescriptor

        Raises:
            ValueError: If the model is not supported by this provider
        """
        from ..protocols.text_embedder import TextEmbedderDescriptor

        # Determine model name
        model_name = model or self.DEFAULT_TEXT_EMBEDDER

        # Validate model is supported by this provider
        if model_name not in self.SUPPORT_EMBEDDER:
            raise ValueError(
                f"Model '{model_name}' is not supported by {self._name} provider. "
                f"Supported models: {', '.join(self.SUPPORT_EMBEDDER)}. "
                f"Note: This validation checks provider compatibility. "
                f"If using a custom model, ensure it's registered via register_custom_model()."
            )

        # Build descriptor kwargs (only include non-None values)
        descriptor_kwargs = {
            "provider_name": self._name,
            "provider_options": self._options,
            "max_batch_tokens": self._max_batch_tokens,
            "model_name": model_name,
            "dimensions": dimensions,
            "model_options": options,
        }

        # Add retry/error handling params if provided
        if retry_strategy is not None:
            descriptor_kwargs["retry_strategy"] = retry_strategy
        if max_retries is not None:
            descriptor_kwargs["max_retries"] = max_retries
        if initial_delay is not None:
            descriptor_kwargs["initial_delay"] = initial_delay
        if max_delay is not None:
            descriptor_kwargs["max_delay"] = max_delay
        if exponential_base is not None:
            descriptor_kwargs["exponential_base"] = exponential_base
        if jitter is not None:
            descriptor_kwargs["jitter"] = jitter
        if error_handling is not None:
            descriptor_kwargs["error_handling"] = error_handling

        return TextEmbedderDescriptor(**descriptor_kwargs)

    def get_pdf_parser(
        self,
        parser_type: Optional[str] = None,
        document_type: str = "pdf",
        parser_mode: str = "pipeline",
        poll_interval: int = 2,
        poll_timeout: int = 300,
        custom_options: Optional[Dict[str, Any]] = None,
        retry_strategy: Optional[Any] = None,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        exponential_base: Optional[float] = None,
        jitter: Optional[bool] = None,
        error_handling: Optional[Any] = None,
        **options: Any
    ):
        """
        Create a PDF parser descriptor.

        Args:
            parser_type: Parser backend (uses DEFAULT_PDF_PARSER if not specified)
            document_type: Type of document (default: "pdf")
            parser_mode: Parsing mode (default: "pipeline")
            poll_interval: Seconds between status polls (default: 2)
            poll_timeout: Maximum seconds to wait for job completion (default: 300)
            custom_options: Additional parser options (default: {})
            retry_strategy: Retry strategy (RetryStrategy enum or string)
            max_retries: Maximum retry attempts (for LIMITED strategy)
            initial_delay: Initial delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds for exponential backoff
            exponential_base: Exponential base for backoff calculation
            jitter: Whether to add random jitter to delays
            error_handling: Error handling strategy (ErrorHandlingStrategy enum or string)
            **options: Additional options

        Returns:
            PDFParserDescriptor

        Raises:
            ValueError: If the parser type is not supported by this provider

        Example:
            >>> provider = AIWorksProvider(
            ...     name="AIWorks",
            ...     base_url="http://172.16.99.68:8011"
            ... )
            >>> descriptor = provider.get_pdf_parser(
            ...     parser_type="mineru",
            ...     retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
            ...     max_retries=3,
            ...     error_handling=ErrorHandlingStrategy.FAIL_FAST
            ... )
            >>> parser = descriptor.instantiate()
            >>> result = parser.parse_files(files, output_dir)
        """
        from ..protocols.pdf_parser import PDFParserDescriptor

        # Determine parser type
        parser_name = parser_type or self.DEFAULT_PDF_PARSER

        # Validate parser is supported by this provider
        if parser_name not in self.SUPPORT_PDF_PARSER:
            raise ValueError(
                f"Parser '{parser_name}' is not supported by {self._name} provider. "
                f"Supported parsers: {', '.join(self.SUPPORT_PDF_PARSER)}."
            )

        # Build descriptor kwargs (only include non-None values)
        descriptor_kwargs = {
            "provider_name": self._name,
            "provider_options": self._options,
            "parser_type": parser_name,
            "document_type": document_type,
            "parser_mode": parser_mode,
            "poll_interval": poll_interval,
            "poll_timeout": poll_timeout,
            "custom_options": custom_options or {},
        }

        # Add retry params if provided
        if retry_strategy is not None:
            descriptor_kwargs["retry_strategy"] = retry_strategy
        if max_retries is not None:
            descriptor_kwargs["max_retries"] = max_retries
        if initial_delay is not None:
            descriptor_kwargs["initial_delay"] = initial_delay
        if max_delay is not None:
            descriptor_kwargs["max_delay"] = max_delay
        if exponential_base is not None:
            descriptor_kwargs["exponential_base"] = exponential_base
        if jitter is not None:
            descriptor_kwargs["jitter"] = jitter
        if error_handling is not None:
            descriptor_kwargs["error_handling"] = error_handling

        return PDFParserDescriptor(**descriptor_kwargs)
