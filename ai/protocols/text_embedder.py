"""
Text Embedder Implementation with Intelligent Batching

This module implements an improved text embedding system with clear separation
of concerns between model properties and API provider properties.
"""
import asyncio
import logging
import time
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import requests
import numpy as np

from ..typing import OpenAIProviderOptions, Embedding, EmbeddingDimensions, Options
from ..models import get_model_profile
from ..utils import calculate_delay, should_retry, RetryStrategy, ErrorHandlingStrategy

logger = logging.getLogger(__name__)


def parse_retry_strategy(value: Optional[str | RetryStrategy]) -> RetryStrategy:
    """
    Parse retry strategy from string or enum.

    Args:
        value: String or RetryStrategy enum

    Returns:
        RetryStrategy enum

    Raises:
        ValueError: If string value is invalid
    """
    if value is None:
        return RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED

    if isinstance(value, RetryStrategy):
        return value

    if isinstance(value, str):
        value_lower = value.lower()
        for strategy in RetryStrategy:
            if strategy.value == value_lower:
                return strategy
        raise ValueError(
            f"Invalid retry strategy: '{value}'. "
            f"Valid options: {[s.value for s in RetryStrategy]}"
        )

    raise TypeError(f"retry_strategy must be str or RetryStrategy, got {type(value)}")


def parse_error_handling(value: Optional[str | ErrorHandlingStrategy]) -> ErrorHandlingStrategy:
    """
    Parse error handling strategy from string or enum.

    Args:
        value: String or ErrorHandlingStrategy enum

    Returns:
        ErrorHandlingStrategy enum

    Raises:
        ValueError: If string value is invalid
    """
    if value is None:
        return ErrorHandlingStrategy.FAIL_FAST

    if isinstance(value, ErrorHandlingStrategy):
        return value

    if isinstance(value, str):
        value_lower = value.lower()
        for strategy in ErrorHandlingStrategy:
            if strategy.value == value_lower:
                return strategy
        raise ValueError(
            f"Invalid error handling strategy: '{value}'. "
            f"Valid options: {[s.value for s in ErrorHandlingStrategy]}"
        )

    raise TypeError(f"error_handling must be str or ErrorHandlingStrategy, got {type(value)}")


@dataclass
class TextEmbedderDescriptor:
    """
    Text Embedder Descriptor (Serializable Configuration)

    This stores all configuration needed to create a TextEmbedder instance.
    It's designed to be serializable for distributed computing environments.

    Design Pattern:
    - Descriptor stores configuration (serializable)
    - instantiate() creates the actual embedder (non-serializable, has API clients)
    """
    provider_name: str
    provider_options: OpenAIProviderOptions
    max_batch_tokens: int  # API-level limit from Provider
    model_name: str
    dimensions: Optional[int]
    model_options: Options

    # Retry and error handling configuration
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED
    max_retries: int = 3  # Maximum retry attempts (for LIMITED strategy only)
    initial_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential base for backoff calculation
    jitter: bool = True  # Whether to add random jitter to delays
    error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST

    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        # Convert string strategies to enums if needed
        if not isinstance(self.retry_strategy, RetryStrategy):
            object.__setattr__(self, 'retry_strategy', parse_retry_strategy(self.retry_strategy))

        if not isinstance(self.error_handling, ErrorHandlingStrategy):
            object.__setattr__(self, 'error_handling', parse_error_handling(self.error_handling))

        # Only validate for official OpenAI (base_url is None)
        if self.provider_options.get("base_url") is None:
            # Get model profile and validate
            model = get_model_profile(self.model_name)

            # Check if custom dimensions are supported
            if self.dimensions is not None and not model.supports_overriding_dimensions:
                raise ValueError(
                    f"Model '{self.model_name}' does not support custom dimensions"
                )

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_dimensions(self) -> EmbeddingDimensions:
        """Get embedding dimensions (custom or model default)"""
        if self.dimensions is not None:
            return EmbeddingDimensions(size=self.dimensions, dtype="float32")

        # Use model's default dimensions
        model = get_model_profile(self.model_name)
        return model.dimensions

    def get_max_batch_tokens(self) -> int:
        """Get maximum tokens per batch (API limit)"""
        return self.max_batch_tokens

    def instantiate(self) -> "TextEmbedder":
        """Create actual TextEmbedder instance (non-serializable)"""
        logger.info(f"Instantiating TextEmbedder: {self.model_name} via {self.provider_name}")
        return TextEmbedder(
            base_url=self.provider_options.get("base_url", "https://api.openai.com/v1"),
            api_key=self.provider_options.get("api_key", ""),
            model=self.model_name,
            dimensions=self.dimensions,
            max_batch_tokens=self.max_batch_tokens,
            retry_strategy=self.retry_strategy,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
            error_handling=self.error_handling,
        )


class TextEmbedder:
    """
    Text Embedder with Intelligent Batching

    Key Features:
    1. Respects both model and API limits:
       - Model limit: max_input_tokens (from model profile)
       - API limit: max_batch_tokens (from provider)
    2. Handles oversized inputs by chunking and weighted averaging
    3. Maintains output order strictly matching input order
    4. Minimizes API calls through smart batching

    Example:
        embedder = TextEmbedder(
            base_url="http://localhost:9997/v1",
            api_key="empty",
            model="conan-embedding-v1",
            max_batch_tokens=100_000  # Custom API limit
        )

        texts = ["Hello", "World", "...very long text..."]
        embeddings = embedder.embed_text(texts)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: Optional[int] = None,
        max_batch_tokens: int = 300_000,  # API limit (from Provider)
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.max_batch_tokens = max_batch_tokens

        # Retry configuration
        self.retry_strategy = retry_strategy
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

        # Error handling configuration
        self.error_handling = error_handling

        # Get model-specific limits
        model_profile = get_model_profile(model)
        self.max_input_tokens = model_profile.max_input_tokens  # Model's context window

        # Token estimation (conservative: 1 token ≈ 3-4 chars)
        self.approx_chars_per_token = 3

        logger.info(
            f"TextEmbedder initialized: model={model}, "
            f"max_input_tokens={self.max_input_tokens} (model limit), "
            f"max_batch_tokens={self.max_batch_tokens} (API limit), "
            f"retry_strategy={retry_strategy.value}, "
            f"error_handling={error_handling.value}"
        )

    def embed_text(self, texts: List[str]) -> List[Embedding]:
        """
        Embed a list of texts with intelligent batching.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (same order as input)

        Algorithm:
        1. Process texts in order, accumulating into batches
        2. When encountering oversized text (> max_input_tokens):
           a. Flush accumulated batch first (maintain order)
           b. Chunk the oversized text
           c. Embed chunks and merge with weighted average
        3. When batch reaches max_batch_tokens, flush and start new batch
        4. Final flush for remaining texts
        """
        embeddings: List[Embedding] = []
        curr_batch: List[str] = []
        curr_batch_token_count: int = 0

        input_text_chars_limit = self.max_input_tokens * self.approx_chars_per_token

        def flush() -> None:
            """Process accumulated batch"""
            nonlocal curr_batch, curr_batch_token_count

            if len(curr_batch) == 0:
                return

            batch_embeddings = self._embed_text_batch(curr_batch)
            embeddings.extend(batch_embeddings)

            # Reset batch
            curr_batch = []
            curr_batch_token_count = 0

        # Process texts in order
        for input_text in texts:
            # Handle None
            if input_text is None:
                input_text = ""

            # Estimate token count
            input_text_token_count = len(input_text) // self.approx_chars_per_token

            # Case 1: Oversized single text (> max_input_tokens)
            if input_text_token_count > self.max_input_tokens:
                logger.warning(
                    f"Text exceeds max_input_tokens ({self.max_input_tokens}), "
                    f"chunking and averaging..."
                )

                # CRITICAL: Flush previous batch first to maintain order
                flush()

                # Chunk the oversized text
                chunks = chunk_text(input_text, input_text_chars_limit)
                chunk_embeddings = self._embed_text_batch(chunks)

                # Weighted average by chunk length
                chunk_lens = [len(chunk) for chunk in chunks]
                merged_vec = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
                merged_vec = merged_vec / np.linalg.norm(merged_vec)  # Normalize

                embeddings.append(merged_vec.tolist())

            # Case 2: Adding this text would exceed batch limit
            elif input_text_token_count + curr_batch_token_count >= self.max_batch_tokens:
                # Flush current batch, then add this text to new batch
                flush()
                curr_batch.append(input_text)
                curr_batch_token_count = input_text_token_count

            # Case 3: Normal accumulation
            else:
                curr_batch.append(input_text)
                curr_batch_token_count += input_text_token_count

        # Final flush
        flush()

        return embeddings

    def _embed_text_batch(self, texts: List[str]) -> List[Embedding]:
        """
        Call embedding API for a batch of texts with retry and error handling.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            requests.RequestException: If API call fails and error_handling is FAIL_FAST
        """
        if not texts:
            return []

        # Prepare request
        payload = {
            "model": self.model,
            "input": texts,
        }

        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        attempt = 0
        last_exception = None

        while True:
            try:
                # Call API
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                response.raise_for_status()

                result = response.json()

                # Extract embeddings (OpenAI format)
                embeddings = []
                for item in sorted(result.get("data", []), key=lambda x: x.get("index", 0)):
                    embeddings.append(item.get("embedding", []))

                return embeddings

            except requests.RequestException as e:
                last_exception = e
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")

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
                    # No more retries, handle the error
                    break

        # Error handling: no more retries
        if self.error_handling == ErrorHandlingStrategy.FAIL_FAST:
            logger.error(f"All retries exhausted, raising exception")
            raise last_exception

        elif self.error_handling == ErrorHandlingStrategy.ZERO_VECTOR_FALLBACK:
            logger.warning(f"All retries exhausted, falling back to zero vectors")
            # Get embedding dimensions
            if self.dimensions is not None:
                embedding_dim = self.dimensions
            else:
                model_profile = get_model_profile(self.model)
                embedding_dim = model_profile.dimensions.size

            # Return zero vectors for each text
            return [[0.0] * embedding_dim for _ in texts]

        # Should not reach here
        raise last_exception


def chunk_text(text: str, size: int) -> List[str]:
    """
    Split text into chunks of specified character size.

    Args:
        text: Text to split
        size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    return [text[i : i + size] for i in range(0, len(text), size)]
