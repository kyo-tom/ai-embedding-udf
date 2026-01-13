"""
Shared retry utilities for AI protocols.

This module provides common retry logic used by TextEmbedder and PDFParser.
"""
import time
import random
import logging
from enum import Enum
from typing import Protocol, Optional

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy for API calls"""
    NO_RETRY = "no_retry"  # No retry
    EXPONENTIAL_BACKOFF_LIMITED = "exponential_backoff_limited"  # Exponential backoff with max retries
    EXPONENTIAL_BACKOFF_UNLIMITED = "exponential_backoff_unlimited"  # Exponential backoff with unlimited retries


class ErrorHandlingStrategy(Enum):
    """Error handling strategy for failed API calls"""
    FAIL_FAST = "fail_fast"  # Raise exception immediately
    ZERO_VECTOR_FALLBACK = "zero_vector_fallback"  # Return zero vectors as fallback


class RetryConfig(Protocol):
    """Protocol defining retry configuration interface."""
    retry_strategy: RetryStrategy
    max_retries: int
    initial_delay: float
    max_delay: float
    exponential_base: float
    jitter: bool


def calculate_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """
    Calculate delay for exponential backoff.

    Args:
        attempt: Current retry attempt (0-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds

    Example:
        >>> calculate_delay(0, 1.0, 60.0, 2.0, False)
        1.0
        >>> calculate_delay(1, 1.0, 60.0, 2.0, False)
        2.0
        >>> calculate_delay(2, 1.0, 60.0, 2.0, False)
        4.0
    """
    delay = min(
        initial_delay * (exponential_base ** attempt),
        max_delay
    )

    if jitter:
        # Add random jitter (±25%)
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def should_retry(
    attempt: int,
    exception: Exception,
    retry_strategy: RetryStrategy,
    max_retries: int
) -> bool:
    """
    Determine if we should retry based on the retry strategy.

    Args:
        attempt: Current retry attempt (0-indexed)
        exception: The exception that occurred
        retry_strategy: The retry strategy to use
        max_retries: Maximum number of retries (for LIMITED strategy)

    Returns:
        True if we should retry, False otherwise

    Example:
        >>> should_retry(0, Exception(), RetryStrategy.NO_RETRY, 3)
        False
        >>> should_retry(0, Exception(), RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED, 3)
        True
        >>> should_retry(3, Exception(), RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED, 3)
        False
        >>> should_retry(100, Exception(), RetryStrategy.EXPONENTIAL_BACKOFF_UNLIMITED, 3)
        True
    """
    if retry_strategy == RetryStrategy.NO_RETRY:
        return False

    if retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED:
        return attempt < max_retries

    if retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF_UNLIMITED:
        return True

    return False


def retry_with_backoff(
    func,
    config: RetryConfig,
    operation_name: str = "operation"
):
    """
    Execute a function with retry and exponential backoff.

    Args:
        func: Function to execute (should raise exception on failure)
        config: Retry configuration object
        operation_name: Name of operation for logging

    Returns:
        Result of successful function execution

    Raises:
        Exception: The last exception if all retries are exhausted

    Example:
        >>> def api_call():
        ...     response = requests.get("https://api.example.com")
        ...     response.raise_for_status()
        ...     return response.json()
        >>>
        >>> config = MyConfig(
        ...     retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
        ...     max_retries=3,
        ...     initial_delay=1.0,
        ...     max_delay=60.0,
        ...     exponential_base=2.0,
        ...     jitter=True
        ... )
        >>> result = retry_with_backoff(api_call, config, "API call")
    """
    attempt = 0
    last_exception = None

    while True:
        try:
            return func()

        except Exception as e:
            last_exception = e
            logger.error(f"{operation_name} failed (attempt {attempt + 1}): {e}")

            # Check if we should retry
            if should_retry(attempt, e, config.retry_strategy, config.max_retries):
                delay = calculate_delay(
                    attempt,
                    config.initial_delay,
                    config.max_delay,
                    config.exponential_base,
                    config.jitter
                )
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                attempt += 1
                continue
            else:
                # No more retries
                break

    # All retries exhausted
    raise last_exception
