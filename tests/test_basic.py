"""
Basic tests for text embedder functionality
"""
import pytest
from ai.providers.aiworks_provider import AIWorksProvider
from ai.providers.openai_provider import OpenAIProvider
from ai import RetryStrategy, ErrorHandlingStrategy


class TestAIWorksProvider:
    """Tests for AIWorksProvider"""

    def test_provider_initialization(self):
        """Test provider can be initialized"""
        provider = AIWorksProvider(
            name="TestAIWorks",
            base_url="http://localhost:9997/v1",
            max_batch_tokens=100_000,
        )
        assert provider.name == "TestAIWorks"
        assert provider.max_batch_tokens == 100_000

    def test_get_text_embedder_descriptor(self):
        """Test getting embedder descriptor"""
        provider = AIWorksProvider(
            name="TestAIWorks",
            base_url="http://localhost:9997/v1",
            max_batch_tokens=100_000,
        )
        descriptor = provider.get_text_embedder(model="conan-embedding-v1")
        assert descriptor.get_model() == "conan-embedding-v1"
        assert descriptor.get_provider() == "TestAIWorks"
        assert descriptor.get_max_batch_tokens() == 100_000

    def test_unsupported_model_raises_error(self):
        """Test that unsupported model raises ValueError"""
        provider = AIWorksProvider(
            name="TestAIWorks",
            base_url="http://localhost:9997/v1",
        )
        with pytest.raises(ValueError, match="not supported"):
            provider.get_text_embedder(model="unsupported-model")


class TestOpenAIProvider:
    """Tests for OpenAIProvider"""

    def test_provider_initialization(self):
        """Test provider can be initialized"""
        provider = OpenAIProvider(
            name="TestOpenAI",
            api_key="test-key",
            max_batch_tokens=300_000,
        )
        assert provider.name == "TestOpenAI"
        assert provider.max_batch_tokens == 300_000

    def test_default_model(self):
        """Test default model is set correctly"""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.DEFAULT_TEXT_EMBEDDER == "text-embedding-3-small"

    def test_supported_models(self):
        """Test supported models list"""
        provider = OpenAIProvider(api_key="test-key")
        assert "text-embedding-ada-002" in provider.SUPPORT_EMBEDDER
        assert "text-embedding-3-small" in provider.SUPPORT_EMBEDDER
        assert "text-embedding-3-large" in provider.SUPPORT_EMBEDDER


class TestTextEmbedderDescriptor:
    """Tests for TextEmbedderDescriptor"""

    def test_descriptor_attributes(self):
        """Test descriptor stores configuration correctly"""
        provider = AIWorksProvider(
            name="TestProvider",
            base_url="http://localhost:9997/v1",
            max_batch_tokens=50_000,
        )
        descriptor = provider.get_text_embedder(model="conan-embedding-v1")

        assert descriptor.provider_name == "TestProvider"
        assert descriptor.model_name == "conan-embedding-v1"
        assert descriptor.max_batch_tokens == 50_000

    def test_descriptor_instantiate(self):
        """Test descriptor can instantiate embedder"""
        provider = AIWorksProvider(
            name="TestProvider",
            base_url="http://localhost:9997/v1",
            max_batch_tokens=50_000,
        )
        descriptor = provider.get_text_embedder(model="conan-embedding-v1")
        embedder = descriptor.instantiate()

        assert embedder.model == "conan-embedding-v1"
        assert embedder.max_batch_tokens == 50_000


class TestRetryAndErrorHandling:
    """Tests for retry and error handling strategies"""

    def test_retry_strategy_default(self):
        """Test default retry strategy"""
        provider = AIWorksProvider(
            name="TestProvider",
            base_url="http://localhost:9997/v1",
        )
        descriptor = provider.get_text_embedder(model="conan-embedding-v1")
        assert descriptor.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED
        assert descriptor.max_retries == 3

    def test_retry_strategy_custom(self):
        """Test custom retry strategy"""
        provider = AIWorksProvider(
            name="TestProvider",
            base_url="http://localhost:9997/v1",
        )
        descriptor = provider.get_text_embedder(
            model="conan-embedding-v1",
            retry_strategy=RetryStrategy.NO_RETRY,
            max_retries=5,
        )
        assert descriptor.retry_strategy == RetryStrategy.NO_RETRY
        assert descriptor.max_retries == 5

    def test_error_handling_default(self):
        """Test default error handling strategy"""
        provider = AIWorksProvider(
            name="TestProvider",
            base_url="http://localhost:9997/v1",
        )
        descriptor = provider.get_text_embedder(model="conan-embedding-v1")
        assert descriptor.error_handling == ErrorHandlingStrategy.FAIL_FAST

    def test_error_handling_zero_vector_fallback(self):
        """Test zero vector fallback error handling"""
        provider = AIWorksProvider(
            name="TestProvider",
            base_url="http://localhost:9997/v1",
        )
        descriptor = provider.get_text_embedder(
            model="conan-embedding-v1",
            error_handling=ErrorHandlingStrategy.ZERO_VECTOR_FALLBACK,
        )
        assert descriptor.error_handling == ErrorHandlingStrategy.ZERO_VECTOR_FALLBACK

    def test_retry_configuration(self):
        """Test retry configuration parameters"""
        provider = AIWorksProvider(
            name="TestProvider",
            base_url="http://localhost:9997/v1",
        )
        descriptor = provider.get_text_embedder(
            model="conan-embedding-v1",
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
            max_retries=5,
            initial_delay=2.0,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert descriptor.max_retries == 5
        assert descriptor.initial_delay == 2.0
        assert descriptor.max_delay == 30.0
        assert descriptor.exponential_base == 3.0
        assert descriptor.jitter is False
