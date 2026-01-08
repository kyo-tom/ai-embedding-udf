"""
Prism AI Module

This module provides AI capabilities with a clean separation of concerns:

- providers: API service providers (OpenAI, Azure, custom endpoints)
- protocols: AI protocols (TextEmbedder, etc.)
- models: Model configurations and profiles
- typing: Type definitions

Key Design Principles:
1. Model properties (dimensions, context window) are separate from API properties (batch limits)
2. Descriptor pattern for serialization in distributed environments
3. Smart batching to minimize API calls while respecting limits
"""
from .providers import Provider, OpenAIProvider
from .protocols import TextEmbedder, TextEmbedderDescriptor
from .protocols.text_embedder import (
    RetryStrategy,
    ErrorHandlingStrategy,
    parse_retry_strategy,
    parse_error_handling,
)
from .models import register_custom_model, get_model_profile, MODELS
from .typing import Embedding, EmbeddingDimensions, OpenAIProviderOptions

__version__ = "0.1.0"

__all__ = [
    # Providers
    "Provider",
    "OpenAIProvider",
    # Protocols
    "TextEmbedder",
    "TextEmbedderDescriptor",
    # Retry and Error Handling
    "RetryStrategy",
    "ErrorHandlingStrategy",
    "parse_retry_strategy",
    "parse_error_handling",
    # Models
    "register_custom_model",
    "get_model_profile",
    "MODELS",
    # Types
    "Embedding",
    "EmbeddingDimensions",
    "OpenAIProviderOptions",
]
