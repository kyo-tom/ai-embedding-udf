"""
Prism AI Module

This module provides AI capabilities with a clean separation of concerns:

- providers: API service providers (OpenAI, Azure, custom endpoints)
- protocols: AI protocol interfaces (TextEmbedder, PDFParser, etc.)
- models: Model configurations and profiles
- typing: Type definitions

Key Design Principles:
1. Model properties (dimensions, context window) are separate from API properties (batch limits)
2. Descriptor pattern for serialization in distributed environments
3. Smart batching to minimize API calls while respecting limits
4. Unified retry and error handling strategies
"""
from .providers import Provider, OpenAIProvider, AIWorksProvider
from .protocols import (
    TextEmbedder,
    TextEmbedderDescriptor,
    PDFParser,
    PDFParserDescriptor,
    PDFParseError,
    FileParseResult,
    BatchParseResult,
    RetryStrategy,
    ErrorHandlingStrategy,
)
from .models import register_custom_model, get_model_profile, MODELS
from .typing import Embedding, EmbeddingDimensions, OpenAIProviderOptions

__version__ = "0.1.0"

__all__ = [
    # Providers
    "Provider",
    "OpenAIProvider",
    "AIWorksProvider",
    # Text Embedder Protocol
    "TextEmbedder",
    "TextEmbedderDescriptor",
    # PDF Parser Protocol
    "PDFParser",
    "PDFParserDescriptor",
    "PDFParseError",
    "FileParseResult",
    "BatchParseResult",
    # Retry and Error Handling (Shared)
    "RetryStrategy",
    "ErrorHandlingStrategy",
    # Models
    "register_custom_model",
    "get_model_profile",
    "MODELS",
    # Types
    "Embedding",
    "EmbeddingDimensions",
    "OpenAIProviderOptions",
]
