"""
Type definitions for AI module
"""
from typing import TypedDict, List
from dataclasses import dataclass


class OpenAIProviderOptions(TypedDict, total=False):
    """OpenAI Provider configuration options"""
    api_key: str
    base_url: str
    organization: str
    timeout: int
    max_retries: int


@dataclass(frozen=True)
class EmbeddingDimensions:
    """Embedding vector dimensions configuration"""
    size: int  # Vector dimension size (e.g., 1536, 3072)
    dtype: str = "float32"  # Data type for embeddings


# Type aliases
Embedding = List[float]
Options = dict
