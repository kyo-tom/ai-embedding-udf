"""
Model profiles and configurations
"""
from dataclasses import dataclass
from typing import Dict
from .typing import EmbeddingDimensions


@dataclass(frozen=True)
class ModelProfile:
    """
    Model profile containing model-specific metadata.

    This stores MODEL INHERENT properties that don't change regardless of
    which API provider is used (OpenAI official, Azure, custom endpoint, etc.)

    Attributes:
        dimensions: Default embedding dimensions for this model
        supports_overriding_dimensions: Whether the model supports custom dimensions
        max_input_tokens: Maximum tokens per single input text (context window)
                         This is determined during model training and cannot be exceeded
    """
    dimensions: EmbeddingDimensions
    supports_overriding_dimensions: bool
    max_input_tokens: int = 8191  # OpenAI embedding models standard context window


# Predefined OpenAI embedding models
# These are MODEL configurations, not API configurations
MODELS: Dict[str, ModelProfile] = {
    "text-embedding-ada-002": ModelProfile(
        dimensions=EmbeddingDimensions(size=1536, dtype="float32"),
        supports_overriding_dimensions=False,  # v2 doesn't support custom dimensions
        max_input_tokens=8191,
    ),
    "text-embedding-3-small": ModelProfile(
        dimensions=EmbeddingDimensions(size=1536, dtype="float32"),
        supports_overriding_dimensions=True,  # v3 supports dimensions 256-1536
        max_input_tokens=8191,
    ),
    "text-embedding-3-large": ModelProfile(
        dimensions=EmbeddingDimensions(size=3072, dtype="float32"),
        supports_overriding_dimensions=True,  # v3 supports dimensions 256-3072
        max_input_tokens=8191,
    ),
    "conan-embedding-v1": ModelProfile(
        dimensions=EmbeddingDimensions(size=1792, dtype="float32"),
        supports_overriding_dimensions=False,
        max_input_tokens=8191,
    ),
}


def register_custom_model(
    model_name: str,
    dimensions: int,
    supports_overriding_dimensions: bool = False,
    max_input_tokens: int = 8191,
) -> None:
    """
    Register a custom embedding model.

    Example:
        register_custom_model(
            "conan-embedding-v1",
            dimensions=1792,
            supports_overriding_dimensions=False,
            max_input_tokens=8191
        )
    """
    MODELS[model_name] = ModelProfile(
        dimensions=EmbeddingDimensions(size=dimensions, dtype="float32"),
        supports_overriding_dimensions=supports_overriding_dimensions,
        max_input_tokens=max_input_tokens,
    )


def get_model_profile(model_name: str) -> ModelProfile:
    """Get model profile by name, raise ValueError if not found"""
    if model_name not in MODELS:
        supported = ", ".join(MODELS.keys())
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {supported}. "
            f"Use register_custom_model() to add custom models."
        )
    return MODELS[model_name]
