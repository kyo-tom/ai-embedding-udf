"""
Base Provider class
"""
from abc import ABC, abstractmethod
from typing import Optional, Any


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
        api_endpoint: Optional[str] = None,
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
