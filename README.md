# AI Embedder

English | [简体中文](README.zh-CN.md)

A flexible and efficient text embedding library supporting multiple AI providers with intelligent batching and model management.

## Features

- Multi-Provider Support: Seamlessly switch between OpenAI, AIWorks, and other compatible providers
- Intelligent Batching: Automatically optimizes API calls by batching requests while respecting token limits
- Model Registry: Centralized model profile management with configurable dimensions and token limits
- Chunking Support: Handles oversized texts through intelligent chunking and weighted averaging
- Type-Safe: Full type hints for better IDE support and code reliability
- Extensible: Easy to add new providers and models

## Installation

### Using uv (Recommended)

```bash
# Install with base dependencies
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"

# Install with visualization support
uv pip install -e ".[viz]"
```

### Using pip

```bash
# Install with base dependencies
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage with AIWorks Provider

```python
from ai.providers.aiworks_provider import AIWorksProvider

# Create provider
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.10.163:9997/v1",
    max_batch_tokens=100_000,
)

# Get embedder descriptor
descriptor = provider.get_text_embedder(model="conan-embedding-v1")

# Instantiate embedder
embedder = descriptor.instantiate()

# Embed texts
texts = [
    "人工智能正在改变世界",
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络",
]

embeddings = embedder.embed_text(texts)
print(f"Generated {len(embeddings)} embeddings")
print(f"Dimension: {len(embeddings[0])}")
```

### Using OpenAI Provider

```python
from ai.providers.openai_provider import OpenAIProvider

# Create provider
provider = OpenAIProvider(
    api_key="your-api-key-here",
    max_batch_tokens=300_000,
)

# Get embedder with custom dimensions
descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    dimensions=512  # Optional: override default dimensions
)

embedder = descriptor.instantiate()
embeddings = embedder.embed_text(["Hello, world!"])
```

## Project Structure

```
ai-embedder/
├── ai/                          # Main package
│   ├── __init__.py
│   ├── models.py                # Model registry and profiles
│   ├── typing.py                # Type definitions
│   ├── protocols/               # Protocol implementations
│   │   ├── __init__.py
│   │   └── text_embedder.py    # TextEmbedder implementation
│   ├── providers/               # Provider implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base provider interface
│   │   ├── aiworks_provider.py # AIWorks provider
│   │   └── openai_provider.py  # OpenAI provider
│   └── utils/                   # Utility functions
│       └── embedding_viz.py     # Visualization tools
├── tests/                       # Test suite
├── examples/                    # Example scripts
│   └── test_embedding_aiworks.py
├── pyproject.toml              # Project configuration
├── README.md                   # English documentation (this file)
├── README.zh-CN.md            # Chinese documentation
└── .gitignore                  # Git ignore rules
```

## Architecture

### Provider-Descriptor-Embedder Pattern

The library follows a three-tier architecture:

1. **Provider**: Stores API-level configurations (base_url, api_key, max_batch_tokens)
2. **Descriptor**: Serializable configuration for creating embedders
3. **Embedder**: The actual embedding instance that makes API calls

This pattern allows for:
- Clean separation between API and model configurations
- Serializable configurations for distributed computing
- Easy testing and mocking

### Intelligent Batching

The embedder automatically handles:
- **Batch accumulation**: Groups texts to maximize API efficiency
- **Token limit respect**: Ensures batches stay within API limits
- **Oversized text handling**: Chunks and merges large texts
- **Order preservation**: Maintains input-output correspondence

## Configuration

### Supported Models

#### OpenAI Models
- `text-embedding-ada-002` (1536 dimensions)
- `text-embedding-3-small` (512/1536 dimensions, configurable)
- `text-embedding-3-large` (256/1024/3072 dimensions, configurable)

#### AIWorks Models
- `conan-embedding-v1` (Custom dimensions)

### Adding Custom Models

```python
from ai.models import register_custom_model, EmbeddingDimensions

register_custom_model(
    model_name="my-custom-model",
    max_input_tokens=8192,
    dimensions=EmbeddingDimensions(size=768, dtype="float32"),
    supports_overriding_dimensions=False
)
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai --cov-report=html

# Run specific test file
pytest tests/test_basic.py
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy ai/
```

## API Reference

### Provider Classes

- `OpenAIProvider`: OpenAI-compatible API provider
- `AIWorksProvider`: AIWorks API provider

### Core Classes

- `TextEmbedderDescriptor`: Serializable embedder configuration
- `TextEmbedder`: Main embedding class with batching logic

### Utility Functions

- `register_custom_model()`: Register new model profiles
- `get_model_profile()`: Retrieve model configurations
- `chunk_text()`: Split text into chunks

## Examples

See the `examples/` directory for complete working examples:

- `test_embedding_aiworks.py`: Complete example using AIWorks provider

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- requests >= 2.25.0
- typing-extensions >= 4.0.0 (for Python < 3.11)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with modern Python best practices and type safety in mind.

