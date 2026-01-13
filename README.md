# AI Embedder & PDF Parser

English | [简体中文](README.zh-CN.md)

A flexible and efficient AI library supporting text embedding and PDF parsing with multiple providers, intelligent batching, and robust error handling.

## Features

### Text Embedding
- **Multi-Provider Support**: Seamlessly switch between OpenAI, AIWorks, and other compatible providers
- **Intelligent Batching**: Automatically optimizes API calls by batching requests while respecting token limits
- **Model Registry**: Centralized model profile management with configurable dimensions and token limits
- **Chunking Support**: Handles oversized texts through intelligent chunking and weighted averaging
- **Retry Strategies**: Configurable retry mechanisms with exponential backoff
- **Error Handling**: Multiple error handling strategies (fail-fast or zero-vector fallback)

### PDF Parsing
- **Async Job Processing**: Submit parsing jobs and poll for completion with configurable timeouts
- **Batch Processing**: Parse multiple PDF files with partial success support
- **Retry Support**: Built-in retry logic with exponential backoff for API reliability
- **Error Handling**: Configurable error handling (fail-fast or continue on error)
- **MinIO Integration**: Direct integration with MinIO object storage

### Shared Features
- **Type-Safe**: Full type hints for better IDE support and code reliability
- **Extensible**: Easy to add new providers and models
- **Logging**: Secure logging with automatic API key sanitization

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

### Text Embedding with AIWorks Provider

```python
from ai.providers import AIWorksProvider

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
    "Artificial intelligence is changing the world",
    "Machine learning is a branch of AI",
    "Deep learning uses neural networks",
]

embeddings = embedder.embed_text(texts)
print(f"Generated {len(embeddings)} embeddings")
print(f"Dimension: {len(embeddings[0])}")
```

### PDF Parsing with AIWorks Provider

```python
from ai.providers import AIWorksProvider
from ai.protocols import RetryStrategy, ErrorHandlingStrategy

# Create provider
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.99.68:8011",
    max_batch_tokens=100_000
)

# Get PDF parser descriptor
descriptor = provider.get_pdf_parser(
    parser_type="mineru",
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=5,
    error_handling=ErrorHandlingStrategy.FAIL_FAST
)

# Instantiate parser
parser = descriptor.instantiate()

# Parse files
files = [
    "/uni-parse-documents/documents/report1.pdf",
    "/uni-parse-documents/documents/report2.pdf",
]

result = parser.parse_files(
    files=files,
    source_parent_path="/uni-parse-documents/output"
)

print(f"Success: {result.success_count}, Failed: {result.failed_count}")
for path in result.successful:
    print(f"✓ {path}")
```

### Using OpenAI Provider

```python
from ai.providers import OpenAIProvider

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
│   │   ├── text_embedder.py    # TextEmbedder implementation
│   │   └── pdf_parser.py       # PDFParser implementation
│   ├── providers/               # Provider implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base provider interface
│   │   ├── aiworks_provider.py # AIWorks provider
│   │   └── openai_provider.py  # OpenAI provider
│   └── utils/                   # Utility functions
│       ├── retry_utils.py       # Retry strategies and utilities
│       ├── logging_utils.py     # Secure logging utilities
│       └── embedding_viz.py     # Visualization tools
├── tests/                       # Test suite
├── examples/                    # Example scripts
│   ├── example_text_embedder.py
│   └── example_pdf_parser.py
├── doc/                         # Documentation
│   └── AI模块算子说明.md       # Chinese documentation
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

#### Text Embedding
- `TextEmbedderDescriptor`: Serializable embedder configuration
- `TextEmbedder`: Main embedding class with batching logic

#### PDF Parsing
- `PDFParserDescriptor`: Serializable parser configuration
- `PDFParser`: Main PDF parsing class with async job handling
- `BatchParseResult`: Result of batch parsing operation
- `FileParseResult`: Result of single file parsing

### Enums

- `RetryStrategy`: NO_RETRY, EXPONENTIAL_BACKOFF_LIMITED, EXPONENTIAL_BACKOFF_UNLIMITED
- `ErrorHandlingStrategy`: FAIL_FAST, ZERO_VECTOR_FALLBACK

### Utility Functions

- `register_custom_model()`: Register new model profiles
- `get_model_profile()`: Retrieve model configurations
- `chunk_text()`: Split text into chunks
- `sanitize_dict()`: Sanitize sensitive data for logging
- `calculate_delay()`: Calculate exponential backoff delay
- `should_retry()`: Determine if retry should be attempted

## Examples

See the `examples/` directory for complete working examples:

- `example_text_embedder.py`: Complete example using text embedding
- `example_pdf_parser.py`: Complete example using PDF parsing with AIWorks provider

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

