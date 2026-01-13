# AI Embedder & PDF Parser - AI 嵌入与解析库

[English](README.md) | 简体中文

一个灵活高效的 AI 库，支持文本嵌入和 PDF 解析，具有多提供商支持、智能批处理和健壮的错误处理功能。

## 特性

### 文本嵌入
- **多提供商支持**：无缝切换 OpenAI、AIWorks 等兼容提供商
- **智能批处理**：自动优化 API 调用，通过批量请求提高效率，同时遵守令牌限制
- **模型注册表**：集中式模型配置管理，支持自定义维度和令牌限制
- **分块支持**：通过智能分块和加权平均处理超大文本
- **重试策略**：可配置的重试机制，支持指数退避
- **错误处理**：多种错误处理策略（快速失败或零向量回退）

### PDF 解析
- **异步作业处理**：提交解析作业并轮询完成状态，支持可配置超时
- **批量处理**：解析多个 PDF 文件，支持部分成功
- **重试支持**：内置重试逻辑和指数退避，提高 API 可靠性
- **错误处理**：可配置错误处理（快速失败或继续处理）
- **MinIO 集成**：与 MinIO 对象存储直接集成

### 共享功能
- **类型安全**：完整的类型提示，提供更好的 IDE 支持和代码可靠性
- **可扩展**：轻松添加新的提供商和模型
- **安全日志**：自动 API 密钥脱敏的安全日志记录

## 安装

### 使用 uv（推荐）

```bash
# 安装基础依赖
uv pip install -e .

# 安装开发依赖
uv pip install -e ".[dev]"

# 安装可视化支持
uv pip install -e ".[viz]"
```

### 使用 pip

```bash
# 安装基础依赖
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"
```

## 快速开始

### 使用 AIWorks 提供商进行文本嵌入

```python
from ai.providers import AIWorksProvider

# 创建提供商
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.10.163:9997/v1",
    max_batch_tokens=100_000,
)

# 获取嵌入器描述符
descriptor = provider.get_text_embedder(model="conan-embedding-v1")

# 实例化嵌入器
embedder = descriptor.instantiate()

# 嵌入文本
texts = [
    "人工智能正在改变世界",
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络",
]

embeddings = embedder.embed_text(texts)
print(f"生成了 {len(embeddings)} 个嵌入向量")
print(f"维度: {len(embeddings[0])}")
```

### 使用 AIWorks 提供商进行 PDF 解析

```python
from ai.providers import AIWorksProvider
from ai.protocols import RetryStrategy, ErrorHandlingStrategy

# 创建提供商
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.99.68:8011",
    max_batch_tokens=100_000
)

# 获取 PDF 解析器描述符
descriptor = provider.get_pdf_parser(
    parser_type="mineru",
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=5,
    error_handling=ErrorHandlingStrategy.FAIL_FAST
)

# 实例化解析器
parser = descriptor.instantiate()

# 解析文件
files = [
    "/uni-parse-documents/documents/report1.pdf",
    "/uni-parse-documents/documents/report2.pdf",
]

result = parser.parse_files(
    files=files,
    source_parent_path="/uni-parse-documents/output"
)

print(f"成功: {result.success_count}, 失败: {result.failed_count}")
for path in result.successful:
    print(f"✓ {path}")
```

### 使用 OpenAI 提供商

```python
from ai.providers import OpenAIProvider

# 创建提供商
provider = OpenAIProvider(
    api_key="your-api-key-here",
    max_batch_tokens=300_000,
)

# 获取嵌入器（支持自定义维度）
descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    dimensions=512  # 可选：覆盖默认维度
)

embedder = descriptor.instantiate()
embeddings = embedder.embed_text(["你好，世界！"])
```

## 项目结构

```
ai-embedder/
├── ai/                          # 主包
│   ├── __init__.py
│   ├── models.py                # 模型注册表和配置
│   ├── typing.py                # 类型定义
│   ├── protocols/               # 协议实现
│   │   ├── __init__.py
│   │   ├── text_embedder.py    # TextEmbedder 实现
│   │   └── pdf_parser.py       # PDFParser 实现
│   ├── providers/               # 提供商实现
│   │   ├── __init__.py
│   │   ├── base.py             # 基础提供商接口
│   │   ├── aiworks_provider.py # AIWorks 提供商
│   │   └── openai_provider.py  # OpenAI 提供商
│   └── utils/                   # 工具函数
│       ├── retry_utils.py       # 重试策略和工具
│       ├── logging_utils.py     # 安全日志工具
│       └── embedding_viz.py     # 可视化工具
├── tests/                       # 测试套件
├── examples/                    # 示例脚本
│   ├── example_text_embedder.py
│   └── example_pdf_parser.py
├── doc/                         # 文档
│   └── AI模块算子说明.md       # 中文详细文档
├── pyproject.toml              # 项目配置
├── README.md                   # 英文文档
├── README.zh-CN.md            # 中文文档（本文件）
└── .gitignore                  # Git 忽略规则
```

## 架构设计

### 提供商-描述符-嵌入器模式

本库采用三层架构设计：

1. **提供商（Provider）**：存储 API 级别的配置（base_url、api_key、max_batch_tokens）
2. **描述符（Descriptor）**：可序列化的嵌入器配置
3. **嵌入器（Embedder）**：实际执行嵌入操作的实例

这种模式的优势：
- API 配置与模型配置清晰分离
- 支持分布式计算的可序列化配置
- 便于测试和模拟

### 智能批处理

嵌入器自动处理：
- **批次累积**：将文本分组以最大化 API 效率
- **令牌限制遵守**：确保批次不超过 API 限制
- **超大文本处理**：自动分块并合并大型文本
- **顺序保持**：严格保持输入输出的对应关系

## 配置说明

### 支持的模型

#### OpenAI 模型
- `text-embedding-ada-002`（1536 维）
- `text-embedding-3-small`（512/1536 维，可配置）
- `text-embedding-3-large`（256/1024/3072 维，可配置）

#### AIWorks 模型
- `conan-embedding-v1`（自定义维度）

### 添加自定义模型

```python
from ai.models import register_custom_model, EmbeddingDimensions

register_custom_model(
    model_name="my-custom-model",
    max_input_tokens=8192,
    dimensions=EmbeddingDimensions(size=768, dtype="float32"),
    supports_overriding_dimensions=False
)
```

## 开发指南

### 设置开发环境

```bash
# 安装开发依赖
uv pip install -e ".[dev]"
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=ai --cov-report=html

# 运行特定测试文件
pytest tests/test_basic.py
```

### 代码质量检查

```bash
# 格式化代码
black .

# 代码检查
ruff check .

# 类型检查
mypy ai/
```

## API 参考

### 提供商类

- `OpenAIProvider`：OpenAI 兼容 API 提供商
- `AIWorksProvider`：AIWorks API 提供商

### 核心类

#### 文本嵌入
- `TextEmbedderDescriptor`：可序列化的嵌入器配置
- `TextEmbedder`：主要的嵌入类，包含批处理逻辑

#### PDF 解析
- `PDFParserDescriptor`：可序列化的解析器配置
- `PDFParser`：主要的 PDF 解析类，带异步作业处理
- `BatchParseResult`：批量解析操作结果
- `FileParseResult`：单个文件解析结果

### 枚举

- `RetryStrategy`：NO_RETRY, EXPONENTIAL_BACKOFF_LIMITED, EXPONENTIAL_BACKOFF_UNLIMITED
- `ErrorHandlingStrategy`：FAIL_FAST, ZERO_VECTOR_FALLBACK

### 工具函数

- `register_custom_model()`：注册新模型配置
- `get_model_profile()`：获取模型配置
- `chunk_text()`：将文本分割成块
- `sanitize_dict()`：脱敏敏感数据用于日志记录
- `calculate_delay()`：计算指数退避延迟
- `should_retry()`：判断是否应该重试

## 示例

查看 `examples/` 目录获取完整的工作示例：

- `example_text_embedder.py`：文本嵌入完整示例
- `example_pdf_parser.py`：使用 AIWorks 提供商的 PDF 解析完整示例

## 系统要求

- Python >= 3.9
- numpy >= 1.20.0
- requests >= 2.25.0
- typing-extensions >= 4.0.0（Python < 3.11）

## 许可证

MIT License

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 致谢

本项目采用现代 Python 最佳实践构建，注重类型安全。
