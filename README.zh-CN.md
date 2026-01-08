# AI Embedder - 文本嵌入库

[English](README.md) | 简体中文

一个灵活高效的文本嵌入库，支持多种 AI 提供商，具有智能批处理和模型管理功能。

## 特性

- **多提供商支持**：无缝切换 OpenAI、AIWorks 等兼容提供商
- **智能批处理**：自动优化 API 调用，通过批量请求提高效率，同时遵守令牌限制
- **模型注册表**：集中式模型配置管理，支持自定义维度和令牌限制
- **分块支持**：通过智能分块和加权平均处理超大文本
- **类型安全**：完整的类型提示，提供更好的 IDE 支持和代码可靠性
- **可扩展**：轻松添加新的提供商和模型

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

### 使用 AIWorks 提供商

```python
from ai.providers.aiworks_provider import AIWorksProvider

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

### 使用 OpenAI 提供商

```python
from ai.providers.openai_provider import OpenAIProvider

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
│   │   └── text_embedder.py    # TextEmbedder 实现
│   ├── providers/               # 提供商实现
│   │   ├── __init__.py
│   │   ├── base.py             # 基础提供商接口
│   │   ├── aiworks_provider.py # AIWorks 提供商
│   │   └── openai_provider.py  # OpenAI 提供商
│   └── utils/                   # 工具函数
│       └── embedding_viz.py     # 可视化工具
├── tests/                       # 测试套件
├── examples/                    # 示例脚本
│   └── test_embedding_aiworks.py
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

- `TextEmbedderDescriptor`：可序列化的嵌入器配置
- `TextEmbedder`：主要的嵌入类，包含批处理逻辑

### 工具函数

- `register_custom_model()`：注册新模型配置
- `get_model_profile()`：获取模型配置
- `chunk_text()`：将文本分割成块

## 示例

查看 `examples/` 目录获取完整的工作示例：

- `test_embedding_aiworks.py`：使用 AIWorks 提供商的完整示例

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
