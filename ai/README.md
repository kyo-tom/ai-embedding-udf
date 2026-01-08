# Text Embedder 改进设计

## 概述

这是一个改进的 Text Embedder 实现，参考了 Daft 的设计但做了重要改进，特别是**明确区分了模型属性和 API 提供方属性**。

## 核心改进

### 1. 清晰的职责分离

```
┌─────────────────────────────────────────────────────┐
│  ModelProfile (模型固有属性)                        │
│  ├─ dimensions: 输出维度                           │
│  ├─ supports_overriding_dimensions: 是否支持自定义  │
│  └─ max_input_tokens: 单文本最大 tokens (模型限制) │
└─────────────────────────────────────────────────────┘
                      ↑
                      │ 模型元数据
                      │
┌─────────────────────────────────────────────────────┐
│  Provider (API 提供方属性)                          │
│  ├─ base_url: API 地址                             │
│  ├─ api_key: 认证密钥                              │
│  └─ max_batch_tokens: 批量请求最大 tokens (API限制)│
└─────────────────────────────────────────────────────┘
```

**关键区别：**
- `max_input_tokens` (模型限制): 由模型训练时决定，所有 Provider 共享
- `max_batch_tokens` (API 限制): 由 API 提供方决定，不同 Provider 可以不同

### 2. 架构对比

#### ❌ Daft 原设计的问题
```python
# 问题：批量限制硬编码在代码中
batch_token_limit = 300_000  # 所有 provider 都是这个值

# 无法为不同的 API 提供方设置不同的限制
```

#### ✅ 新设计的解决方案
```python
# Provider 可以指定自己的 API 限制
provider1 = OpenAIProvider(max_batch_tokens=300_000)  # OpenAI 官方
provider2 = OpenAIProvider(max_batch_tokens=100_000)  # 自定义服务器
provider3 = OpenAIProvider(max_batch_tokens=500_000)  # Azure
```

## 项目结构

```
prism/src/prism/ai/
├── __init__.py               # 主入口
├── typing.py                 # 类型定义
├── models.py                 # 模型配置和注册
├── providers/
│   ├── __init__.py
│   ├── base.py              # Provider 基类
│   └── openai_provider.py   # OpenAI Provider 实现
└── protocols/
    ├── __init__.py
    └── text_embedder.py     # TextEmbedder 实现

examples/
├── text_embedder_basic.py    # 基础使用示例
└── text_embedder_advanced.py # 高级功能示例

tests/
└── test_text_embedder.py     # 单元测试
```

## 快速开始

### 1. 基础用法

```python
from prism.ai import OpenAIProvider, register_custom_model

# 注册自定义模型
register_custom_model(
    model_name="conan-embedding-v1",
    dimensions=1792,
    max_input_tokens=8191,
)

# 创建 Provider
provider = OpenAIProvider(
    name="CustomAPI",
    base_url="http://172.16.10.163:9997/v1",
    api_key="empty",
    max_batch_tokens=100_000,  # 自定义服务器的批量限制
)

# 获取 embedder
embedder = provider.get_text_embedder(
    model="conan-embedding-v1"
).instantiate()

# 生成 embeddings
texts = ["你好", "世界"]
embeddings = embedder.embed_text(texts)
```

### 2. 处理超长文本

```python
# 超长文本会自动分块并合并
long_text = "重复内容 " * 10000  # 超过 8191 tokens

texts = [
    "正常文本",
    long_text,  # 自动分块
    "正常文本",
]

embeddings = embedder.embed_text(texts)
# 输出顺序保持一致：3 个 embeddings
```

### 3. 同一模型，不同 Provider

```python
register_custom_model("my-model", 1024)

# OpenAI 官方
provider1 = OpenAIProvider(
    api_key="sk-...",
    max_batch_tokens=300_000,
)

# 自定义服务器
provider2 = OpenAIProvider(
    base_url="http://localhost:9997/v1",
    max_batch_tokens=50_000,  # 不同的限制
)

# 两个 provider 使用同一个模型
embedder1 = provider1.get_text_embedder(model="my-model").instantiate()
embedder2 = provider2.get_text_embedder(model="my-model").instantiate()
```

## 核心特性

### 1. 智能批处理

```python
# 自动优化 API 调用
texts = ["短文本"] * 100 + ["超长文本" * 1000] + ["短文本"] * 50

embeddings = embedder.embed_text(texts)

# 内部流程：
# 1. 短文本累积到接近 max_batch_tokens
# 2. 批量发送（减少 API 调用）
# 3. 遇到超长文本，先 flush 累积的批次
# 4. 分块处理超长文本
# 5. 继续累积后续文本
```

### 2. 严格保持顺序

```python
# 即使有超长文本需要特殊处理，输出顺序也严格一致
texts = ["A", "B", "超长C", "D"]
embeddings = embedder.embed_text(texts)

# 输出: [emb_A, emb_B, emb_C, emb_D]
# 保证: texts[i] → embeddings[i]
```

### 3. 超长文本分块合并

```python
# 超长文本处理策略：
# 1. 分块: chunk_text(long_text, max_chars)
# 2. 每块生成 embedding
# 3. 加权平均: np.average(embeddings, weights=chunk_lens)
# 4. L2 归一化: vec / np.linalg.norm(vec)
```

## 运行示例

### 基础示例
```bash
python examples/text_embedder_basic.py
```

### 高级示例
```bash
python examples/text_embedder_advanced.py
```

### 运行测试
```bash
pytest tests/test_text_embedder.py -v
```

## 设计原则

### 1. 模型的归模型，API 的归 API

| 属性 | 归属 | 原因 |
|------|------|------|
| `dimensions` | ModelProfile | 模型输出维度是固定的 |
| `max_input_tokens` | ModelProfile | 模型上下文窗口是训练时决定的 |
| `supports_overriding_dimensions` | ModelProfile | 模型能力 |
| `max_batch_tokens` | Provider | API 提供方的限制 |
| `base_url` | Provider | API 地址 |
| `api_key` | Provider | 认证信息 |

### 2. Descriptor 模式

```python
# Descriptor: 可序列化配置
descriptor = provider.get_text_embedder(model="my-model")
# 可以跨进程传递

# Embedder: 不可序列化实例（包含 HTTP 客户端）
embedder = descriptor.instantiate()
# 在 worker 进程中创建
```

### 3. 为什么超长文本要先 flush？

```python
# ✓ 正确：先 flush
texts = ["A", "B", "超长C", "D"]

# 处理流程：
# 1. 累积 A, B
# 2. 遇到超长C，先 flush [A, B]
#    → embeddings = [emb_A, emb_B]
# 3. 分块处理 C
#    → embeddings = [emb_A, emb_B, emb_C]
# 4. 累积 D，最后 flush
#    → embeddings = [emb_A, emb_B, emb_C, emb_D]

# ❌ 错误：不先 flush
# 会导致输出顺序变成：[emb_C, emb_A, emb_B, emb_D]
```

## 与 Daft 设计对比

| 特性 | Daft 原设计 | 新设计 |
|------|------------|--------|
| `max_batch_tokens` | 硬编码在代码中 | Provider 参数化 |
| 自定义模型 | 需要修改全局字典 | `register_custom_model()` |
| 同一模型不同 API | 不支持 | 完全支持 |
| 职责分离 | 不清晰 | 模型/API 严格分离 |

## 许可证

MIT
