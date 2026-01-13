# AI 模块算子说明文档

## 目录

- [模块概述](#模块概述)
- [核心组件](#核心组件)
- [Provider 算子](#provider-算子)
  - [OpenAIProvider](#openaiprovider)
  - [AIWorksProvider](#aiworksprovider)
- [TextEmbedder 算子](#textembedder-算子)
- [PDFParser 算子](#pdfparser-算子)
- [重试策略](#重试策略)
- [错误处理策略](#错误处理策略)
- [模型配置](#模型配置)
- [完整参数说明](#完整参数说明)
- [使用示例](#使用示例)

---

## 模块概述

AI 模块提供了统一的 AI 能力接口,具有清晰的关注点分离:

- **providers**: API 服务提供商 (OpenAI, Azure, 自定义端点)
- **protocols**: AI 协议接口 (TextEmbedder, PDFParser 等)
- **models**: 模型配置和配置文件
- **typing**: 类型定义

### 关键设计原则

1. 模型属性(维度、上下文窗口)与 API 属性(批次限制)分离
2. 描述符模式用于分布式环境中的序列化
3. 智能批处理以最小化 API 调用,同时遵守限制
4. 支持重试机制和错误处理策略

---

## 核心组件

### 1. Provider (提供商)

提供商代表托管 AI 模型的 API 端点/服务。不同的提供商可能具有不同的 API 限制、认证方法等。

**主要功能:**
- 管理 API 连接配置
- 创建 TextEmbedder 描述符
- 设置批次处理限制

### 2. TextEmbedderDescriptor (文本嵌入器描述符)

可序列化的配置对象,存储创建 TextEmbedder 实例所需的所有配置。

**设计模式:**
- 描述符存储配置(可序列化)
- `instantiate()` 方法创建实际的嵌入器(不可序列化,包含 API 客户端)

### 3. TextEmbedder (文本嵌入器)

具有智能批处理功能的文本嵌入器实现。

**核心特性:**
- 同时遵守模型和 API 限制
- 通过分块和加权平均处理超大输入
- 严格保持输出顺序与输入顺序匹配
- 通过智能批处理最小化 API 调用
- 支持重试和错误处理策略

### 4. RetryStrategy (重试策略)

定义 API 调用失败时的重试行为。

### 5. ErrorHandlingStrategy (错误处理策略)

定义重试次数耗尽后的错误处理行为。

---

## Provider 算子

### OpenAIProvider

**描述:** OpenAI (或兼容 OpenAI 的) API 提供商实现。

#### 构造函数参数

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `name` | `Optional[str]` | `"openai"` | 否 | 提供商名称 |
| `max_batch_tokens` | `int` | `300000` | 否 | 单个批次请求的最大 token 数量(API 限制) |
| `api_key` | `str` | - | 是 | OpenAI API 密钥 |
| `base_url` | `str` | `"https://api.openai.com/v1"` | 否 | API 基础 URL |
| `organization` | `str` | - | 否 | OpenAI 组织 ID |
| `timeout` | `int` | `30` | 否 | 请求超时时间(秒) |

#### 支持的模型

- `text-embedding-ada-002` (默认维度: 1536)
- `text-embedding-3-small` (默认维度: 1536)
- `text-embedding-3-large` (默认维度: 3072)

#### 方法

##### get_text_embedder()

创建文本嵌入器描述符。

**参数:**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model` | `Optional[str]` | `"text-embedding-3-small"` | 模型名称 |
| `dimensions` | `Optional[int]` | `None` | 自定义维度(如果模型支持) |
| `retry_strategy` | `Optional[RetryStrategy \| str]` | `"exponential_backoff_limited"` | 重试策略 |
| `max_retries` | `Optional[int]` | `3` | 最大重试次数(用于 LIMITED 策略) |
| `initial_delay` | `Optional[float]` | `1.0` | 指数退避的初始延迟(秒) |
| `max_delay` | `Optional[float]` | `60.0` | 指数退避的最大延迟(秒) |
| `exponential_base` | `Optional[float]` | `2.0` | 退避计算的指数基数 |
| `jitter` | `Optional[bool]` | `True` | 是否向延迟添加随机抖动 |
| `error_handling` | `Optional[ErrorHandlingStrategy \| str]` | `"fail_fast"` | 错误处理策略 |

**返回:** `TextEmbedderDescriptor`

**异常:**
- `ValueError`: 如果模型不被此提供商支持

#### 使用示例

```python
from ai.providers import OpenAIProvider

# 使用官方 OpenAI API
provider = OpenAIProvider(
    api_key="sk-...",
    max_batch_tokens=300_000
)

descriptor = provider.get_text_embedder(
    model="text-embedding-3-small"
)
embedder = descriptor.instantiate()
```

---

### AIWorksProvider

**描述:** AIWorks API 提供商实现,用于连接自定义的嵌入服务。

#### 构造函数参数

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `name` | `Optional[str]` | `"aiworks"` | 否 | 提供商名称 |
| `max_batch_tokens` | `int` | `10000` | 否 | 单个批次请求的最大 token 数量(API 限制) |
| `api_key` | `str` | - | 是 | API 密钥 |
| `base_url` | `str` | - | 是 | API 基础 URL |
| `organization` | `str` | - | 否 | 组织 ID |
| `timeout` | `int` | `30` | 否 | 请求超时时间(秒) |

#### 支持的模型

- `conan-embedding-v1` (默认维度: 1792)

#### 支持的解析器

- `mineru` (PDF 文档解析器)

#### 方法

##### get_text_embedder()

参数与 `OpenAIProvider.get_text_embedder()` 相同。

##### get_pdf_parser()

创建 PDF 解析器描述符。

**参数:**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `parser_type` | `Optional[str]` | `"mineru"` | 解析器类型 |
| `document_type` | `str` | `"pdf"` | 文档类型 |
| `parser_mode` | `str` | `"pipeline"` | 解析模式 |
| `poll_interval` | `int` | `2` | 状态轮询间隔(秒) |
| `poll_timeout` | `int` | `300` | 最大轮询超时时间(秒) |
| `custom_options` | `Optional[Dict[str, Any]]` | `{}` | 自定义解析选项 |
| `retry_strategy` | `Optional[RetryStrategy \| str]` | `"exponential_backoff_limited"` | 重试策略 |
| `max_retries` | `Optional[int]` | `3` | 最大重试次数 |
| `initial_delay` | `Optional[float]` | `1.0` | 初始延迟(秒) |
| `max_delay` | `Optional[float]` | `60.0` | 最大延迟(秒) |
| `exponential_base` | `Optional[float]` | `2.0` | 指数基数 |
| `jitter` | `Optional[bool]` | `True` | 是否添加随机抖动 |

**返回:** `PDFParserDescriptor`

**异常:**
- `ValueError`: 如果解析器类型不被此提供商支持

#### 使用示例

**文本嵌入:**

```python
from ai.providers import AIWorksProvider

# 使用自定义 AIWorks 端点
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.10.163:9997/v1",
    api_key="your-api-key",
    max_batch_tokens=100_000
)

descriptor = provider.get_text_embedder(
    model="conan-embedding-v1"
)
embedder = descriptor.instantiate()
embeddings = embedder.embed_text(["测试文本"])
```

**PDF 解析:**

```python
from ai.providers import AIWorksProvider
from ai.protocols import RetryStrategy

# 创建提供商
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.99.68:8011",
    max_batch_tokens=100_000
)

# 获取 PDF 解析器描述符
descriptor = provider.get_pdf_parser(
    parser_type="mineru",
    poll_timeout=600,  # 10 分钟超时
    custom_options={
        "language": "zh-CN",
        "enable_ocr": True,
    },
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=5
)

# 实例化解析器
parser = descriptor.instantiate()

# 解析文件
result = parser.parse_files(
    files=[
        "/uni-parse-documents/documents/file1.pdf",
        "/uni-parse-documents/documents/file2.pdf"
    ],
    source_parent_path="/uni-parse-documents/output"
)

print(f"成功: {result.success_count}, 失败: {result.failed_count}")
```

---

## TextEmbedder 算子

**描述:** 具有智能批处理功能的文本嵌入器。

### 构造函数参数

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `base_url` | `str` | - | 是 | API 基础 URL |
| `api_key` | `str` | - | 是 | API 密钥 |
| `model` | `str` | - | 是 | 模型名称 |
| `dimensions` | `Optional[int]` | `None` | 否 | 自定义嵌入维度 |
| `max_batch_tokens` | `int` | `300000` | 否 | API 批次限制 |
| `retry_strategy` | `RetryStrategy` | `EXPONENTIAL_BACKOFF_LIMITED` | 否 | 重试策略 |
| `max_retries` | `int` | `3` | 否 | 最大重试次数 |
| `initial_delay` | `float` | `1.0` | 否 | 初始延迟(秒) |
| `max_delay` | `float` | `60.0` | 否 | 最大延迟(秒) |
| `exponential_base` | `float` | `2.0` | 否 | 指数基数 |
| `jitter` | `bool` | `True` | 否 | 是否添加随机抖动 |
| `error_handling` | `ErrorHandlingStrategy` | `FAIL_FAST` | 否 | 错误处理策略 |

### 方法

#### embed_text(texts: List[str]) -> List[Embedding]

对文本列表进行嵌入,使用智能批处理。

**参数:**
- `texts` (`List[str]`): 要嵌入的文本字符串列表

**返回:**
- `List[Embedding]`: 嵌入向量列表(与输入顺序相同)

**算法流程:**
1. 按顺序处理文本,累积到批次中
2. 遇到超大文本(> max_input_tokens)时:
   - 首先刷新累积的批次(保持顺序)
   - 对超大文本进行分块
   - 嵌入分块并使用加权平均合并
3. 当批次达到 max_batch_tokens 时,刷新并开始新批次
4. 最后刷新剩余文本

**示例:**

```python
texts = ["人工智能", "机器学习", "深度学习"]
embeddings = embedder.embed_text(texts)

# embeddings 是一个列表,每个元素是一个浮点数列表
# 例如: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
```

#### _embed_text_batch(texts: List[str]) -> List[Embedding]

调用嵌入 API 处理一批文本,带有重试和错误处理。

**参数:**
- `texts` (`List[str]`): 要嵌入的文本列表

**返回:**
- `List[Embedding]`: 嵌入向量列表

**异常:**
- `requests.RequestException`: 如果 API 调用失败且 error_handling 为 FAIL_FAST

---

## PDFParser 算子

**描述:** PDF 文档解析器,支持异步任务轮询、重试机制和部分成功处理。

### 设计架构

PDFParser 遵循与 TextEmbedder 相同的三层架构:

```
1. Provider (存储 API 配置)
   └─> AIWorksProvider.get_pdf_parser()

2. Descriptor (可序列化配置)
   └─> PDFParserDescriptor

3. Parser (实际实例)
   └─> descriptor.instantiate() → PDFParser
```

### 构造函数参数

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `base_url` | `str` | - | 是 | 解析服务的 API 基础 URL |
| `document_type` | `str` | `"pdf"` | 否 | 文档类型 |
| `parser_type` | `str` | `"mineru"` | 否 | 解析器后端 |
| `parser_mode` | `str` | `"pipeline"` | 否 | 解析模式 |
| `poll_interval` | `int` | `2` | 否 | 状态轮询间隔(秒) |
| `poll_timeout` | `int` | `300` | 否 | 最大轮询超时时间(秒) |
| `custom_options` | `Optional[Dict[str, Any]]` | `{}` | 否 | 自定义解析选项 |
| `retry_strategy` | `RetryStrategy` | `EXPONENTIAL_BACKOFF_LIMITED` | 否 | 重试策略 |
| `max_retries` | `int` | `3` | 否 | 最大重试次数 |
| `initial_delay` | `float` | `1.0` | 否 | 初始延迟(秒) |
| `max_delay` | `float` | `60.0` | 否 | 最大延迟(秒) |
| `exponential_base` | `float` | `2.0` | 否 | 指数基数 |
| `jitter` | `bool` | `True` | 否 | 是否添加随机抖动 |

### 方法

#### parse_file(source_path: str, output_path: str) -> str

解析单个 PDF 文件。

**参数:**
- `source_path` (`str`): MinIO 上的源 PDF 文件路径(不含 s3:// 前缀)
- `output_path` (`str`): 解析后的 Markdown 文件输出路径

**返回:**
- `str`: 解析后的 Markdown 文件路径

**异常:**
- `PDFParseError`: 如果解析失败

**流程:**
1. 提交解析任务到服务端
2. 如果 30 秒内完成,直接返回结果
3. 如果超时,获取 job_id 并开始轮询
4. 轮询直到任务完成或超时

**示例:**

```python
parser = descriptor.instantiate()

output_path = parser.parse_file(
    source_path="/uni-parse-documents/documents/report.pdf",
    output_path="/uni-parse-documents/output/report.md"
)

print(f"解析完成: {output_path}")
```

#### parse_files(files: List[str], source_parent_path: str) -> BatchParseResult

批量解析多个 PDF 文件,支持部分成功。

**参数:**
- `files` (`List[str]`): PDF 文件路径列表
- `source_parent_path` (`str`): 输出 Markdown 文件的父目录

**返回:**
- `BatchParseResult`: 批量解析结果对象,包含:
  - `successful` (`List[str]`): 成功解析的文件路径列表
  - `failed` (`List[FileParseResult]`): 失败的文件结果列表
  - `success_count` (`int`): 成功数量
  - `failed_count` (`int`): 失败数量
  - `total_count` (`int`): 总数量

**特性:**
- **部分成功支持**: 单个文件失败不会中断整个批次
- **自动路径处理**: 自动将文件后缀改为 .md
- **重试机制**: 每个文件独立重试

**示例:**

```python
parser = descriptor.instantiate()

result = parser.parse_files(
    files=[
        "/uni-parse-documents/documents/file1.pdf",
        "/uni-parse-documents/documents/file2.pdf",
        "/uni-parse-documents/documents/file3.pdf"
    ],
    source_parent_path="/uni-parse-documents/output"
)

print(f"成功: {result.success_count}/{result.total_count}")
print(f"失败: {result.failed_count}/{result.total_count}")

# 处理成功的文件
for path in result.successful:
    print(f"✓ {path}")

# 处理失败的文件
for failed in result.failed:
    print(f"✗ {failed.source_path}: {failed.error_message}")
```

### 重试范围

PDFParser 对以下 API 调用应用重试机制:

1. **提交解析任务** (`POST /api/v1/parse_from_oss`)
   - 网络错误
   - 5xx 服务器错误
   - 超时错误

2. **轮询任务状态** (`GET /api/v1/jobs/{job_id}`)
   - 网络错误
   - 5xx 服务器错误
   - 超时错误

### 数据类型

#### FileParseResult

单个文件的解析结果。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `source_path` | `str` | 源文件路径 |
| `output_path` | `Optional[str]` | 输出文件路径(失败时为 None) |
| `success` | `bool` | 是否成功 |
| `error_message` | `Optional[str]` | 错误信息(成功时为 None) |
| `job_id` | `Optional[str]` | 任务 ID |

#### BatchParseResult

批量解析结果。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `successful` | `List[str]` | 成功解析的输出路径列表 |
| `failed` | `List[FileParseResult]` | 失败的文件结果列表 |
| `success_count` | `int` | 成功数量(属性) |
| `failed_count` | `int` | 失败数量(属性) |
| `total_count` | `int` | 总数量(属性) |

### 使用建议

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| **标准解析** | `poll_timeout=300`, `max_retries=3` | 适合大多数场景 |
| **大文件解析** | `poll_timeout=600`, `max_retries=5` | 增加超时和重试次数 |
| **快速失败** | `retry_strategy=NO_RETRY` | 网络稳定环境,快速发现问题 |
| **关键文档** | `retry_strategy=UNLIMITED` | 确保解析成功 |
| **批量处理** | 使用 `parse_files()` | 容忍部分失败,高效处理 |

---

## 重试策略

### RetryStrategy 枚举

定义 API 调用失败时的重试行为。

#### 策略选项

| 策略名称 | 枚举值 | 说明 |
|----------|--------|------|
| `NO_RETRY` | `"no_retry"` | 不重试,失败立即返回 |
| `EXPONENTIAL_BACKOFF_LIMITED` | `"exponential_backoff_limited"` | 指数退避 + 有限次数重试(默认) |
| `EXPONENTIAL_BACKOFF_UNLIMITED` | `"exponential_backoff_unlimited"` | 指数退避 + 无限重试 |

#### 使用方法

**方式 1: 使用枚举**

```python
from ai import RetryStrategy

descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED
)
```

**方式 2: 使用字符串**

```python
descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    retry_strategy="exponential_backoff_limited"
)
```

#### 重试配置参数

当使用指数退避策略时,可以配置以下参数:

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_retries` | `int` | `3` | 最大重试次数(仅用于 LIMITED 策略) |
| `initial_delay` | `float` | `1.0` | 初始延迟秒数 |
| `max_delay` | `float` | `60.0` | 最大延迟秒数 |
| `exponential_base` | `float` | `2.0` | 指数基数(delay = initial_delay * base^attempt) |
| `jitter` | `bool` | `True` | 是否添加随机抖动(±25%)以避免惊群效应 |

**延迟计算公式:**

```
delay = min(initial_delay * (exponential_base ^ attempt), max_delay)

如果 jitter 为 True:
    jitter_range = delay * 0.25
    delay = delay + random.uniform(-jitter_range, jitter_range)
```

**示例:**

```python
from ai import RetryStrategy

descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=5,          # 最多重试 5 次
    initial_delay=2.0,      # 第一次重试等待 2 秒
    max_delay=30.0,         # 最长等待 30 秒
    exponential_base=2.0,   # 延迟翻倍: 2s, 4s, 8s, 16s, 30s
    jitter=True            # 添加随机抖动
)
```

---

## 错误处理策略

### ErrorHandlingStrategy 枚举

定义重试次数耗尽后的错误处理行为。

#### 策略选项

| 策略名称 | 枚举值 | 说明 |
|----------|--------|------|
| `FAIL_FAST` | `"fail_fast"` | 快速失败,抛出异常(默认) |
| `ZERO_VECTOR_FALLBACK` | `"zero_vector_fallback"` | 零向量回退,失败时返回零向量 |

#### 使用方法

**FAIL_FAST (默认)**

```python
from ai import ErrorHandlingStrategy

descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    error_handling=ErrorHandlingStrategy.FAIL_FAST
)

# 如果 API 调用失败,将抛出异常
embedder = descriptor.instantiate()
try:
    embeddings = embedder.embed_text(texts)
except Exception as e:
    print(f"嵌入失败: {e}")
```

**ZERO_VECTOR_FALLBACK**

```python
from ai import ErrorHandlingStrategy, RetryStrategy

descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    retry_strategy=RetryStrategy.NO_RETRY,  # 不重试
    error_handling=ErrorHandlingStrategy.ZERO_VECTOR_FALLBACK
)

# 即使 API 失败,也会返回零向量而不是抛出异常
embedder = descriptor.instantiate()
embeddings = embedder.embed_text(texts)
# 失败的文本将得到零向量: [0.0, 0.0, ..., 0.0]
```

#### 使用建议

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 生产环境 | `FAIL_FAST` | 及时发现问题,避免脏数据 |
| 批量处理(可容忍部分失败) | `ZERO_VECTOR_FALLBACK` | 不中断整个流程 |
| 开发调试 | `FAIL_FAST` + `NO_RETRY` | 快速失败,便于调试 |
| 关键业务 | `FAIL_FAST` + `EXPONENTIAL_BACKOFF_LIMITED` | 重试后仍失败则中断 |

---

## 模型配置

### 预定义模型

模块内置了以下模型配置:

#### OpenAI 模型

| 模型名称 | 默认维度 | 支持自定义维度 | 最大输入 Token |
|----------|---------|---------------|---------------|
| `text-embedding-ada-002` | 1536 | 否 | 8191 |
| `text-embedding-3-small` | 1536 | 是 (256-1536) | 8191 |
| `text-embedding-3-large` | 3072 | 是 (256-3072) | 8191 |

#### 自定义模型

| 模型名称 | 默认维度 | 支持自定义维度 | 最大输入 Token |
|----------|---------|---------------|---------------|
| `conan-embedding-v1` | 1792 | 否 | 8191 |

### 注册自定义模型

使用 `register_custom_model()` 函数注册自定义模型:

**函数签名:**

```python
def register_custom_model(
    model_name: str,
    dimensions: int,
    supports_overriding_dimensions: bool = False,
    max_input_tokens: int = 8191,
) -> None
```

**参数说明:**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model_name` | `str` | - | 模型名称 |
| `dimensions` | `int` | - | 默认嵌入维度 |
| `supports_overriding_dimensions` | `bool` | `False` | 是否支持自定义维度 |
| `max_input_tokens` | `int` | `8191` | 最大输入 token 数量 |

**示例:**

```python
from ai import register_custom_model

register_custom_model(
    model_name="my-custom-model",
    dimensions=2048,
    supports_overriding_dimensions=True,
    max_input_tokens=4096
)
```

### 查询模型配置

使用 `get_model_profile()` 函数查询模型配置:

```python
from ai import get_model_profile

profile = get_model_profile("text-embedding-3-small")
print(f"维度: {profile.dimensions.size}")
print(f"支持自定义维度: {profile.supports_overriding_dimensions}")
print(f"最大输入 Token: {profile.max_input_tokens}")
```

---

## 完整参数说明

### Provider 层参数

这些参数在创建 Provider 时设置,代表 **API 层面的配置**:

| 参数名 | 作用域 | 类型 | 说明 |
|--------|--------|------|------|
| `name` | Provider | `str` | 提供商名称 |
| `api_key` | Provider | `str` | API 密钥 |
| `base_url` | Provider | `str` | API 基础 URL |
| `max_batch_tokens` | Provider | `int` | API 批次 token 限制 |
| `organization` | Provider | `str` | 组织 ID (可选) |
| `timeout` | Provider | `int` | 请求超时时间(秒) |

### Embedder 配置参数

这些参数在调用 `get_text_embedder()` 时设置,代表 **模型和策略配置**:

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model` | `str` | Provider 默认模型 | 模型名称 |
| `dimensions` | `int` | 模型默认维度 | 自定义维度(如果模型支持) |
| `retry_strategy` | `RetryStrategy \| str` | `EXPONENTIAL_BACKOFF_LIMITED` | 重试策略 |
| `max_retries` | `int` | `3` | 最大重试次数 |
| `initial_delay` | `float` | `1.0` | 初始延迟(秒) |
| `max_delay` | `float` | `60.0` | 最大延迟(秒) |
| `exponential_base` | `float` | `2.0` | 指数基数 |
| `jitter` | `bool` | `True` | 是否添加随机抖动 |
| `error_handling` | `ErrorHandlingStrategy \| str` | `FAIL_FAST` | 错误处理策略 |

### 模型属性

这些属性在模型注册时定义,代表 **模型固有属性**:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `dimensions` | `EmbeddingDimensions` | 默认嵌入维度 |
| `supports_overriding_dimensions` | `bool` | 是否支持自定义维度 |
| `max_input_tokens` | `int` | 最大输入 token 数量(上下文窗口) |

---

## 使用示例

### 示例 1: 基础用法

```python
from ai.providers.aiworks_provider import AIWorksProvider

# 创建提供商
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.10.163:9997/v1",
    api_key="your-api-key",
    max_batch_tokens=100_000
)

# 获取嵌入器描述符(使用默认配置)
descriptor = provider.get_text_embedder(model="conan-embedding-v1")

# 实例化嵌入器
embedder = descriptor.instantiate()

# 执行嵌入
texts = ["人工智能", "机器学习", "深度学习"]
embeddings = embedder.embed_text(texts)

print(f"嵌入了 {len(embeddings)} 个文本")
print(f"每个嵌入的维度: {len(embeddings[0])}")
```

### 示例 2: 自定义重试策略

```python
from ai import RetryStrategy
from ai.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="sk-...",
    max_batch_tokens=300_000
)

descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=5,
    initial_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

embedder = descriptor.instantiate()
embeddings = embedder.embed_text(["测试文本"])
```

### 示例 3: 零向量回退

```python
from ai import ErrorHandlingStrategy, RetryStrategy
from ai.providers.aiworks_provider import AIWorksProvider

provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.10.163:9997/v1",
    api_key="your-api-key",
    max_batch_tokens=100_000
)

# 配置零向量回退策略
descriptor = provider.get_text_embedder(
    model="conan-embedding-v1",
    retry_strategy=RetryStrategy.NO_RETRY,
    error_handling=ErrorHandlingStrategy.ZERO_VECTOR_FALLBACK
)

embedder = descriptor.instantiate()

# 即使 API 失败,也会返回零向量
texts = ["文本1", "文本2", "文本3"]
embeddings = embedder.embed_text(texts)

# 检查是否有零向量(表示失败)
for i, emb in enumerate(embeddings):
    if all(v == 0.0 for v in emb):
        print(f"文本 {i} 嵌入失败,返回零向量")
```

### 示例 4: 自定义维度

```python
from ai.providers import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")

# text-embedding-3-small 支持 256-1536 的自定义维度
descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    dimensions=512  # 自定义维度
)

embedder = descriptor.instantiate()
embeddings = embedder.embed_text(["测试"])

print(f"嵌入维度: {len(embeddings[0])}")  # 输出: 512
```

### 示例 5: 注册和使用自定义模型

```python
from ai import register_custom_model
from ai.providers.aiworks_provider import AIWorksProvider

# 注册自定义模型
register_custom_model(
    model_name="my-custom-embedding",
    dimensions=2048,
    supports_overriding_dimensions=False,
    max_input_tokens=4096
)

# 使用自定义模型
provider = AIWorksProvider(
    base_url="http://my-api-server.com/v1",
    api_key="my-key",
    max_batch_tokens=50_000
)

descriptor = provider.get_text_embedder(model="my-custom-embedding")
embedder = descriptor.instantiate()
embeddings = embedder.embed_text(["测试自定义模型"])
```

### 示例 6: 批量处理大量文本

```python
from ai.providers.aiworks_provider import AIWorksProvider
from ai import RetryStrategy, ErrorHandlingStrategy

provider = AIWorksProvider(
    base_url="http://172.16.10.163:9997/v1",
    api_key="your-api-key",
    max_batch_tokens=100_000
)

# 配置适合批量处理的策略
descriptor = provider.get_text_embedder(
    model="conan-embedding-v1",
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=3,
    error_handling=ErrorHandlingStrategy.ZERO_VECTOR_FALLBACK  # 容忍部分失败
)

embedder = descriptor.instantiate()

# 处理大量文本
large_text_list = [f"文档 {i}" for i in range(10000)]
embeddings = embedder.embed_text(large_text_list)

# 统计成功和失败
failed_count = sum(1 for emb in embeddings if all(v == 0.0 for v in emb))
print(f"成功: {len(embeddings) - failed_count}, 失败: {failed_count}")
```

### 示例 7: 处理超长文本

```python
from ai.providers.aiworks_provider import AIWorksProvider

provider = AIWorksProvider(
    base_url="http://172.16.10.163:9997/v1",
    api_key="your-api-key",
    max_batch_tokens=100_000
)

descriptor = provider.get_text_embedder(model="conan-embedding-v1")
embedder = descriptor.instantiate()

# 超长文本(超过 max_input_tokens)
very_long_text = "这是一段很长的文本..." * 10000

# TextEmbedder 会自动分块、嵌入、加权平均
texts = [very_long_text, "正常长度的文本"]
embeddings = embedder.embed_text(texts)

# 两个文本都能成功嵌入
print(f"成功嵌入 {len(embeddings)} 个文本")
```

### 示例 8: 生产环境配置

```python
from ai import RetryStrategy, ErrorHandlingStrategy
from ai.providers import OpenAIProvider
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# 生产环境推荐配置
provider = OpenAIProvider(
    api_key="sk-...",
    max_batch_tokens=300_000,
    timeout=60  # 增加超时时间
)

descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    # 重试策略:有限次数重试
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,  # 避免惊群效应
    # 错误处理:快速失败
    error_handling=ErrorHandlingStrategy.FAIL_FAST
)

embedder = descriptor.instantiate()

# 生产环境中应该捕获异常
try:
    embeddings = embedder.embed_text(["重要文本"])
    # 保存到数据库等后续处理...
except Exception as e:
    # 记录错误,发送告警等
    logging.error(f"嵌入失败: {e}")
    raise
```

### 示例 9: PDF 文档解析(基础用法)

```python
from ai.providers import AIWorksProvider
from ai.protocols import RetryStrategy

# 创建提供商
provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.99.68:8011",
    max_batch_tokens=100_000
)

# 获取 PDF 解析器描述符
descriptor = provider.get_pdf_parser(
    parser_type="mineru",
    poll_timeout=600,  # 10 分钟超时
    custom_options={
        "language": "zh-CN",
        "enable_ocr": True,
    },
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=5
)

# 实例化解析器
parser = descriptor.instantiate()

# 解析单个文件
output_path = parser.parse_file(
    source_path="/uni-parse-documents/documents/report.pdf",
    output_path="/uni-parse-documents/output/report.md"
)

print(f"解析完成: {output_path}")
```

### 示例 10: PDF 批量解析(部分成功处理)

```python
from ai.providers import AIWorksProvider
from ai.protocols import RetryStrategy

provider = AIWorksProvider(
    name="AIWorks",
    base_url="http://172.16.99.68:8011",
    max_batch_tokens=100_000
)

descriptor = provider.get_pdf_parser(
    parser_type="mineru",
    poll_timeout=600,
    custom_options={
        "language": "zh-CN",
        "enable_ocr": True,
    },
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF_LIMITED,
    max_retries=3
)

parser = descriptor.instantiate()

# 批量解析多个文件
files = [
    "/uni-parse-documents/documents/file1.pdf",
    "/uni-parse-documents/documents/file2.pdf",
    "/uni-parse-documents/documents/file3.pdf"
]

result = parser.parse_files(
    files=files,
    source_parent_path="/uni-parse-documents/output"
)

# 处理结果
print(f"\n解析完成!")
print(f"总数: {result.total_count}")
print(f"成功: {result.success_count}")
print(f"失败: {result.failed_count}")

# 输出成功的文件
if result.successful:
    print("\n✓ 成功解析的文件:")
    for path in result.successful:
        print(f"  - {path}")

# 输出失败的文件
if result.failed:
    print("\n✗ 解析失败的文件:")
    for failed in result.failed:
        print(f"  - {failed.source_path}")
        print(f"    错误: {failed.error_message}")
```

---

## 附录

### A. 术语表

| 术语 | 说明 |
|------|------|
| Provider | 提供商,代表 API 服务端点 |
| Embedder | 嵌入器,负责将文本转换为向量 |
| Descriptor | 描述符,可序列化的配置对象 |
| Token | 文本的最小单位,约 3-4 个字符 |
| Batch | 批次,一次 API 请求处理的文本集合 |
| Retry | 重试,API 失败后的重新尝试 |
| Backoff | 退避,重试之间的延迟时间 |
| Jitter | 抖动,延迟时间的随机扰动 |

### B. 性能优化建议

1. **合理设置 max_batch_tokens**
   - 根据 API 服务器的限制设置
   - 过大会导致请求失败
   - 过小会增加 API 调用次数

2. **选择合适的重试策略**
   - 生产环境:使用 `EXPONENTIAL_BACKOFF_LIMITED`
   - 开发调试:使用 `NO_RETRY`
   - 关键任务:增加 `max_retries` 和 `max_delay`

3. **批量处理优化**
   - 将相似长度的文本分组
   - 避免单个超长文本影响批次效率
   - 使用 `ZERO_VECTOR_FALLBACK` 容忍部分失败

4. **监控和日志**
   - 启用 INFO 级别日志查看批次信息
   - 监控重试次数和失败率
   - 记录零向量回退的情况

### C. 常见问题

#### Q1: 如何选择 Provider?

- 使用官方 OpenAI API → `OpenAIProvider`
- 使用自定义端点 → `AIWorksProvider` 或自定义 Provider

#### Q2: 什么时候使用零向量回退?

- 批量处理可以容忍部分失败时
- 非关键业务场景
- 注意:需要在后续处理中检测和过滤零向量

#### Q3: 如何处理超长文本?

`TextEmbedder` 会自动:
1. 检测超长文本
2. 分块处理
3. 加权平均合并
4. 归一化向量

无需手动处理。

#### Q4: 重试策略如何影响性能?

- `NO_RETRY`: 最快,但容错性差
- `EXPONENTIAL_BACKOFF_LIMITED`: 平衡性能和可靠性
- `EXPONENTIAL_BACKOFF_UNLIMITED`: 最可靠,但可能长时间阻塞

#### Q5: 如何自定义嵌入维度?

只有支持自定义维度的模型(如 `text-embedding-3-small`)才能自定义:

```python
descriptor = provider.get_text_embedder(
    model="text-embedding-3-small",
    dimensions=512  # 256-1536 之间
)
```

#### Q6: PDF 解析任务超时怎么办?

- 增加 `poll_timeout` 参数(默认 300 秒)
- 对于大文件,建议设置 600 秒以上
- 检查服务端解析性能

#### Q7: 如何处理 PDF 批量解析中的部分失败?

使用 `parse_files()` 方法会返回 `BatchParseResult`:

```python
result = parser.parse_files(files, output_dir)

# 成功的文件
for path in result.successful:
    process_success(path)

# 失败的文件
for failed in result.failed:
    log_error(failed.source_path, failed.error_message)
```

### D. 错误码说明

| 错误类型 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `ValueError` | 模型不支持 | 检查模型名称或注册自定义模型 |
| `requests.RequestException` | API 调用失败 | 检查网络、API 密钥、URL |
| `TypeError` | 参数类型错误 | 检查参数类型 |
| `KeyError` | 响应格式错误 | 检查 API 兼容性 |
| `PDFParseError` | PDF 解析失败 | 检查文件路径、服务状态、网络连接 |
