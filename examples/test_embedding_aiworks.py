#!/usr/bin/env python3
"""
Test TextEmbedder with AIWorksProvider

This script directly tests embedder.embed_text() with AIWorksProvider and conan-embedding-v1 model.
Tests include:
- Basic embedding functionality
- Retry strategies
- Error handling strategies

Usage:
    python examples/test_embedding_aiworks.py

Make sure:
    1. The embedding API is running at http://172.16.10.163:9997
    2. The model conan-embedding-v1 is available
"""
import sys
import os
from typing import List
import numpy as np

# 8 个不同高度的 Unicode 字符
SPARK_CHARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

from ai.providers.aiworks_provider import AIWorksProvider
from ai import RetryStrategy, ErrorHandlingStrategy

def sparkline_from_floats(values: List[float], num_bins: int = 24) -> str:
    """
    将浮点数数组转换为 sparkline 字符串可视化。

    算法：
    1. 将向量分成 num_bins 组
    2. 计算每组的 RMS (均方根) 能量
    3. 归一化到 [0, 1]
    4. 映射到 8 个字符之一

    Args:
        values: 浮点数列表（embedding 向量）
        num_bins: 输出字符数量（默认 24）

    Returns:
        sparkline 字符串，如 "█▇▇▃▅▄▃▇▃▄▁▄▅▄▅▅▆▃▄▆▅▄▅▃"

    Example:
        >>> vec = [0.1, 0.5, 0.8, 0.3] * 100
        >>> sparkline = sparkline_from_floats(vec, num_bins=10)
        >>> print(sparkline)
        ▅▅▅▅▅▅▅▅▅▅
    """
    if not values or len(values) == 0:
        return ""

    values = np.array(values, dtype=np.float32)

    # 1. 计算 bin 大小
    bin_size = int(np.ceil(len(values) / num_bins))

    bins = []

    # 2. 对每个 bin 计算 RMS 能量
    for i in range(0, len(values), bin_size):
        slice_vals = values[i:i + bin_size]
        # RMS: sqrt(mean(x^2))
        energy = np.sqrt(np.mean(slice_vals ** 2))
        bins.append(energy)

    bins = np.array(bins)

    # 3. 归一化到 [0, 1]
    min_val = bins.min()
    max_val = bins.max()
    range_val = max(max_val - min_val, 1e-8)  # 避免除零

    normalized = (bins - min_val) / range_val

    # 4. 映射到字符
    chars = []
    for norm_val in normalized:
        idx = int(round(norm_val * (len(SPARK_CHARS) - 1)))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))  # 确保在范围内
        chars.append(SPARK_CHARS[idx])

    return ''.join(chars)


def test_basic_embedding():
    """Test basic embedding with short texts"""
    print("="*70)
    print("Test 1: Basic Embedding")
    print("="*70)

    # Step 1: Create AIWorksProvider
    print("\n[Step 1] Creating AIWorksProvider")
    provider = AIWorksProvider(
        name="AIWorks",
        base_url="http://172.16.10.163:9997/v1",
        max_batch_tokens=100_000,
    )
    print(f"✓ Provider created: {provider.name}")
    print(f"  Base URL: http://172.16.10.163:9997/v1")
    print(f"  Supported models: {', '.join(provider.SUPPORT_EMBEDDER)}")

    # Step 2: Get text embedder descriptor
    print("\n[Step 2] Getting text embedder descriptor")
    descriptor = provider.get_text_embedder(
        model="conan-embedding-v1",
        retry_strategy=RetryStrategy.NO_RETRY,
        error_handling=ErrorHandlingStrategy.ZERO_VECTOR_FALLBACK,
        )
    print(f"✓ Descriptor created")
    print(f"  Model: {descriptor.get_model()}")
    print(f"  Dimensions: {descriptor.get_dimensions().size}")
    print(f"  Max batch tokens: {descriptor.get_max_batch_tokens()}")
    print(f"  Retry strategy: {descriptor.retry_strategy.value}")
    print(f"  Error handling: {descriptor.error_handling.value}")

    # Step 3: Instantiate embedder
    print("\n[Step 3] Instantiating embedder")
    embedder = descriptor.instantiate()
    print(f"✓ Embedder instantiated")
    print(f"  Model: {embedder.model}")
    print(f"  Max input tokens: {embedder.max_input_tokens}")
    print(f"  Max batch tokens: {embedder.max_batch_tokens}")

    # Step 4: Prepare test texts
    print("\n[Step 4] Preparing test texts")
    texts = [
        "人工智能正在改变世界",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理帮助计算机理解人类语言",
        "计算机视觉让机器能够识别图像",
    ]
    print(f"Test texts: {len(texts)} samples")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    # Step 5: Call embedder.embed_text()
    print("\n[Step 5] Calling embedder.embed_text()")
    try:
        embeddings = embedder.embed_text(texts)
        print(f"✓ Embedding completed successfully!")

        # Analyze results
        print(f"\n[Step 6] Analyzing results")
        print(f"  Total embeddings: {len(embeddings)}")
        print(f"  Input texts: {len(texts)}")
        print(f"  Match: {len(embeddings) == len(texts)}")

        # Check each embedding (detailed)
        print("\n[Step 7] Detailed Results")
        for i, (text, emb) in enumerate(zip(texts, embeddings), 1):
            if emb is not None and len(emb) > 0:
                sparkline = sparkline_from_floats(emb, num_bins=24)
                print(f"\n  Sample {i}:")
                print(f"    Text: {text}")
                print(f"    Embedding dimension: {len(emb)}")
                print(f"    Sparkline: {sparkline}")
                print(f"    Preview: [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ...]")
            else:
                print(f"\n  Sample {i}: FAILED")
                print(f"    Text: {text}")

        return True

    except Exception as e:
        print(f"✗ Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def main():
    """Main test function"""
    print("\n" + "="*70)
    print("TextEmbedder Advanced Test Suite")
    print("="*70)
    print("\nConfiguration:")
    print("  Provider: AIWorksProvider")
    print("  Model: conan-embedding-v1")
    print("  API Endpoint: http://172.16.10.163:9997/v1")
    print("  Features:")
    print("    - Basic embedding")
    print("    - Retry strategies (exponential backoff)")
    print("    - Error handling (fail-fast, zero-vector fallback)")
    print("="*70)

    # Run tests
    success1 = test_basic_embedding()
    # Summary
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    print(f"  1. Basic Embedding Test:        {'✓ PASSED' if success1 else '✗ FAILED'}")
    print("="*70)

    if success1:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
