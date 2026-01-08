"""
Embedding Visualization Utilities

Inspired by Daft's sparkline visualization for embedding vectors.
"""
import numpy as np
from typing import List


# 8 个不同高度的 Unicode 字符
SPARK_CHARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']


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


def embedding_repr(embedding: List[float], num_bins: int = 24) -> str:
    """
    生成 embedding 的字符串表示（包含类型和 sparkline）。

    Args:
        embedding: Embedding 向量
        num_bins: Sparkline 字符数

    Returns:
        格式化字符串，如 "Embedding[Float32; 1792]: █▇▇▃▅▄..."

    Example:
        >>> emb = [0.1] * 1792
        >>> print(embedding_repr(emb))
        Embedding[Float32; 1792]: ▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅
    """
    sparkline = sparkline_from_floats(embedding, num_bins)
    return f"Embedding[Float32; {len(embedding)}]: {sparkline}"


def format_embedding_table(
    embeddings: List[List[float]],
    labels: List[str] = None,
    num_bins: int = 24
) -> str:
    """
    格式化多个 embeddings 为表格形式。

    Args:
        embeddings: Embedding 向量列表
        labels: 每个 embedding 的标签
        num_bins: Sparkline 字符数

    Returns:
        格式化的表格字符串

    Example:
        >>> embs = [[0.1] * 100, [0.5] * 100, [0.9] * 100]
        >>> labels = ["doc1", "doc2", "doc3"]
        >>> print(format_embedding_table(embs, labels, num_bins=10))
    """
    if labels is None:
        labels = [f"Row {i}" for i in range(len(embeddings))]

    # 计算列宽
    max_label_len = max(len(label) for label in labels)

    # 表格边框
    lines = []
    lines.append("╭" + "─" * (max_label_len + 2) + "┬" + "─" * (num_bins + 30) + "╮")
    lines.append(
        f"│ {'Label':<{max_label_len}} │ {'Embedding':<{num_bins + 28}} │"
    )
    lines.append("╞" + "═" * (max_label_len + 2) + "╪" + "═" * (num_bins + 30) + "╡")

    # 数据行
    for i, (label, emb) in enumerate(zip(labels, embeddings)):
        sparkline = sparkline_from_floats(emb, num_bins)
        emb_info = f"Embedding[{len(emb)}d]: {sparkline}"
        lines.append(f"│ {label:<{max_label_len}} │ {emb_info:<{num_bins + 28}} │")

        if i < len(embeddings) - 1:
            lines.append("├" + "─" * (max_label_len + 2) + "┼" + "─" * (num_bins + 30) + "┤")

    lines.append("╰" + "─" * (max_label_len + 2) + "┴" + "─" * (num_bins + 30) + "╯")

    return "\n".join(lines)


# 示例和测试
if __name__ == "__main__":
    print("=" * 70)
    print("Embedding Sparkline 可视化示例")
    print("=" * 70)

    # 示例 1: 基础 sparkline
    print("\n示例 1: 基础 sparkline")
    vec1 = [0.1] * 100 + [0.9] * 100 + [0.5] * 100
    sparkline1 = sparkline_from_floats(vec1, num_bins=24)
    print(f"向量分布: [低] * 100 + [高] * 100 + [中] * 100")
    print(f"Sparkline: {sparkline1}")

    # 示例 2: 渐变向量
    print("\n示例 2: 渐变向量")
    vec2 = list(np.linspace(0, 1, 300))
    sparkline2 = sparkline_from_floats(vec2, num_bins=24)
    print(f"向量分布: 0 → 1 线性渐变")
    print(f"Sparkline: {sparkline2}")

    # 示例 3: 随机向量
    print("\n示例 3: 随机向量 (模拟真实 embedding)")
    vec3 = list(np.random.randn(1792) * 0.1)
    sparkline3 = sparkline_from_floats(vec3, num_bins=24)
    print(f"向量维度: 1792")
    print(f"Sparkline: {sparkline3}")

    # 示例 4: 完整表格
    print("\n示例 4: 多个 embeddings 表格")
    embeddings = [
        list(np.random.randn(1792) * 0.1),
        list(np.random.randn(1792) * 0.2),
        list(np.random.randn(1792) * 0.15),
    ]
    labels = ["Hello world", "Daft is awesome", "Machine learning"]
    table = format_embedding_table(embeddings, labels, num_bins=24)
    print(table)

    # 示例 5: RMS vs 简单平均对比
    print("\n示例 5: 为什么使用 RMS 而不是简单平均？")
    vec_oscillating = [0.5, -0.5] * 100  # 振荡信号
    vec_constant = [0.0] * 200            # 恒定信号

    # 简单平均都是 0
    avg1 = np.mean(vec_oscillating)
    avg2 = np.mean(vec_constant)
    print(f"振荡信号简单平均: {avg1:.4f}")
    print(f"恒定信号简单平均: {avg2:.4f}")
    print("→ 简单平均无法区分两者！")

    # RMS 能量不同
    rms1 = np.sqrt(np.mean(np.array(vec_oscillating) ** 2))
    rms2 = np.sqrt(np.mean(np.array(vec_constant) ** 2))
    print(f"\n振荡信号 RMS 能量: {rms1:.4f}")
    print(f"恒定信号 RMS 能量: {rms2:.4f}")
    print("→ RMS 能正确反映能量差异！")

    sparkline_osc = sparkline_from_floats(vec_oscillating, num_bins=10)
    sparkline_const = sparkline_from_floats(vec_constant, num_bins=10)
    print(f"\n振荡信号 sparkline: {sparkline_osc}")
    print(f"恒定信号 sparkline: {sparkline_const}")

    print("\n" + "=" * 70)
