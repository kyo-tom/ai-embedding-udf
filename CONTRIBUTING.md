# Contributing to AI Embedder

[English](#english) | [中文](#中文)

---

## English

Thank you for your interest in contributing to AI Embedder!

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/ai-embedder.git
cd ai-embedder

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

### Code Quality

Before submitting a PR, please ensure:

- All tests pass: `pytest`
- Code is formatted: `black .`
- Code passes linting: `ruff check .`
- Type checking passes: `mypy ai/`

### Documentation

When updating documentation:
- Update both `README.md` (English) and `README.zh-CN.md` (Chinese)
- Keep both versions synchronized in content

---

## 中文

感谢您对 AI Embedder 项目的贡献兴趣！

### 如何贡献

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

### 开发环境设置

```bash
# 克隆您的 fork
git clone https://github.com/your-username/ai-embedder.git
cd ai-embedder

# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
pytest
```

### 代码质量

在提交 PR 之前，请确保：

- 所有测试通过：`pytest`
- 代码已格式化：`black .`
- 代码通过检查：`ruff check .`
- 类型检查通过：`mypy ai/`

### 文档

更新文档时：
- 同时更新 `README.md`（英文）和 `README.zh-CN.md`（中文）
- 保持两个版本的内容同步
