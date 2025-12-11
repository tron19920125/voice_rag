# Scripts 工具目录

这个目录包含项目的验证和测试工具脚本。

## 📝 脚本列表

### 1. test_api_tokens.py
验证所有 API token 是否可用。

**功能**：
- 测试 Qwen LLM API
- 测试 Embedding API
- 测试 Reranking API

**运行**：
```bash
./scripts/run_clean.sh uv run python scripts/test_api_tokens.py
```

### 2. verify_infrastructure.py
验证 RAG 基础设施的所有组件。

**功能**：
- 测试模块导入
- 测试 Embedding 服务
- 测试向量索引
- 测试 Reranking 服务
- 测试完整 RAG 流程

**运行**：
```bash
./scripts/run_clean.sh uv run python scripts/verify_infrastructure.py
```

### 3. run_clean.sh
环境清理包装脚本，清除可能冲突的环境变量。

**用途**：确保脚本从 `.env` 文件读取最新配置，而不是使用 shell 中缓存的旧环境变量。

**使用**：
```bash
./scripts/run_clean.sh <your_command>
```

## ⚠️ 重要提示

由于 shell 可能缓存环境变量，建议**始终使用 `run_clean.sh`** 来运行脚本：

```bash
# ✅ 推荐
./scripts/run_clean.sh uv run python scripts/test_api_tokens.py

# ❌ 不推荐（可能使用旧的环境变量）
uv run python scripts/test_api_tokens.py
```

## 🔍 故障排除

### 问题：Reranking API 404 错误

**症状**：
```
404 Client Error: Not Found for url: https://api.siliconflow.cn/v1/rerankings
```

**原因**：Shell 环境变量缓存了旧的 URL (`/rerankings` 而不是 `/rerank`)

**解决方案**：
```bash
# 使用 run_clean.sh 清理环境
./scripts/run_clean.sh uv run python scripts/verify_infrastructure.py

# 或者手动清理
unset RERANKING_URL && uv run python scripts/verify_infrastructure.py
```

## 📊 预期输出

### test_api_tokens.py
```
通过率: 3/3
🎉 所有 API 测试通过！
```

### verify_infrastructure.py
```
通过率: 5/5
🎉 基础设施搭建完成！所有组件工作正常。
```
