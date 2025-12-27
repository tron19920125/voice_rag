# 语音客服系统使用指南

基于 Azure AI 和 Qwen 的完整语音客服 Demo，整合了 STT（语音识别）+ RAG（检索增强生成）+ LLM（大语言模型）+ TTS（语音合成）。

## 系统架构

```
用户语音输入
    ↓
[Azure STT + VAD] 语音识别
    ↓
[RAG检索] Azure AI Search（混合检索 + Reranking + QA优先级）
    ↓
[LLM生成] Qwen（流式调用 + 上下文管理）
    ↓
[Azure TTS] 语音合成（流式输出）
    ↓
扬声器播放
```

## 功能特性

### 1. 语音交互
- **Azure STT**: 实时语音识别，内置VAD自动检测语音开始/结束
- **Azure TTS**: 流式语音合成，支持语速调节
- **连续对话**: 支持多轮对话，自然流畅

### 2. 知识库RAG
- **混合检索**: BM25关键词 + 向量语义 + RRF融合
- **智能Reranking**: 使用bge-reranker-v2-m3精排
- **QA优先策略**: 高置信度QA（>0.85）直接返回答案
- **多格式支持**: PDF/Word/Excel文档 + JSON/CSV/Markdown QA数据

### 3. 智能对话
- **上下文管理**: 阈值触发压缩（4000 tokens），保留最近2轮
- **流式生成**: Qwen流式输出，降低延迟
- **异步压缩**: 后台压缩历史对话，不影响交互

## 快速开始

### 1. 环境要求
- Python 3.8+
- 麦克风和扬声器
- Azure订阅（Speech Service + AI Search）
- Qwen API密钥

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置环境变量

创建 `.env` 文件：

```bash
# Azure Speech Service
AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_REGION=eastus
AZURE_SPEECH_ENDPOINT=https://eastus.api.cognitive.microsoft.com/

# Azure AI Search
AZURE_SEARCH_KEY=your_key
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_INDEX_NAME=customer-service-kb

# Qwen LLM
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
QWEN_TOKEN=your_token

# Embedding Service
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_URL=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_TOKEN=your_token

# Reranking Service
RERANKING_MODEL=BAAI/bge-reranker-v2-m3
RERANKING_URL=https://api.siliconflow.cn/v1/rerank
RERANKING_TOKEN=your_token

# 上下文管理
CONTEXT_THRESHOLD=4000
CONTEXT_KEEP_RECENT_TURNS=2
```

### 4. 处理知识库

将文档放入 `data/documents/` 和 `data/qa/` 目录，然后运行：

```bash
./scripts/process_knowledge.sh
```

支持的格式：
- **文档**: PDF, Word (.docx), Excel (.xlsx)
- **QA数据**: JSON, JSONL, CSV, Excel, Markdown

### 5. 启动系统

```bash
./scripts/start_assistant.sh
```

或直接运行：

```bash
uv run python src/main.py
```

## 项目结构

```
tts/
├── src/
│   ├── config/          # 配置管理
│   │   └── settings.py
│   ├── speech/          # 语音服务
│   │   ├── stt_service.py
│   │   └── tts_service.py
│   ├── knowledge/       # 知识库
│   │   ├── document_processor.py
│   │   ├── indexer.py
│   │   └── rag_searcher.py
│   ├── llm/            # LLM服务
│   │   ├── qwen_service.py
│   │   └── context_manager.py
│   ├── pipeline/       # 流程编排
│   │   └── voice_assistant.py
│   └── main.py         # 主程序入口
├── scripts/
│   ├── process_knowledge.py    # 知识库处理
│   ├── process_knowledge.sh
│   ├── start_assistant.sh      # 启动脚本
│   └── test_rag.py            # RAG测试
├── data/
│   ├── documents/      # 文档数据
│   └── qa/            # QA数据
├── rag_utils.py       # RAG工具库
└── pyproject.toml
```

## 使用说明

### 对话控制
- **开始对话**: 直接说话，系统自动识别
- **结束对话**: 说"退出"、"再见"、"结束对话"或按 Ctrl+C

### 知识库管理

#### 添加文档
1. 将文件放入 `data/documents/` 或 `data/qa/`
2. 运行 `./scripts/process_knowledge.sh`

#### 更新索引
```bash
# 增量更新
./scripts/process_knowledge.sh

# 强制重建
./scripts/process_knowledge.sh --force
```

### 测试RAG
```bash
uv run python scripts/test_rag.py
```

## 技术细节

### RAG检索流程
1. **Embedding生成**: 使用bge-m3生成1024维向量
2. **混合检索**: Azure AI Search（关键词 + 向量 + 语义排序）
3. **召回**: Top-20候选文档
4. **Reranking**: 精排并QA加权（×1.5）
5. **结果**: Top-5最终文档或直接答案

### 上下文管理
- **Token计数**: 中文1字≈2tokens，英文1词≈1.3tokens
- **压缩触发**: 超过4000 tokens
- **压缩策略**: LLM总结旧对话 + 保留最近2轮
- **异步执行**: 不阻塞主流程

### 性能优化
- 流式生成：边生成边播放，降低首字延迟
- 异步压缩：后台处理，不影响交互
- 混合检索：BM25补充语义检索的精确匹配

## 故障排除

### 麦克风无法使用
- macOS: 检查系统设置 → 隐私与安全性 → 麦克风权限
- 确认麦克风已连接并设为默认设备

### Azure服务错误
- 检查 `.env` 中的密钥和端点
- 确认Azure服务已启用并有配额

### 知识库检索无结果
- 确认已运行 `process_knowledge.sh`
- 检查Azure AI Search索引是否创建成功
- 验证Embedding和Reranking服务可用

## 开发说明

### 添加新的文档格式
在 `src/knowledge/document_processor.py` 中扩展：

```python
class DocumentProcessor:
    def _extract_xxx(self, file_path: str) -> str:
        # 实现新格式的文本提取
        pass
```

### 自定义RAG策略
修改 `src/knowledge/rag_searcher.py` 中的参数：

```python
RAGSearcher(
    top_k=20,              # 召回数量
    rerank_top_k=5,        # 精排数量
    qa_weight_boost=1.5,   # QA加权系数
    qa_direct_threshold=0.85  # 直接回答阈值
)
```

### 调整LLM参数
在 `src/main.py` 或 `.env` 中配置：

```python
QwenService(
    temperature=0.7,  # 温度（0-1）
    # ... 其他参数
)
```

## License

MIT License
