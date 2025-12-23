# 实验3 v3 - 优化版实验设计（并行处理）

## 改进点

### 1. 简化总结数据结构
- **之前（v2）**: 返回复杂的JSON结构（summary + key_points + filtered_noise + structured_query等）
- **现在（v3）**: 只返回简单的summary文本
- **优势**: 更快的响应速度，减少LLM生成时间

### 2. 使用LLM进行所有评估
- **之前**: 使用硬编码的规则评估（字符串匹配、包含检查等）
- **现在**: 所有评估指标都交给LLM打分
- **评估维度**:
  - 信息保留率 (0-100分)
  - 噪音过滤率 (0-100分)
  - RAG相关性 (0-100分)
  - 回复质量 (0-100分)
  - 简洁度 (0-100分)

### 3. 模拟真实延迟
- 按每秒3个字的语速模拟用户说话时间
- 渐进式总结在用户说话的同时进行处理
- **关键优势**: 总结时间隐藏在用户输入过程中

### 4. 使用合适的模型
- **总结模型**: qwen3-8b（快速，用于实时总结）
- **回复模型**: qwen3-14b（高质量，性价比最优）
- **评估模型**: qwen3-14b（准确评分）

### 5. 流式输出
- 最终回复使用流式生成
- 记录关键延迟指标:
  - **TTFT** (Time To First Token): 首token延迟
  - **Generation Time**: 完整生成时间
  - **Tokens Per Second**: 输出速度

### 6. 评估时间从用户输入完成开始计算
- **方法1 (Baseline)**: total_time = RAG + 生成时间
- **方法2 (Batch Summary)**: total_time = 总结 + RAG + 生成时间
- **方法3 (Incremental)**:
  - total_time_after_input = RAG + 生成时间（用户感知延迟）
  - summary_processing_time = 总结时间（在用户输入过程中完成）

### 7. ⭐ 并行处理 - 新增！
- **模拟真实场景**: 用户只说一次，三个agent同时处理
- **并行执行**: 使用ThreadPoolExecutor并行运行三个方法
- **大幅提速**: 相比串行执行，总时间约等于最慢方法的时间
- **实时对比**: 可以看到哪个方法先完成响应

## 实验对比

### 三种方法
1. **Baseline**: 直接使用800字原文进行RAG
2. **Batch Summary**: 等用户说完后，一次性总结，然后RAG
3. **Incremental Summary**: 边听边总结，用户说完后直接RAG

### 关键指标
- **综合评分**: LLM对5个维度的综合打分
- **用户感知延迟**: 用户输入完成后的等待时间
- **首token延迟 (TTFT)**: 开始看到回复的时间
- **Query压缩比**: 总结相比原文的长度比例
- **输出速度**: tokens/秒

## 运行实验

```bash
# 运行实验
uv run python experiments/test_03_v3.py

# 分析结果
uv run python experiments/analyze_exp3_v3_results.py
```

## 预期结果

### 渐进式总结的优势
1. **用户感知延迟最低**: 总结时间隐藏在输入过程中
2. **压缩比最优**: 经过多次精炼，summary更简洁
3. **信息保留率高**: 逐步积累，不会遗漏关键信息
4. **噪音过滤效果好**: 每一步都在过滤，累积效果更好

### 权衡
- **总处理时间**: 可能比批量总结稍长（多次LLM调用）
- **适用场景**: 实时语音交互，用户边说边处理
- **不适合**: 离线批处理场景

## 文件结构

```
experiments/
├── test_03_v3.py                    # 主实验脚本（优化版）
├── incremental_summarizer_v2.py     # 简化的渐进式总结器
├── analyze_exp3_v3_results.py       # 结果分析脚本
└── long_audio_test_cases_v2.json    # 测试用例（5个，每个800字）
```

## 模型配置

```python
SUMMARY_MODEL = "qwen3-8b"     # 8B模型：快速总结
RESPONSE_MODEL = "qwen3-14b"   # 14B模型：高质量回复
EVAL_MODEL = "qwen3-14b"       # 14B模型：准确评估
```

## 评估示例

LLM评分输出示例：
```json
{
  "info_retention_score": 85,      // 信息保留率
  "noise_filtering_score": 90,     // 噪音过滤率
  "rag_relevance_score": 75,       // RAG相关性
  "response_quality_score": 88,    // 回复质量
  "conciseness_score": 80,         // 简洁度
  "total_score": 83.5,             // 综合得分
  "reasoning": "评分理由..."
}
```
