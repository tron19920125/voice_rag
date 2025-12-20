# Experiment 3: 本地模型部署性能对比

## 目标

在真实服务器环境（Azure V100）上部署本地 LLM，并与云端 API 进行性能对比。

## 实验配置

### 本地部署
- **硬件**: Azure VM with 2x Tesla V100-PCIE-16GB (16GB VRAM each)
- **推理引擎**: vLLM 0.11.0
- **模型**: Qwen/Qwen3-8B
- **并行方式**: Tensor Parallelism (2 GPUs)
- **Attention Backend**: TORCH_SDPA (PyTorch native, V100 compatible)
- **思考模式**: 禁用 (`chat_template_kwargs: {enable_thinking: false}`)

### 云端 API
- **服务**: 通义千问 API
- **模型**: qwen3-8b
- **思考模式**: 禁用 (`extra_body: {enable_thinking: false}`)

### Embedding & Reranking
- **Embedding**: 云端 API (BAAI/bge-m3)
- **Reranking**: 云端 API (BAAI/bge-reranker-v2-m3)
- **原因**: vLLM 不支持 embedding 模型，本地部署需要额外推理引擎

## 关键技术挑战与解决方案

### 1. V100 GPU 兼容性
**问题**: V100 是 sm70 架构，不支持 FlashInfer 和 Flash Attention 2
- FlashInfer: 需要 sm75+ (Turing/Ampere)
- Flash Attention 2: 需要 sm80+ (Ampere)

**解决方案**:
```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### 2. 显存 OOM (Out of Memory)
**问题**: Qwen3-8B 单卡加载 ~15.3GB，超过 V100 16GB 限制

**解决方案**:
```bash
--tensor-parallel-size 2  # 使用 2 张 GPU 做模型并行
--gpu-memory-utilization 0.90
--max-model-len 16384
--enforce-eager  # 禁用 torch.compile 减少内存
```

### 3. 思考模式 (Thinking Mode)
**问题**: Qwen3-8B 默认输出 `<think>` 标签，增加延迟和 token 消耗

**解决方案**:
- 云端 API: `extra_body={"enable_thinking": False}`
- 本地 vLLM: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- vLLM 启动参数: `--reasoning-parser qwen3`

### 4. V1 引擎兼容性
**问题**: vLLM 0.11.0 必须使用 V1 引擎，V0 已被移除

**解决方案**: 不设置 `VLLM_USE_V1=0`，使用默认的 V1 引擎

## 性能结果

### 最终测试结果 (思考模式禁用后)

| 指标 | 云端 API | 本地 vLLM (V100 x2) | 云端优势 |
|------|---------|-------------------|---------|
| **平均延迟** | 1.023s | 2.960s | 2.9x 更快 |
| **平均 Tokens** | 81 | 77 | 本地更少 |
| **首次推理** | 1.52s | 6.63s | 4.4x 更快 |
| **后续推理** | 0.41-1.13s | 0.38-1.87s | 相近 |

### 延迟分析

**本地 vLLM 延迟组成**:
1. 首次推理较慢 (6.63s) - KV cache 预热
2. 后续推理快得多 (0.38-1.87s) - 缓存命中
3. 平均延迟 2.96s

**云端 API 延迟**:
- 更稳定，1.0s 左右
- 包含网络往返时间

## 运行实验

### 1. 启动本地 vLLM 服务 (在服务器上)

```bash
cd ~/tts
bash scripts/start_local_services.sh
```

### 2. 运行性能对比测试

```bash
cd ~/tts
source ~/miniconda3/bin/activate
python3 experiments/test_03_simple.py
```

### 3. 查看结果

结果保存在 `outputs/experiment3_simple_*.json`

### 4. 停止服务

```bash
cd ~/tts
bash scripts/stop_local_services.sh
```

## 成本与性能权衡

### 本地部署优势
✅ **数据隐私**: 敏感数据不离开服务器
✅ **成本可控**: 大规模使用时成本更低
✅ **无网络依赖**: 内网环境可用
✅ **可定制**: 可以调整参数优化性能

### 本地部署劣势
❌ **初始成本高**: 需要 GPU 服务器投入
❌ **延迟较高**: V100 性能不如新一代 GPU
❌ **运维复杂**: 需要管理服务和模型更新
❌ **首次推理慢**: KV cache 预热需要时间

### 云端 API 优势
✅ **延迟低**: 专用优化的推理集群
✅ **零运维**: 无需管理基础设施
✅ **按需付费**: 灵活的成本结构
✅ **持续优化**: 自动获得性能改进

### 云端 API 劣势
❌ **数据隐私**: 需要传输数据到云端
❌ **成本不可控**: 大规模使用成本高
❌ **网络依赖**: 需要稳定的互联网连接
❌ **定制受限**: 无法修改底层配置

## 结论

1. **本地部署可行**: 在 V100 上成功部署 Qwen3-8B，延迟约 3s
2. **云端更快**: 云端 API 延迟约 1s，比本地快 2.9x
3. **V100 限制**: 老一代 GPU 性能受限，新 GPU (A100/H100) 会有更好表现
4. **思考模式**: 必须禁用才能获得合理性能
5. **选择建议**:
   - **原型阶段**: 推荐云端 API（快速迭代）
   - **生产阶段（大规模）**: 考虑本地部署（成本优势）
   - **隐私敏感**: 必须本地部署
   - **性能优先**: 云端 API 或升级到新 GPU

## 文件说明

- `test_03_simple.py`: 性能对比测试脚本
- `scripts/start_local_services.sh`: vLLM 启动脚本
- `scripts/stop_local_services.sh`: vLLM 停止脚本
- `outputs/experiment3_simple_*.json`: 测试结果
