# 本地vLLM模式使用指南

本项目支持使用本地vLLM服务替代远程API，实现完全本地化部署。

## 架构说明

本项目使用双模型架构：
- **Qwen3-8B** (端口8000): 用于轻量级任务（RAG需求判断、输入完整性判断）
- **Qwen3-14B** (端口8001): 用于主对话生成

## 快速启动

### 方式1：一键启动（推荐）

```bash
# 自动检查vLLM服务并启动TTS程序
bash scripts/run_with_local_vllm.sh
```

该脚本会：
1. 检查vLLM服务是否运行
2. 如果未运行，自动在azure-a100上启动vLLM
3. 设置.env使用本地模型
4. 启动TTS语音助手
5. 退出时自动恢复.env配置

### 方式2：手动启动

#### 步骤1：在azure-a100上启动vLLM服务

```bash
# SSH到服务器
ssh azure-a100

# 上传并执行启动脚本
cd ~/tts
bash scripts/start_dual_vllm_services.sh
```

等待服务启动（约1-2分钟），确认两个服务都已运行：
```bash
# 检查8B服务
curl http://localhost:8000/v1/models

# 检查14B服务
curl http://localhost:8001/v1/models
```

#### 步骤2：配置.env

编辑 `.env` 文件，修改以下配置：
```bash
USE_LOCAL_VLLM=true
LOCAL_VLLM_8B_BASE=http://localhost:8000/v1
LOCAL_VLLM_14B_BASE=http://localhost:8001/v1
```

#### 步骤3：启动TTS程序

```bash
uv run python src/main.py
```

## 停止服务

### 停止TTS程序
按 `Ctrl+C` 或说 "退出"、"再见"

### 停止vLLM服务

在azure-a100上执行：
```bash
# 停止所有vLLM进程
pkill -f "vllm.entrypoints.openai.api_server"
```

## 配置说明

### .env配置项

```bash
# 是否使用本地vLLM（true/false）
USE_LOCAL_VLLM=false

# Qwen3-8B本地服务地址（用于辅助任务）
LOCAL_VLLM_8B_BASE=http://localhost:8000/v1

# Qwen3-14B本地服务地址（用于主对话）
LOCAL_VLLM_14B_BASE=http://localhost:8001/v1

# 远程API配置（USE_LOCAL_VLLM=false时使用）
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-14b
QWEN_SUB_MODEL=qwen3-8b
QWEN_TOKEN=your_token
```

### vLLM服务参数

位于 `scripts/start_dual_vllm_services.sh`：

- **GPU分配**:
  - 8B模型使用GPU 0 (`CUDA_VISIBLE_DEVICES=0`)
  - 14B模型使用GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
- **显存占用**: `--gpu-memory-utilization 0.85`（85%）
- **最大长度**: `--max-model-len 16384`（16K tokens）
- **并行**: `--tensor-parallel-size 1`（单GPU）

## 日志查看

### vLLM服务日志

在azure-a100上：
```bash
# 8B模型日志
tail -f ~/tts/logs/vllm_8b.log

# 14B模型日志
tail -f ~/tts/logs/vllm_14b.log
```

### TTS程序日志

程序输出到终端，或根据你的日志配置查看。

## 故障排查

### vLLM服务启动失败

1. **检查端口占用**:
   ```bash
   lsof -i:8000
   lsof -i:8001
   ```

2. **检查GPU可用性**:
   ```bash
   nvidia-smi
   ```

3. **查看详细日志**:
   ```bash
   tail -100 ~/tts/logs/vllm_8b.log
   ```

### 连接失败

1. **确认vLLM服务运行**:
   ```bash
   curl http://localhost:8000/v1/models
   curl http://localhost:8001/v1/models
   ```

2. **检查防火墙/网络**:
   - 确保程序和vLLM在同一台机器上（都在azure-a100）
   - 或配置正确的远程地址和端口转发

### 性能问题

- 如果响应慢，检查GPU显存占用：`nvidia-smi`
- 可调整 `--gpu-memory-utilization` 参数（0.7-0.95）
- 可减少 `--max-model-len` 以降低显存占用

## 切换回远程API

修改 `.env`:
```bash
USE_LOCAL_VLLM=false
```

然后重启TTS程序即可。

## 性能对比

| 模式 | 延迟 | 成本 | 隐私 |
|------|------|------|------|
| 本地vLLM | 低（本地推理） | 仅GPU电费 | 完全本地 |
| 远程API | 中（网络延迟） | 按token计费 | 数据上传云端 |

选择本地模式适合：
- 需要低延迟的实时对话
- 数据隐私要求高
- 有GPU资源可用
- 高频使用降低成本
