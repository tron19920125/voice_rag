#!/bin/bash
# 启动本地模型服务 (Azure Server)
# 包括: LLM (vLLM), Embedding (infinity_emb)

set -e

echo "=========================================="
echo "Starting Local Model Services"
echo "=========================================="
echo ""

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
echo "✓ HF_ENDPOINT=$HF_ENDPOINT"

# 检查 GPU
echo ""
echo "📊 GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# 进入项目目录
cd ~/tts
source ~/miniconda3/bin/activate

echo ""
echo "=========================================="
echo "1. Starting LLM Service (vLLM)"
echo "=========================================="
echo ""

# 安装 vLLM (如果还没安装)
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm -q
fi

# 启动 vLLM 服务 (使用 GPU 0,1 做 tensor parallelism)
echo "🚀 Starting vLLM server on GPU 0,1..."
echo "   Model: Qwen/Qwen3-8B (tensor-parallel-size=2)"
echo "   Port: 8000"
echo ""

# 使用 2 张 GPU 做 tensor parallelism，避免单卡 OOM
# V100 是 sm70，不支持 FlashInfer 和 Flash Attention
# 使用 TORCH_SDPA backend (PyTorch 原生，V100 兼容)
# vLLM 0.11.0 必须使用 V1 引擎
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_USE_FLASHINFER_SAMPLER=0

nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --disable-log-requests \
    --enforce-eager \
    --reasoning-parser qwen3 \
    > logs/vllm.log 2>&1 &

VLLM_PID=$!
echo "✓ vLLM started (PID: $VLLM_PID)"
echo "  Log: logs/vllm.log"

# 等待服务启动
echo ""
echo "⏳ Waiting for vLLM to be ready..."
sleep 30

# 测试 LLM
echo "🧪 Testing LLM service..."
curl -s http://localhost:8000/v1/models | python -m json.tool | head -20 || echo "⚠️  vLLM not ready yet"

echo ""
echo "=========================================="
echo "✅ LLM Service Started"
echo "=========================================="
echo ""
echo "Service Status:"
echo "  - vLLM (LLM):      http://localhost:8000  (PID: $VLLM_PID)"
echo ""
echo "Process IDs saved to:"
echo "  echo $VLLM_PID > logs/vllm.pid"
echo ""

# 保存 PID
mkdir -p logs
echo $VLLM_PID > logs/vllm.pid

echo "To stop services, run:"
echo "  ./scripts/stop_local_services.sh"
echo ""
echo "To check logs:"
echo "  tail -f logs/vllm.log"
echo ""
echo "Note: Embedding service is using cloud API (not deployed locally)"
echo ""
