#!/bin/bash
# 停止本地模型服务

echo "=========================================="
echo "Stopping Local Model Services"
echo "=========================================="
echo ""

cd ~/tts

# 停止 vLLM
if [ -f logs/vllm.pid ]; then
    VLLM_PID=$(cat logs/vllm.pid)
    echo "🛑 Stopping vLLM (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null && echo "✓ vLLM stopped" || echo "⚠️  Process not found"
    rm logs/vllm.pid
else
    echo "⚠️  No vLLM PID file found"
fi

# 停止 Embedding
if [ -f logs/embedding.pid ]; then
    EMBED_PID=$(cat logs/embedding.pid)
    echo "🛑 Stopping Embedding (PID: $EMBED_PID)..."
    kill $EMBED_PID 2>/dev/null && echo "✓ Embedding stopped" || echo "⚠️  Process not found"
    rm logs/embedding.pid
else
    echo "⚠️  No Embedding PID file found"
fi

# 检查是否还有残留进程
echo ""
echo "Checking for remaining processes..."
pgrep -f "vllm.entrypoints" && pkill -f "vllm.entrypoints" && echo "✓ Killed remaining vLLM processes"
pgrep -f "infinity_emb" && pkill -f "infinity_emb" && echo "✓ Killed remaining embedding processes"

echo ""
echo "✅ All services stopped"
echo ""
