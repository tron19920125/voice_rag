#!/bin/bash
# 运行实验3 v3 - 服务器版本
# 测试4种长文本RAG方法对比

set -e

echo "=========================================="
echo "Experiment 3 v3 - Server Version"
echo "=========================================="
echo ""

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 进入项目目录
cd ~/tts

# 激活环境
echo "📦 激活Python环境..."
source ~/miniconda3/bin/activate

# 检查vLLM服务状态
echo ""
echo "🔍 检查vLLM服务..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✓ vLLM服务运行正常 (localhost:8000)"
else
    echo "❌ vLLM服务未运行！"
    echo "请先运行: ./scripts/start_local_services.sh"
    exit 1
fi

# 检查.env配置
echo ""
echo "🔍 检查配置文件..."
if [ ! -f .env ]; then
    echo "❌ .env文件不存在！"
    echo "请创建.env文件并配置Embedding/Reranking API"
    exit 1
fi

# 显示配置
echo ""
echo "📋 实验配置:"
echo "  LLM: Qwen/Qwen3-32B @ localhost:8000"
echo "  Embedding: $(grep EMBEDDING_MODEL .env | cut -d'=' -f2)"
echo "  测试用例: 5个长文本场景"
echo "  方法数量: 4个"
echo ""

# 创建输出目录
mkdir -p outputs
mkdir -p logs

# 运行实验
echo "=========================================="
echo "🚀 开始运行实验..."
echo "=========================================="
echo ""

python experiments/test_03_v3_server.py 2>&1 | tee logs/experiment3_v3_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "✅ 实验完成！"
echo "=========================================="
echo ""
echo "结果文件: outputs/experiment3_v3_server_results_*.json"
echo "日志文件: logs/experiment3_v3_*.log"
echo ""
echo "分析结果:"
echo "  python experiments/analyze_exp3_v3_results.py"
echo ""
