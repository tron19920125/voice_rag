#!/bin/bash
# 一键启动：vLLM服务 + TTS语音助手（使用本地模型）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "启动TTS语音助手（本地vLLM模式）"
echo "=========================================="
echo ""

# ===== 1. 检查并启动vLLM服务 =====
echo "🔍 检查vLLM服务状态..."

check_vllm() {
    local port=$1
    curl -s http://localhost:$port/v1/models > /dev/null 2>&1
}

# 检查两个端口
vllm_8b_running=false
vllm_14b_running=false

if check_vllm 8000; then
    echo "  ✓ vLLM 8B服务已运行 (端口8000)"
    vllm_8b_running=true
else
    echo "  ✗ vLLM 8B服务未运行"
fi

if check_vllm 8001; then
    echo "  ✓ vLLM 14B服务已运行 (端口8001)"
    vllm_14b_running=true
else
    echo "  ✗ vLLM 14B服务未运行"
fi

# 如果有任一服务未运行，启动vLLM服务
if [ "$vllm_8b_running" = false ] || [ "$vllm_14b_running" = false ]; then
    echo ""
    echo "🚀 启动vLLM服务..."
    echo ""

    # 在远程服务器上执行启动脚本
    if [ -f "$SCRIPT_DIR/start_dual_vllm_services.sh" ]; then
        # 上传并执行
        scp "$SCRIPT_DIR/start_dual_vllm_services.sh" azure-a100:~/
        ssh azure-a100 "bash ~/start_dual_vllm_services.sh"

        echo ""
        echo "⏳ 等待vLLM服务完全启动（60秒）..."
        sleep 60
    else
        echo "  ⚠️  未找到start_dual_vllm_services.sh，请手动启动vLLM服务"
        echo "  命令: ssh azure-a100 'cd ~/tts && bash scripts/start_dual_vllm_services.sh'"
        exit 1
    fi
fi

# ===== 2. 配置使用本地vLLM =====
echo ""
echo "⚙️  配置.env使用本地vLLM..."

# 备份.env
if [ ! -f "$PROJECT_ROOT/.env.backup" ]; then
    cp "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.backup"
    echo "  已备份.env到.env.backup"
fi

# 修改USE_LOCAL_VLLM为true
sed -i.tmp 's/^USE_LOCAL_VLLM=.*/USE_LOCAL_VLLM=true/' "$PROJECT_ROOT/.env"
rm -f "$PROJECT_ROOT/.env.tmp"
echo "  ✓ 已设置 USE_LOCAL_VLLM=true"

# ===== 3. 启动TTS语音助手 =====
echo ""
echo "🎙️  启动TTS语音助手..."
echo ""

cd "$PROJECT_ROOT"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# 启动主程序
uv run python src/main.py

# ===== 清理：退出时恢复.env =====
trap cleanup EXIT

cleanup() {
    echo ""
    echo "🔄 恢复.env配置..."
    if [ -f "$PROJECT_ROOT/.env.backup" ]; then
        mv "$PROJECT_ROOT/.env.backup" "$PROJECT_ROOT/.env"
        echo "  ✓ 已恢复.env"
    fi
}
