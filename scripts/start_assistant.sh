#!/bin/bash
# 启动语音客服系统

set -e

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "================================"
echo "启动语音客服系统"
echo "================================"
echo ""

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "❌ 错误: .env 文件不存在"
    echo "请参考 .env.example 创建配置文件"
    exit 1
fi

# 检查知识库索引
echo "提示: 请确保已运行 ./scripts/process_knowledge.sh 处理知识库"
echo ""

# 启动主程序
echo "正在启动..."
uv run python src/main.py

echo ""
echo "系统已退出"
