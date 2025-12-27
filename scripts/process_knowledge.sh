#!/bin/bash
# 知识库处理bash wrapper

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 默认参数
DOCS_DIR="${DOCS_DIR:-data/documents}"
QA_DIR="${QA_DIR:-data/qa}"
INDEX_NAME="${INDEX_NAME:-customer-service-kb}"

echo "=================================="
echo "知识库处理脚本"
echo "=================================="
echo "文档目录: $DOCS_DIR"
echo "QA目录: $QA_DIR"
echo "索引名称: $INDEX_NAME"
echo "=================================="
echo ""

# 运行Python脚本
uv run python scripts/process_knowledge.py \
    --docs-dir "$DOCS_DIR" \
    --qa-dir "$QA_DIR" \
    --index-name "$INDEX_NAME" \
    "$@"
