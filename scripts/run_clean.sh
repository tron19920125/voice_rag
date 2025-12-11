#!/bin/bash
# 快速启动脚本 - 清除环境变量并运行命令

# 清除可能冲突的环境变量
unset RERANKING_URL
unset RERANKING_MODEL
unset RERANKING_TOKEN
unset EMBEDDING_URL
unset EMBEDDING_MODEL
unset EMBEDDING_TOKEN
unset QWEN_API_BASE
unset QWEN_MODEL
unset QWEN_TOKEN

# 运行传入的命令
exec "$@"
