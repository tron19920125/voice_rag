#!/usr/bin/env python3
"""
知识库处理脚本
扫描文件夹 → 解析文档/QA → 生成embedding → 上传Azure AI Search
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.knowledge.document_processor import DocumentProcessor, QAProcessor
from src.knowledge.indexer import IndexManager
from rag_utils import EmbeddingService


def scan_files(directory: str, extensions: List[str]) -> List[str]:
    """
    扫描目录下的指定类型文件

    Args:
        directory: 目录路径
        extensions: 文件扩展名列表（如['.pdf', '.docx']）

    Returns:
        文件路径列表
    """
    if not os.path.exists(directory):
        return []

    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))

    return files


def process_documents(doc_processor: DocumentProcessor, file_paths: List[str]) -> List[Dict]:
    """处理文档文件"""
    all_chunks = []
    for file_path in file_paths:
        try:
            print(f"处理文档: {os.path.basename(file_path)}")
            chunks = doc_processor.process_file(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue

    return all_chunks


def process_qa_files(qa_processor: QAProcessor, file_paths: List[str]) -> List[Dict]:
    """处理QA文件"""
    all_qa = []
    for file_path in file_paths:
        try:
            print(f"处理QA: {os.path.basename(file_path)}")
            qa_list = qa_processor.process_file(file_path)
            all_qa.extend(qa_list)
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue

    return all_qa


def add_embeddings(records: List[Dict], embedding_service: EmbeddingService) -> List[Dict]:
    """为记录添加向量"""
    print(f"生成embedding: {len(records)} 条记录")

    # 提取文本
    texts = [record["content"] for record in records]

    # 批量生成embedding
    embeddings = embedding_service.embed_texts(texts)

    # 添加到记录
    for record, embedding in zip(records, embeddings):
        record["content_vector"] = embedding

    return records


def main():
    parser = argparse.ArgumentParser(description="处理知识库并上传到Azure AI Search")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="data/documents",
        help="文档文件夹路径",
    )
    parser.add_argument(
        "--qa-dir",
        type=str,
        default="data/qa",
        help="QA文件夹路径",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="customer-service-kb",
        help="Azure AI Search索引名称",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建索引",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量更新（尚未实现）",
    )

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("知识库处理")
    print("=" * 60)

    # 1. 扫描文件
    print("\n[1/5] 扫描文件...")
    doc_extensions = [".pdf", ".docx", ".doc"]
    qa_extensions = [".json", ".jsonl", ".csv", ".xlsx", ".xls", ".md", ".markdown"]

    doc_files = scan_files(args.docs_dir, doc_extensions)
    qa_files = scan_files(args.qa_dir, qa_extensions)

    print(f"✓ 文档文件夹: {args.docs_dir} ({len(doc_files)} 个文件)")
    print(f"✓ QA文件夹: {args.qa_dir} ({len(qa_files)} 个文件)")

    if not doc_files and not qa_files:
        print("未找到任何文件，退出")
        return

    # 2. 处理文档
    print("\n[2/5] 处理文档...")
    doc_processor = DocumentProcessor()
    qa_processor = QAProcessor()

    doc_chunks = process_documents(doc_processor, doc_files) if doc_files else []
    qa_records = process_qa_files(qa_processor, qa_files) if qa_files else []

    print(f"✓ 文档处理: {len(doc_files)} 个文档 → {len(doc_chunks)} 个块")
    print(f"✓ QA处理: {len(qa_files)} 个文件 → {len(qa_records)} 个问答对")

    all_records = doc_chunks + qa_records

    if not all_records:
        print("没有有效记录，退出")
        return

    # 3. 生成embedding
    print("\n[3/5] 生成embedding...")
    embedding_service = EmbeddingService()
    all_records = add_embeddings(all_records, embedding_service)
    print(f"✓ 完成 {len(all_records)} 条记录的向量化")

    # 4. 创建索引
    print("\n[4/5] 管理索引...")
    index_manager = IndexManager(
        endpoint=settings.azure_search.endpoint,
        api_key=settings.azure_search.key,
        index_name=args.index_name,
    )

    if not index_manager.index_exists() or args.force:
        print(f"创建索引: {args.index_name}")
        index_manager.create_index(vector_dimensions=1024, force_recreate=args.force)
    else:
        print(f"索引已存在: {args.index_name}")

    # 5. 上传文档
    print("\n[5/5] 上传文档...")
    stats = index_manager.upload_documents(all_records, batch_size=100)

    # 统计信息
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"✓ 文档: {len(doc_chunks)} 块")
    print(f"✓ QA: {len(qa_records)} 对")
    print(f"✓ 上传: {stats['success']}/{stats['total']} 成功")
    print(f"✓ 索引: {args.index_name}")
    print(f"✓ 总耗时: {elapsed_time:.1f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
