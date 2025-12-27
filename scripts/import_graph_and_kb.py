#!/usr/bin/env python3
"""
将company_graph和fictional_knowledge_base数据导入Azure AI Search
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.knowledge.indexer import IndexManager
from rag_utils import EmbeddingService
import time

# 导入数据源
from data.company_graph import get_all_company_documents
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS


def convert_to_index_format(docs: list, doc_type: str) -> list:
    """
    转换为索引格式

    Args:
        docs: 文档列表
        doc_type: 文档类型（company_graph 或 fictional_kb）

    Returns:
        转换后的文档列表
    """
    results = []

    for doc in docs:
        # 统一格式
        record = {
            "id": doc["id"],
            "type": "document",  # 都作为文档类型
            "content": doc["content"],
            "title": doc.get("title", ""),
            "question": None,
            "answer": None,
            "metadata": {
                "source": doc_type,
                "doc_type": doc.get("type", "unknown"),
                **doc.get("metadata", {})
            }
        }
        results.append(record)

    return results


def main():
    print("=" * 70)
    print("导入企业图谱和知识库数据到Azure AI Search")
    print("=" * 70)

    start_time = time.time()

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    company_docs = get_all_company_documents()
    fictional_docs = FICTIONAL_DOCUMENTS

    print(f"✓ 企业图谱文档: {len(company_docs)} 个")
    print(f"✓ 知识库文档: {len(fictional_docs)} 个")

    # 2. 转换格式
    print("\n[2/5] 转换格式...")
    company_records = convert_to_index_format(company_docs, "company_graph")
    fictional_records = convert_to_index_format(fictional_docs, "fictional_kb")

    all_records = company_records + fictional_records
    print(f"✓ 总记录数: {len(all_records)} 条")

    # 3. 生成embedding
    print("\n[3/5] 生成embedding...")
    embedding_service = EmbeddingService()

    # 批量生成
    texts = [record["content"] for record in all_records]
    embeddings = embedding_service.embed_texts(texts)

    for record, embedding in zip(all_records, embeddings):
        record["content_vector"] = embedding

    print(f"✓ 完成 {len(all_records)} 条记录的向量化")

    # 4. 创建/更新索引
    print("\n[4/5] 管理索引...")
    index_manager = IndexManager(
        endpoint=settings.azure_search.endpoint,
        api_key=settings.azure_search.key,
        index_name="customer-service-kb"  # 使用现有索引名
    )

    # 强制重建索引
    index_manager.create_index(vector_dimensions=1024, force_recreate=True)

    # 5. 上传文档
    print("\n[5/5] 上传文档...")
    stats = index_manager.upload_documents(all_records, batch_size=100)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("导入完成")
    print("=" * 70)
    print(f"✓ 企业图谱: {len(company_records)} 条")
    print(f"✓ 知识库: {len(fictional_records)} 条")
    print(f"✓ 上传成功: {stats['success']}/{stats['total']}")
    print(f"✓ 索引: customer-service-kb")
    print(f"✓ 总耗时: {elapsed:.1f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
