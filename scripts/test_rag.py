#!/usr/bin/env python3
"""测试RAG检索功能"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.knowledge.rag_searcher import RAGSearcher
from rag_utils import EmbeddingService, RerankingService


def main():
    print("初始化RAG检索器...")

    # 初始化服务
    embedding_service = EmbeddingService()
    reranking_service = RerankingService()  # 无需传参，从.env读取

    # 创建RAG检索器
    searcher = RAGSearcher(
        endpoint=settings.azure_search.endpoint,
        api_key=settings.azure_search.key,
        index_name="test-kb",
        embedding_service=embedding_service,
        reranking_service=reranking_service,
    )

    # 测试查询
    test_queries = [
        "如何重置密码？",
        "客服电话是多少？",
        "退货流程是什么？",
    ]

    print("\n" + "=" * 60)
    print("测试RAG检索")
    print("=" * 60)

    for query in test_queries:
        print(f"\n【查询】: {query}")
        result = searcher.search(query)

        if result["type"] == "direct_answer":
            print(f"【直接回答】({result['confidence']:.2f})")
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}")
        else:
            print(f"【检索结果】({len(result['docs'])} 个文档)")
            for i, doc in enumerate(result["docs"][:3]):
                print(f"\n{i+1}. [{doc['type']}] {doc['title']}")
                print(f"   分数: {doc['rerank_score']:.3f}")
                print(f"   内容: {doc['content'][:100]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
