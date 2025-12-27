#!/usr/bin/env python3
"""测试企业图谱和知识库的RAG检索"""

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
    reranking_service = RerankingService()

    # 创建RAG检索器
    searcher = RAGSearcher(
        endpoint=settings.azure_search.endpoint,
        api_key=settings.azure_search.key,
        index_name="customer-service-kb",
        embedding_service=embedding_service,
        reranking_service=reranking_service,
    )

    # 测试查询
    test_queries = [
        "鼎盛科技的合作案例有哪些？",
        "FlowControl-X500的价格是多少？",
        "星辰汽车零部件厂使用什么产品？",
        "云帆新能源的业务痛点是什么？",
    ]

    print("\n" + "=" * 70)
    print("测试企业图谱+知识库 RAG检索")
    print("=" * 70)

    for query in test_queries:
        print(f"\n【查询】: {query}")
        result = searcher.search(query)

        if result["type"] == "direct_answer":
            print(f"【直接回答】(置信度: {result['confidence']:.2f})")
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}")
        else:
            print(f"【检索结果】({len(result['docs'])} 个文档)")
            for i, doc in enumerate(result["docs"][:3], 1):
                print(f"\n{i}. [{doc['title']}]")
                print(f"   分数: {doc['rerank_score']:.3f}")
                print(f"   来源: {doc['metadata'].get('source', 'unknown')}")
                print(f"   内容: {doc['content'][:150]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
