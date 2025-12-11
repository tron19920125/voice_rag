#!/usr/bin/env python3
"""
基础设施验证脚本
测试 RAG 工具库的所有核心组件
"""

import sys
import time


def test_imports():
    """测试导入"""
    print("\n" + "="*70)
    print("【测试 1】模块导入")
    print("="*70)

    try:
        from rag_utils import (
            EmbeddingService,
            VectorIndex,
            RerankingService,
            retrieve_by_similarity,
            rag_retrieve_and_rerank,
            build_rag_context
        )
        from data.knowledge_base import DOCUMENTS, TEST_QUERIES

        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {str(e)}")
        return False


def test_embedding_service():
    """测试 Embedding 服务"""
    print("\n" + "="*70)
    print("【测试 2】Embedding 服务")
    print("="*70)

    try:
        from rag_utils import EmbeddingService

        service = EmbeddingService()
        print(f"Model: {service.model}")
        print(f"Dimension: {service.dimension}")

        # 测试单文本嵌入
        text = "西门子自动化解决方案"
        embedding = service.embed_single(text)

        print(f"\n✅ Embedding 服务工作正常")
        print(f"向量维度: {len(embedding)}")
        print(f"向量前 5 个值: {embedding[:5]}")
        return True

    except Exception as e:
        print(f"\n❌ Embedding 服务失败: {str(e)}")
        return False


def test_vector_index():
    """测试向量索引"""
    print("\n" + "="*70)
    print("【测试 3】向量索引")
    print("="*70)

    try:
        from rag_utils import EmbeddingService, VectorIndex
        from data.knowledge_base import DOCUMENTS

        embedding_service = EmbeddingService()
        index = VectorIndex(embedding_service)

        # 添加文档
        start = time.time()
        index.add_documents(DOCUMENTS)
        elapsed = time.time() - start

        print(f"\n✅ 向量索引工作正常")
        print(f"索引文档数: {len(index.documents)}")
        print(f"索引耗时: {elapsed:.2f}s")

        # 测试检索
        query_vector = embedding_service.embed_single("生产效率提升")
        results = index.search(query_vector, top_k=3)

        print(f"\n检索测试结果 Top-3:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (相似度: {result['similarity']:.3f})")

        return True

    except Exception as e:
        print(f"\n❌ 向量索引失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_reranking_service():
    """测试 Reranking 服务"""
    print("\n" + "="*70)
    print("【测试 4】Reranking 服务")
    print("="*70)

    try:
        from rag_utils import RerankingService

        service = RerankingService()
        print(f"Model: {service.model}")

        # 测试精排序
        query = "生产效率提升方案"
        passages = [
            {"title": "文档1", "content": "西门子 S7 系列 PLC 产品介绍"},
            {"title": "文档2", "content": "生产效率提升案例研究"},
            {"title": "文档3", "content": "MindSphere 工业物联网平台"}
        ]

        results = service.rerank(query, passages, top_k=2)

        print(f"\n✅ Reranking 服务工作正常")
        print(f"精排结果 Top-2:")
        for i, result in enumerate(results, 1):
            score = result.get('rerank_score', 0)
            print(f"  {i}. {result['title']} (分数: {score:.3f})")

        return True

    except Exception as e:
        print(f"\n❌ Reranking 服务失败: {str(e)}")
        return False


def test_rag_pipeline():
    """测试完整 RAG 流程"""
    print("\n" + "="*70)
    print("【测试 5】完整 RAG 流程")
    print("="*70)

    try:
        from rag_utils import (
            EmbeddingService,
            VectorIndex,
            RerankingService,
            rag_retrieve_and_rerank,
            build_rag_context
        )
        from data.knowledge_base import DOCUMENTS

        # 初始化服务
        print("\n初始化服务...")
        embedding_service = EmbeddingService()
        reranking_service = RerankingService()
        index = VectorIndex(embedding_service)

        # 构建索引
        index.add_documents(DOCUMENTS)

        # 测试查询
        query = "我们生产效率低下，想了解西门子的解决方案"
        print(f"\n查询: {query}")

        start = time.time()
        rerank_results, retrieval_results = rag_retrieve_and_rerank(
            query=query,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
            index=index,
            retrieval_top_k=6,
            rerank_top_k=3,
            verbose=False
        )
        elapsed = time.time() - start

        print(f"\n✅ RAG 流程工作正常")
        print(f"总耗时: {elapsed:.2f}s")
        print(f"\n精排结果 Top-3:")
        for i, result in enumerate(rerank_results, 1):
            score = result.get('rerank_score', 0)
            print(f"  {i}. {result['title']} (分数: {score:.3f})")

        # 测试上下文构建
        context = build_rag_context(rerank_results)
        print(f"\n上下文长度: {len(context)} 字符")

        return True

    except Exception as e:
        print(f"\n❌ RAG 流程失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("基础设施验证脚本")
    print("="*70)

    tests = [
        ("模块导入", test_imports),
        ("Embedding 服务", test_embedding_service),
        ("向量索引", test_vector_index),
        ("Reranking 服务", test_reranking_service),
        ("完整 RAG 流程", test_rag_pipeline)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} 发生异常: {str(e)}")
            results[name] = False

    # 输出总结
    print("\n" + "="*70)
    print("【验证总结】")
    print("="*70)

    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s}: {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n通过率: {passed}/{total}")

    if all(results.values()):
        print("\n🎉 基础设施搭建完成！所有组件工作正常。")
        sys.exit(0)
    else:
        print("\n⚠️  部分组件失败，请检查错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
