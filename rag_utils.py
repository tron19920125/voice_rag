"""
RAG 工具库
包含 Embedding、向量索引、Reranking 等核心组件
"""

import os
from typing import List, Dict, Optional
import numpy as np
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class EmbeddingService:
    """在线 Embedding 服务（使用 BAAI/bge-m3）"""

    def __init__(self):
        self.model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.url = os.getenv("EMBEDDING_URL", "https://api.siliconflow.cn/v1/embeddings")
        self.token = os.getenv("EMBEDDING_TOKEN")
        self.dimension = 1024  # bge-m3 输出维度

        if not self.token:
            raise ValueError("EMBEDDING_TOKEN 未设置，请检查 .env 文件")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本 → 向量

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量是长度为 1024 的浮点数列表
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            raise RuntimeError(f"Embedding API 调用失败: {str(e)}")

    def embed_single(self, text: str) -> np.ndarray:
        """
        嵌入单个文本 → 向量（numpy 数组）

        Args:
            text: 单个文本

        Returns:
            numpy 数组向量
        """
        embeddings = self.embed_texts([text])
        return np.array(embeddings[0])


class VectorIndex:
    """向量索引（存储与检索）"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.documents = {}  # doc_id -> {id, title, content}
        self.vectors = {}    # doc_id -> np.array(1024,)

    def add_documents(self, documents: List[Dict]):
        """
        批量添加文档并生成向量

        Args:
            documents: 文档列表，每个文档包含 id, title, content
        """
        if not documents:
            return

        # 提取内容（标题 + 正文）
        contents = [f"{doc['title']} {doc['content']}" for doc in documents]

        # 批量嵌入
        print(f"📊 正在嵌入 {len(documents)} 个文档...")
        embeddings = self.embedding_service.embed_texts(contents)

        # 存储
        for doc, embedding in zip(documents, embeddings):
            self.documents[doc["id"]] = doc
            self.vectors[doc["id"]] = np.array(embedding)

        print(f"✓ 成功索引 {len(documents)} 个文档")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        相似度检索

        Args:
            query_vector: 查询向量
            top_k: 返回前 k 个结果

        Returns:
            结果列表，每个结果包含 doc_id, similarity, content
        """
        if not self.vectors:
            return []

        # 计算余弦相似度
        similarities = []
        for doc_id, doc_vector in self.vectors.items():
            similarity = float(np.dot(query_vector, doc_vector))
            similarities.append({
                "doc_id": doc_id,
                "similarity": similarity,
                "title": self.documents[doc_id]["title"],
                "content": self.documents[doc_id]["content"]
            })

        # 排序并返回 Top-K
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]


class RerankingService:
    """在线 Reranking 服务（使用 BAAI/bge-reranker-v2-m3）"""

    def __init__(self):
        self.model = os.getenv("RERANKING_MODEL", "BAAI/bge-reranker-v2-m3")
        self.url = os.getenv("RERANKING_URL", "https://api.siliconflow.cn/v1/rerank")
        self.token = os.getenv("RERANKING_TOKEN")

        if not self.token:
            raise ValueError("RERANKING_TOKEN 未设置，请检查 .env 文件")

    def rerank(self, query: str, passages: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        精排序

        Args:
            query: 查询文本
            passages: 候选文档列表，每个包含 content 字段
            top_k: 返回前 k 个结果

        Returns:
            重排序后的结果列表
        """
        if not passages:
            return []

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": [p["content"] for p in passages],
            "top_n": top_k
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # 返回重排后的结果
            results = []
            for item in data["results"]:
                idx = item["index"]
                passage = passages[idx].copy()
                passage["rerank_score"] = item.get("relevance_score", 0)
                results.append(passage)

            return results

        except Exception as e:
            raise RuntimeError(f"Reranking API 调用失败: {str(e)}")


def retrieve_by_similarity(query: str, index: VectorIndex, embedding_service: EmbeddingService, top_k: int = 10) -> List[Dict]:
    """
    基于相似度的检索

    Args:
        query: 查询文本
        index: 向量索引
        embedding_service: Embedding 服务
        top_k: 返回前 k 个结果

    Returns:
        检索结果列表
    """
    # 嵌入查询
    query_vector = embedding_service.embed_single(query)

    # 检索
    results = index.search(query_vector, top_k=top_k)

    return results


def rag_retrieve_and_rerank(
    query: str,
    embedding_service: EmbeddingService,
    reranking_service: RerankingService,
    index: VectorIndex,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 3,
    verbose: bool = True
) -> tuple[List[Dict], List[Dict]]:
    """
    完整 RAG 流程：检索 + 精排序

    Args:
        query: 查询文本
        embedding_service: Embedding 服务
        reranking_service: Reranking 服务
        index: 向量索引
        retrieval_top_k: 初筛返回前 k 个
        rerank_top_k: 精排返回前 k 个
        verbose: 是否打印详细信息

    Returns:
        (rerank_results, retrieval_results) 元组
    """
    if verbose:
        print("\n" + "="*70)
        print(f"【RAG 流程】查询: {query}")
        print("="*70)

    # 步骤 1：相似度检索
    if verbose:
        print("\n[步骤 1] 相似度检索...")

    retrieval_results = retrieve_by_similarity(query, index, embedding_service, top_k=retrieval_top_k)

    if verbose:
        print(f"✓ 检索到 {len(retrieval_results)} 个候选文档")
        for i, result in enumerate(retrieval_results[:3], 1):
            print(f"  {i}. {result['title']} (相似度: {result['similarity']:.3f})")

    if not retrieval_results:
        return [], []

    # 步骤 2：Reranking 精排序
    if verbose:
        print(f"\n[步骤 2] Reranking 精排序...")

    rerank_results = reranking_service.rerank(query, retrieval_results, top_k=rerank_top_k)

    if verbose:
        print(f"✓ 精排后 Top {len(rerank_results)} 个结果：")
        for i, result in enumerate(rerank_results, 1):
            score = result.get('rerank_score', 0)
            print(f"  {i}. {result['title']} (分数: {score:.3f})")

    return rerank_results, retrieval_results


def build_rag_context(rag_results: List[Dict]) -> str:
    """
    组织 RAG 上下文

    Args:
        rag_results: RAG 检索结果

    Returns:
        格式化的上下文字符串
    """
    if not rag_results:
        return ""

    context_parts = ["以下是相关的知识库内容：\n"]

    for i, result in enumerate(rag_results, 1):
        context_parts.append(f"【文档 {i}】{result['title']}")
        context_parts.append(result['content'])
        context_parts.append("")

    context_parts.append("请基于以上内容回答用户问题。")

    return "\n".join(context_parts)
