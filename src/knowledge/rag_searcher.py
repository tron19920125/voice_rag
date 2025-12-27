"""
RAG检索模块
支持Azure AI Search混合检索 + Reranking + QA优先级加权
"""

import json
from typing import List, Dict, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


class RAGSearcher:
    """RAG检索器：混合检索 + Reranking + QA加权"""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        embedding_service,
        reranking_service,
        top_k: int = 20,
        rerank_top_k: int = 5,
        qa_weight_boost: float = 1.5,
        qa_direct_threshold: float = 0.85,
    ):
        """
        Args:
            endpoint: Azure AI Search端点
            api_key: API密钥
            index_name: 索引名称
            embedding_service: Embedding服务（来自rag_utils）
            reranking_service: Reranking服务（来自rag_utils）
            top_k: 混合检索召回数量
            rerank_top_k: Rerank后返回数量
            qa_weight_boost: QA类型加权系数
            qa_direct_threshold: QA直接返回阈值
        """
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(api_key)
        self.index_name = index_name

        self.embedding_service = embedding_service
        self.reranking_service = reranking_service

        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.qa_weight_boost = qa_weight_boost
        self.qa_direct_threshold = qa_direct_threshold

        # 创建搜索客户端
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

    def search(self, query: str) -> Dict:
        """
        执行RAG检索

        Args:
            query: 用户查询

        Returns:
            检索结果，格式：
            {
                "type": "direct_answer" 或 "rag_context",
                "answer": "..." (仅direct_answer),
                "docs": [...] (仅rag_context),
                "query": "原始查询"
            }
        """
        # 1. 生成query embedding
        query_vector = self.embedding_service.embed_single(query)

        # 2. 混合检索（关键词 + 向量）
        results = self.search_client.search(
            search_text=query,  # 关键词搜索
            vector_queries=[
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=self.top_k,
                    fields="content_vector",
                )
            ],
            select=["id", "type", "content", "question", "answer", "title", "metadata"],
            top=self.top_k,
            query_type="semantic",  # 启用语义排序
            semantic_configuration_name="semantic-config",
        )

        # 3. 解析检索结果
        retrieved_docs = []
        for result in results:
            # 解析metadata
            metadata = result.get("metadata", "{}")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            doc = {
                "id": result["id"],
                "type": result["type"],
                "content": result["content"],
                "title": result.get("title", ""),
                "question": result.get("question"),
                "answer": result.get("answer"),
                "metadata": metadata,
                "score": result.get("@search.score", 0.0),  # Azure原始分数
                "rerank_score": None,  # 待填充
            }
            retrieved_docs.append(doc)

        if not retrieved_docs:
            return {
                "type": "rag_context",
                "docs": [],
                "query": query,
            }

        # 4. Reranking
        reranked_docs = self._rerank_documents(query, retrieved_docs)

        # 5. QA类型加权
        for doc in reranked_docs:
            if doc["type"] == "qa":
                doc["rerank_score"] *= self.qa_weight_boost

        # 6. 按rerank_score排序
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 7. 特殊处理：高置信度QA直接返回
        top_doc = reranked_docs[0]
        if top_doc["type"] == "qa" and top_doc["rerank_score"] > self.qa_direct_threshold:
            return {
                "type": "direct_answer",
                "answer": top_doc["answer"],
                "question": top_doc["question"],
                "source": top_doc["metadata"].get("source", ""),
                "query": query,
                "confidence": top_doc["rerank_score"],
            }

        # 8. 返回top-k文档
        return {
            "type": "rag_context",
            "docs": reranked_docs[:self.rerank_top_k],
            "query": query,
        }

    def _rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        使用Reranker重新排序

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            重排序后的文档列表（添加rerank_score）
        """
        # 调用reranking服务（传递passages参数）
        try:
            rerank_results = self.reranking_service.rerank(
                query=query,
                passages=documents,  # 修正参数名
                top_k=len(documents),  # 返回全部
            )

            # rerank_results已经包含rerank_score，直接返回
            return rerank_results

        except Exception as e:
            print(f"Reranking失败，使用原始分数: {e}")
            # 使用Azure原始分数作为fallback
            for doc in documents:
                doc["rerank_score"] = doc["score"]

        return documents

    def batch_search(self, queries: List[str]) -> List[Dict]:
        """
        批量检索

        Args:
            queries: 查询列表

        Returns:
            检索结果列表
        """
        return [self.search(query) for query in queries]
