"""
RAG 工具库
包含 Embedding、向量索引、Reranking、BM25、混合检索等核心组件
"""

import os
import time
import math
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 懒加载 jieba（避免导入时报错）
_jieba = None

def get_jieba():
    """懒加载 jieba 分词库"""
    global _jieba
    if _jieba is None:
        try:
            import jieba
            _jieba = jieba
        except ImportError:
            raise ImportError("请安装 jieba: uv add jieba")
    return _jieba


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


# ========== 实验2新增：BM25 检索 ==========

class BM25:
    """
    BM25 算法实现 - 支持中文分词

    用于精确关键词匹配，补充 Dense 向量检索的不足
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 索引

        Args:
            corpus: 文档列表（字符串）
            k1: 词频饱和参数（通常 1.2-2.0）
            b: 长度归一化参数（0-1，0.75 常用）
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus

        jieba = get_jieba()

        # 使用 jieba 进行中文分词
        self.corpus_tokens = [list(jieba.cut(doc)) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 0
        self.doc_freqs = []
        self.idf = {}

        # 计算 IDF (使用分词后的 token)
        df = Counter()
        for tokens in self.corpus_tokens:
            df.update(set(tokens))

        for term, freq in df.items():
            self.idf[term] = math.log((len(corpus) - freq + 0.5) / (freq + 0.5) + 1)

        # 计算每个文档的词频
        for tokens in self.corpus_tokens:
            self.doc_freqs.append(Counter(tokens))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        检索最相关的文档

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            [(doc_idx, score), ...] 列表
        """
        jieba = get_jieba()

        # 查询也进行中文分词
        query_tokens = list(jieba.cut(query))
        scores = []

        for idx, doc_freq in enumerate(self.doc_freqs):
            score = 0
            for term in query_tokens:
                if term not in doc_freq:
                    continue

                freq = doc_freq[term]
                idf = self.idf.get(term, 0)

                # BM25 公式
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                score += idf * (numerator / denominator)

            scores.append((idx, score))

        # 排序返回 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ========== 实验2新增：RRF 融合算法 ==========

def rrf_fusion(rankings: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) 算法

    融合多个排序结果，平衡不同检索器的优势

    Args:
        rankings: 多个排序结果 [[doc1, doc2, ...], [doc3, doc1, ...]]
        k: 平滑参数，避免头部文档权重过大（通常 60）

    Returns:
        融合后的排序结果 [(doc_id, score), ...]

    示例:
        >>> rankings = [['doc1', 'doc2'], ['doc2', 'doc3']]
        >>> rrf_fusion(rankings, k=60)
        [('doc2', 0.033), ('doc1', 0.016), ('doc3', 0.016)]
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ========== 实验2新增：混合检索 ==========

def hybrid_search(
    query: str,
    bm25_index: BM25,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    reranking_service: RerankingService,
    bm25_top_k: int = 20,
    dense_top_k: int = 20,
    rrf_k: int = 60,
    final_top_k: int = 5,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    混合检索主流程（BM25 + Dense 向量 + RRF 融合 + Reranking）

    Args:
        query: 查询文本
        bm25_index: BM25 索引
        vector_index: 向量索引
        embedding_service: Embedding 服务
        reranking_service: Reranking 服务
        bm25_top_k: BM25 返回前 k 个
        dense_top_k: Dense 向量返回前 k 个
        rrf_k: RRF 平滑参数
        final_top_k: 最终返回前 k 个
        verbose: 是否打印详细信息

    Returns:
        (final_results, debug_info) 元组
        - final_results: 最终 Top-K 结果
        - debug_info: 调试信息（包含各阶段结果和延迟）
    """
    latency = {}

    if verbose:
        print("\n" + "="*70)
        print(f"【混合检索】查询: {query}")
        print("="*70)

    # 步骤 1：BM25 稀疏检索
    if verbose:
        print("\n[步骤 1] BM25 稀疏检索...")

    t0 = time.perf_counter()
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    latency["bm25_ms"] = (time.perf_counter() - t0) * 1000

    bm25_doc_ids = [str(idx) for idx, score in bm25_results]

    if verbose:
        print(f"✓ BM25 检索耗时 {latency['bm25_ms']:.1f}ms")
        print(f"  Top 3: {bm25_doc_ids[:3]}")

    # 步骤 2：Dense 向量检索
    if verbose:
        print("\n[步骤 2] Dense 向量检索...")

    t0 = time.perf_counter()
    query_vector = embedding_service.embed_single(query)
    dense_results = vector_index.search(query_vector, top_k=dense_top_k)
    latency["dense_ms"] = (time.perf_counter() - t0) * 1000

    dense_doc_ids = [r['doc_id'] for r in dense_results]

    if verbose:
        print(f"✓ Dense 检索耗时 {latency['dense_ms']:.1f}ms")
        print(f"  Top 3: {dense_doc_ids[:3]}")

    # 步骤 3：RRF 融合
    if verbose:
        print("\n[步骤 3] RRF 融合...")

    t0 = time.perf_counter()
    rrf_results = rrf_fusion([bm25_doc_ids, dense_doc_ids], k=rrf_k)
    latency["rrf_ms"] = (time.perf_counter() - t0) * 1000

    if verbose:
        print(f"✓ RRF 融合耗时 {latency['rrf_ms']:.1f}ms")
        print(f"  BM25 与 Dense 重叠: {len(set(bm25_doc_ids) & set(dense_doc_ids))} 个")

    # 步骤 4：取 Top-40 候选文档
    t0 = time.perf_counter()
    candidate_doc_ids = [doc_id for doc_id, score in rrf_results[:40]]
    candidate_docs = []

    for doc_id in candidate_doc_ids:
        if doc_id in vector_index.documents:
            doc = vector_index.documents[doc_id].copy()
            doc['rrf_score'] = next((score for did, score in rrf_results if did == doc_id), 0)
            candidate_docs.append(doc)

    latency["candidate_fetch_ms"] = (time.perf_counter() - t0) * 1000

    if verbose:
        print(f"\n[步骤 4] 准备候选文档: {len(candidate_docs)} 个")

    # 步骤 5：Reranking 精排
    if verbose:
        print("\n[步骤 5] Reranking 精排...")

    t0 = time.perf_counter()
    final_results = reranking_service.rerank(
        query=query,
        passages=candidate_docs,
        top_k=final_top_k
    )
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000
    latency["retrieval_total_ms"] = sum(latency.values())

    if verbose:
        print(f"✓ Reranking 耗时 {latency['rerank_ms']:.1f}ms")
        print(f"✓ 总耗时 {latency['retrieval_total_ms']:.1f}ms")
        print(f"\n最终 Top {len(final_results)} 结果：")
        for i, result in enumerate(final_results, 1):
            score = result.get('rerank_score', 0)
            print(f"  {i}. {result['title']} (Rerank分数: {score:.3f})")

    # 调试信息
    debug_info = {
        "bm25_results": bm25_doc_ids[:5],
        "dense_results": dense_doc_ids[:5],
        "rrf_fused": candidate_doc_ids[:10],
        "final_reranked": [r.get('id', r.get('doc_id', 'unknown')) for r in final_results],
        "overlap_bm25_dense": len(set(bm25_doc_ids) & set(dense_doc_ids)),
        "latency": latency
    }

    return final_results, debug_info

