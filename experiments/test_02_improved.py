"""
实验2改进版：完整的4方案对比实验

方案对比：
- Baseline A: 仅业务文档 + Dense 向量（无企业图谱）
- Baseline B: 仅企业图谱 + BM25（无业务文档）
- Baseline C: 图谱+文档 + Dense 向量（无混合检索）
- 方案 D: 图谱+文档 + BM25+Dense混合 + Reranking（推荐）
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
from rag_utils import (
    EmbeddingService,
    VectorIndex,
    RerankingService,
    BM25,
    hybrid_search,
    build_rag_context
)
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS
from data.company_graph import convert_all_companies_to_documents
from data.test_cases_exp2 import TEST_CASES

load_dotenv()

MODEL = "qwen3-8b"
SYSTEM_PROMPT = """你是 TechFlow Industrial Solutions 公司的专业销售顾问。基于提供的信息准确推荐产品和解决方案。

重要原则：
1. 充分利用企业图谱信息（行业、规模、痛点、合作历史）
2. 准确识别企业名称，避免混淆相似企业
3. 结合企业痛点与产品特性，进行精准推荐
4. 如果知识库中没有相关信息，明确告知客户

回答要求简洁、准确、专业。"""


class Experiment2Improved:
    """实验2改进版 - 完整4方案对比"""

    def __init__(self):
        print("\n" + "="*70)
        print("实验2改进版：4方案完整对比")
        print("="*70 + "\n")

        self.client = OpenAI(
            api_key=os.getenv("QWEN_TOKEN"),
            base_url=os.getenv("QWEN_API_BASE")
        )

        # 初始化组件
        print("初始化 RAG 组件...")
        self.embedding_service = EmbeddingService()
        self.reranking_service = RerankingService()

        # 准备数据
        self._prepare_data()

        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _prepare_data(self):
        """准备数据"""
        print("\n准备数据...")

        # 企业图谱文档
        self.company_docs = convert_all_companies_to_documents()
        print(f"✓ 企业图谱: {len(self.company_docs)} 个文档")

        # 业务文档
        self.business_docs = FICTIONAL_DOCUMENTS
        print(f"✓ 业务文档: {len(self.business_docs)} 个文档")

        # 合并文档
        self.all_docs = self.company_docs + self.business_docs
        print(f"✓ 总计: {len(self.all_docs)} 个文档")

    def _build_indexes(self, docs: List[Dict]):
        """构建索引"""
        # 向量索引
        vector_index = VectorIndex(self.embedding_service)
        vector_index.add_documents(docs)

        # BM25 索引
        corpus = [f"{doc['title']} {doc['content']}" for doc in docs]
        bm25_index = BM25(corpus)

        return vector_index, bm25_index

    def _generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        """生成回答"""
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n用户问题：{query}"}
            ],
            temperature=0.1,
            max_tokens=800,
            extra_body={"enable_thinking": False}
        )
        answer = response.choices[0].message.content
        elapsed = time.perf_counter() - start
        return answer, elapsed

    def run_method_a_business_only_dense(self, query: str) -> Dict:
        """方案A: 仅业务文档 + Dense向量"""
        print("\n【方案 A】仅业务文档 + Dense向量")

        # 构建索引（仅业务文档）
        vector_index, _ = self._build_indexes(self.business_docs)

        # 检索
        t0 = time.perf_counter()
        query_vector = self.embedding_service.embed_single(query)
        results = vector_index.search(query_vector, top_k=20)
        retrieved = self.reranking_service.rerank(query, results, top_k=5)
        retrieval_time = time.perf_counter() - t0

        # 生成
        context = build_rag_context(retrieved)
        answer, gen_time = self._generate_answer(query, context)

        print(f"✓ 检索耗时: {retrieval_time*1000:.1f}ms, 生成耗时: {gen_time*1000:.1f}ms")

        return {
            "method": "A_business_only_dense",
            "answer": answer,
            "retrieved_docs": [{"id": d.get("id", "unknown"), "title": d.get("title", "")} for d in retrieved],
            "timing": {"retrieval_ms": retrieval_time*1000, "generation_ms": gen_time*1000}
        }

    def run_method_b_company_only_bm25(self, query: str) -> Dict:
        """方案B: 仅企业图谱 + BM25"""
        print("\n【方案 B】仅企业图谱 + BM25")

        # 构建索引（仅企业图谱）
        vector_index, bm25_index = self._build_indexes(self.company_docs)

        # 检索
        t0 = time.perf_counter()
        bm25_results = bm25_index.search(query, top_k=20)
        retrieved = []
        for idx, score in bm25_results[:5]:
            doc = self.company_docs[idx].copy()
            doc['bm25_score'] = score
            retrieved.append(doc)
        retrieval_time = time.perf_counter() - t0

        # 生成
        context = build_rag_context(retrieved)
        answer, gen_time = self._generate_answer(query, context)

        print(f"✓ 检索耗时: {retrieval_time*1000:.1f}ms, 生成耗时: {gen_time*1000:.1f}ms")

        return {
            "method": "B_company_only_bm25",
            "answer": answer,
            "retrieved_docs": [{"id": d.get("id", "unknown"), "title": d.get("title", "")} for d in retrieved],
            "timing": {"retrieval_ms": retrieval_time*1000, "generation_ms": gen_time*1000}
        }

    def run_method_c_all_dense(self, query: str) -> Dict:
        """方案C: 图谱+文档 + Dense向量（无混合检索）"""
        print("\n【方案 C】图谱+文档 + Dense向量")

        # 构建索引（全部文档）
        vector_index, _ = self._build_indexes(self.all_docs)

        # 检索
        t0 = time.perf_counter()
        query_vector = self.embedding_service.embed_single(query)
        results = vector_index.search(query_vector, top_k=20)
        retrieved = self.reranking_service.rerank(query, results, top_k=5)
        retrieval_time = time.perf_counter() - t0

        # 生成
        context = build_rag_context(retrieved)
        answer, gen_time = self._generate_answer(query, context)

        print(f"✓ 检索耗时: {retrieval_time*1000:.1f}ms, 生成耗时: {gen_time*1000:.1f}ms")

        return {
            "method": "C_all_dense",
            "answer": answer,
            "retrieved_docs": [{"id": d.get("id", "unknown"), "title": d.get("title", "")} for d in retrieved],
            "timing": {"retrieval_ms": retrieval_time*1000, "generation_ms": gen_time*1000}
        }

    def run_method_d_hybrid(self, query: str) -> Dict:
        """方案D: 图谱+文档 + 混合检索（BM25+Dense+RRF+Reranking）"""
        print("\n【方案 D】图谱+文档 + 混合检索（推荐）")

        # 构建索引（全部文档）
        vector_index, bm25_index = self._build_indexes(self.all_docs)

        # 混合检索
        t0 = time.perf_counter()
        retrieved, debug_info = hybrid_search(
            query=query,
            bm25_index=bm25_index,
            vector_index=vector_index,
            embedding_service=self.embedding_service,
            reranking_service=self.reranking_service,
            verbose=False
        )
        retrieval_time = time.perf_counter() - t0

        # 生成
        context = build_rag_context(retrieved)
        answer, gen_time = self._generate_answer(query, context)

        print(f"✓ 检索耗时: {retrieval_time*1000:.1f}ms, 生成耗时: {gen_time*1000:.1f}ms")
        print(f"  - BM25与Dense重叠: {debug_info['overlap_bm25_dense']} 个")

        return {
            "method": "D_hybrid",
            "answer": answer,
            "retrieved_docs": [{"id": d.get("id", "unknown"), "title": d.get("title", "")} for d in retrieved],
            "timing": {"retrieval_ms": retrieval_time*1000, "generation_ms": gen_time*1000},
            "debug_info": debug_info
        }

    def run_comparison(self, test_query: str = None):
        """运行4方案对比"""
        if test_query is None:
            test_query = "星辰金融集团想做实时风控，应该推荐哪个产品？"

        print("\n" + "="*70)
        print(f"4方案对比测试")
        print(f"查询: {test_query}")
        print("="*70)

        results = {}

        # 运行4个方案
        results['A'] = self.run_method_a_business_only_dense(test_query)
        results['B'] = self.run_method_b_company_only_bm25(test_query)
        results['C'] = self.run_method_c_all_dense(test_query)
        results['D'] = self.run_method_d_hybrid(test_query)

        # 对比分析
        self._print_comparison(results)

        return results

    def _print_comparison(self, results: Dict):
        """打印对比结果"""
        print("\n" + "="*70)
        print("4方案对比结果")
        print("="*70)

        for method, result in results.items():
            print(f"\n【方案 {method}】{result['method']}")
            print(f"检索耗时: {result['timing']['retrieval_ms']:.1f}ms")
            print(f"生成耗时: {result['timing']['generation_ms']:.1f}ms")
            print(f"召回文档: {', '.join([d['title'][:20] for d in result['retrieved_docs'][:3]])}")
            print(f"回答长度: {len(result['answer'])} 字符")

            # 简单的关键词匹配分析
            answer = result['answer']
            keywords = {
                "企业背景": any(k in answer for k in ["星辰金融", "金融科技", "证券"]),
                "痛点识别": any(k in answer for k in ["实时风控", "延迟", "瓶颈"]),
                "产品推荐": any(k in answer for k in ["DataStream", "FlowMind", "FlowControl"]),
            }
            print(f"内容覆盖: {', '.join([k for k, v in keywords.items() if v])}")

    def save_results(self, results: Dict):
        """保存结果"""
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        filename = f"experiment2_comparison_{self.experiment_timestamp}.json"

        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 结果已保存: {output_dir / filename}")


def main():
    runner = Experiment2Improved()

    # 运行对比实验
    results = runner.run_comparison()

    # 保存结果
    runner.save_results(results)

    print("\n" + "="*70)
    print("实验2改进版完成！")
    print("="*70)


if __name__ == "__main__":
    main()
