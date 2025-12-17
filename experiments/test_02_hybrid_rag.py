"""
实验2：企业图谱 + 业务文档的混合 RAG 检索实验
测试混合检索(BM25 + Dense)与多源融合的效果

测试方案：
1. Baseline A: 仅业务文档 + Dense 向量
2. Baseline B: 仅企业图谱 + BM25
3. Baseline C: 图谱+文档 + Dense 向量
4. 方案 D (推荐): 图谱+文档 + BM25+Dense混合 + Reranking
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

# 添加项目根目录到路径
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
from data.company_graph import (
    COMPANY_GRAPH,
    convert_all_companies_to_documents
)
from data.test_cases_exp2 import TEST_CASES, evaluate_test_case

# 加载环境变量
load_dotenv()

# 模型配置（根据实验1结论，使用 qwen3-8b）
MODEL = "qwen3-8b"

# 系统提示词
SYSTEM_PROMPT = """你是 TechFlow Industrial Solutions 公司的专业销售顾问。你的任务是根据客户需求和企业背景，准确推荐合适的产品和解决方案。

重要原则：
1. 充分利用企业图谱信息（行业、规模、痛点、合作历史）
2. 准确识别企业名称，避免混淆相似企业
3. 结合企业痛点与产品特性，进行精准推荐
4. 注意识别竞品客户，谨慎推荐并说明差异化优势
5. 如果知识库中没有相关信息，明确告知客户
6. 提供的数据必须准确，包括企业信息、产品参数、案例数据等

回答要求：
- 专业、准确、有说服力
- 基于企业背景和事实数据
- 结构清晰，易于理解
"""


class Experiment2Runner:
    """实验2运行器 - 混合RAG检索与企业图谱融合"""

    def __init__(self):
        """初始化实验运行器"""
        print("\n" + "="*70)
        print("实验2：企业图谱 + 业务文档的混合 RAG 检索")
        print("="*70 + "\n")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=os.getenv("QWEN_TOKEN"),
            base_url=os.getenv("QWEN_API_BASE")
        )

        # 初始化 RAG 组件
        print("初始化 RAG 组件...")
        self.embedding_service = EmbeddingService()
        self.reranking_service = RerankingService()
        self.vector_index = VectorIndex(self.embedding_service)

        # 准备数据
        print("\n准备数据...")
        self._prepare_data()

        # 结果存储
        self.results = []
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _prepare_data(self):
        """准备实验数据：企业图谱 + 业务文档"""
        # 1. 转换企业图谱为文档
        print("\n[1] 转换企业图谱...")
        company_docs = convert_all_companies_to_documents()
        print(f"✓ 企业图谱: {len(company_docs)} 个文档")

        # 2. 业务文档
        print(f"\n[2] 业务文档: {len(FICTIONAL_DOCUMENTS)} 个文档")

        # 3. 合并所有文档
        self.all_docs = company_docs + FICTIONAL_DOCUMENTS
        print(f"\n[3] 总计: {len(self.all_docs)} 个文档")

        # 4. 构建向量索引
        print("\n[4] 构建向量索引...")
        self.vector_index.add_documents(self.all_docs)

        # 5. 构建 BM25 索引
        print("\n[5] 构建 BM25 索引...")
        corpus = [f"{doc['title']} {doc['content']}" for doc in self.all_docs]
        self.bm25_index = BM25(corpus)
        print("✓ BM25 索引构建完成")

    def run_single_test(
        self,
        test_case: Dict,
        use_hybrid: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        运行单个测试用例

        Args:
            test_case: 测试用例
            use_hybrid: 是否使用混合检索
            verbose: 是否打印详细信息

        Returns:
            测试结果字典
        """
        query = test_case["query"]
        start_time = time.perf_counter()

        # 步骤 1：检索
        if use_hybrid:
            # 混合检索
            retrieved_docs, debug_info = hybrid_search(
                query=query,
                bm25_index=self.bm25_index,
                vector_index=self.vector_index,
                embedding_service=self.embedding_service,
                reranking_service=self.reranking_service,
                bm25_top_k=20,
                dense_top_k=20,
                final_top_k=5,
                verbose=verbose
            )
        else:
            # 纯向量检索
            query_vector = self.embedding_service.embed_single(query)
            dense_results = self.vector_index.search(query_vector, top_k=20)
            retrieved_docs = self.reranking_service.rerank(query, dense_results, top_k=5)
            debug_info = {"method": "dense_only"}

        retrieval_time = time.perf_counter() - start_time

        # 步骤 2：构建上下文
        context = build_rag_context(retrieved_docs)

        # 步骤 3：LLM 生成回答
        generation_start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n用户问题：{query}"}
            ],
            temperature=0.1,
            max_tokens=1000,
            extra_body={"enable_thinking": False}  # qwen3 requires this for non-streaming
        )
        answer = response.choices[0].message.content
        generation_time = time.perf_counter() - generation_start

        # 步骤 4：评估
        evaluation = evaluate_test_case(test_case, answer, retrieved_docs)

        result = {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "query": query,
            "answer": answer,
            "retrieved_docs": [
                {
                    "id": doc.get("id", doc.get("doc_id", "unknown")),
                    "title": doc.get("title", ""),
                    "type": doc.get("type", "unknown")
                }
                for doc in retrieved_docs
            ],
            "evaluation": evaluation,
            "timing": {
                "retrieval_ms": retrieval_time * 1000,
                "generation_ms": generation_time * 1000,
                "total_ms": (retrieval_time + generation_time) * 1000
            },
            "debug_info": debug_info
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"测试结果: {test_case['id']}")
            print(f"{'='*70}")
            print(f"查询: {query}")
            print(f"\n回答:\n{answer}")
            print(f"\n评估: {'通过' if evaluation['passed'] else '失败'}")
            print(f"关键点覆盖率: {evaluation['key_points_coverage']['recall_score']:.1%}")
            print(f"检索耗时: {result['timing']['retrieval_ms']:.1f}ms")
            print(f"生成耗时: {result['timing']['generation_ms']:.1f}ms")
            print(f"总耗时: {result['timing']['total_ms']:.1f}ms")

        return result

    def run_baseline_comparison(self):
        """运行 Baseline 对比实验"""
        print("\n" + "="*70)
        print("Baseline 对比实验")
        print("="*70)

        # 选择一个测试用例进行对比
        test_case = TEST_CASES[2]  # 产品推荐案例

        print(f"\n测试用例: {test_case['query']}")

        # Baseline D: 混合检索（推荐方案）
        print("\n【方案 D】混合检索 (BM25 + Dense + RRF + Reranking)")
        result_hybrid = self.run_single_test(test_case, use_hybrid=True, verbose=True)

        # Baseline C: 仅 Dense 向量
        print("\n【方案 C】仅 Dense 向量检索")
        result_dense = self.run_single_test(test_case, use_hybrid=False, verbose=True)

        # 对比结果
        print("\n" + "="*70)
        print("对比结果")
        print("="*70)
        print(f"方案 D (混合检索):")
        print(f"  - 关键点覆盖率: {result_hybrid['evaluation']['key_points_coverage']['recall_score']:.1%}")
        print(f"  - 检索耗时: {result_hybrid['timing']['retrieval_ms']:.1f}ms")
        print(f"  - 总耗时: {result_hybrid['timing']['total_ms']:.1f}ms")

        print(f"\n方案 C (Dense检索):")
        print(f"  - 关键点覆盖率: {result_dense['evaluation']['key_points_coverage']['recall_score']:.1%}")
        print(f"  - 检索耗时: {result_dense['timing']['retrieval_ms']:.1f}ms")
        print(f"  - 总耗时: {result_dense['timing']['total_ms']:.1f}ms")

        return {
            "hybrid": result_hybrid,
            "dense_only": result_dense
        }

    def run_all_tests(self):
        """运行所有测试用例"""
        print("\n" + "="*70)
        print(f"运行所有测试用例 (共 {len(TEST_CASES)} 个)")
        print("="*70)

        results = []
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] {test_case['id']}: {test_case['query'][:40]}...")
            result = self.run_single_test(test_case, use_hybrid=True, verbose=False)
            results.append(result)

            # 打印简要结果
            passed = "✓" if result['evaluation']['passed'] else "✗"
            recall = result['evaluation']['key_points_coverage']['recall_score']
            print(f"  {passed} 关键点覆盖: {recall:.1%}, 耗时: {result['timing']['total_ms']:.0f}ms")

        # 统计结果
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[Dict]):
        """打印实验总结"""
        print("\n" + "="*70)
        print("实验总结")
        print("="*70)

        total = len(results)
        passed = sum(1 for r in results if r['evaluation']['passed'])
        avg_recall = sum(r['evaluation']['key_points_coverage']['recall_score'] for r in results) / total
        avg_retrieval = sum(r['timing']['retrieval_ms'] for r in results) / total
        avg_generation = sum(r['timing']['generation_ms'] for r in results) / total
        avg_total = sum(r['timing']['total_ms'] for r in results) / total

        print(f"测试用例数: {total}")
        print(f"通过数: {passed} ({passed/total:.1%})")
        print(f"平均关键点覆盖率: {avg_recall:.1%}")
        print(f"平均检索耗时: {avg_retrieval:.1f}ms")
        print(f"平均生成耗时: {avg_generation:.1f}ms")
        print(f"平均总耗时: {avg_total:.1f}ms")

        # 按类别统计
        print("\n按类别统计:")
        category_stats = {}
        for r in results:
            cat = r['category']
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0, "recalls": []}
            category_stats[cat]["total"] += 1
            category_stats[cat]["passed"] += 1 if r['evaluation']['passed'] else 0
            category_stats[cat]["recalls"].append(r['evaluation']['key_points_coverage']['recall_score'])

        for cat, stats in category_stats.items():
            avg_recall = sum(stats["recalls"]) / len(stats["recalls"])
            print(f"  {cat}: {stats['passed']}/{stats['total']} 通过, 平均覆盖率 {avg_recall:.1%}")

    def save_results(self, results: List[Dict], filename: str = None):
        """保存实验结果"""
        if filename is None:
            filename = f"experiment2_results_{self.experiment_timestamp}.json"

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n实验结果已保存: {output_path}")

    def run(self):
        """运行完整实验流程"""
        try:
            # 1. Baseline 对比
            print("\n【阶段 1】Baseline 对比实验")
            comparison_results = self.run_baseline_comparison()

            # 2. 运行所有测试用例
            print("\n【阶段 2】全量测试")
            all_results = self.run_all_tests()

            # 3. 保存结果
            final_results = {
                "experiment_info": {
                    "experiment_name": "实验2：企业图谱 + 业务文档的混合 RAG 检索",
                    "timestamp": self.experiment_timestamp,
                    "model": MODEL,
                    "total_documents": len(self.all_docs),
                    "company_documents": len(convert_all_companies_to_documents()),
                    "business_documents": len(FICTIONAL_DOCUMENTS)
                },
                "baseline_comparison": comparison_results,
                "all_test_results": all_results
            }

            self.save_results(final_results)

            print("\n" + "="*70)
            print("实验2完成!")
            print("="*70)

        except KeyboardInterrupt:
            print("\n\n实验被用户中断")
            sys.exit(0)
        except Exception as e:
            print(f"\n实验出错: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """主函数"""
    runner = Experiment2Runner()
    runner.run()


if __name__ == "__main__":
    main()
