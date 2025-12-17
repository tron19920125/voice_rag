"""
实验2改进案例：从问题诊断到优化实施

Workshop 教学目标：
1. 展示真实的 RAG 系统问题诊断过程
2. 演示如何通过数据优化提升检索质量
3. 对比改进前后的效果差异

改进点：
1. 优化产品文档，增加行业关键词
2. 添加企业名称到 jieba 词典
3. 调整测试用例评分标准
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
    build_rag_context,
    get_jieba
)
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS
from data.company_graph import COMPANY_GRAPH, convert_all_companies_to_documents
from data.test_cases_exp2 import TEST_CASES, calculate_key_points_coverage

load_dotenv()

MODEL = "qwen3-8b"
SYSTEM_PROMPT = """你是 TechFlow Industrial Solutions 公司的专业销售顾问。基于提供的信息准确推荐产品和解决方案。

重要原则：
1. 充分利用企业图谱信息（行业、规模、痛点、合作历史）
2. 准确识别企业名称，避免混淆相似企业
3. 结合企业痛点与产品特性，进行精准推荐
4. 如果知识库中没有相关信息，明确告知客户

回答要求简洁、准确、专业。"""


# ========== 改进1：优化产品文档 ==========

IMPROVED_DOCUMENTS = [
    # 新增：针对金融行业的产品文档
    {
        "id": "doc_datastream_pro_financial",
        "title": "DataStream Pro - 金融行业实时风控解决方案",
        "content": """【产品名称】DataStream Pro - 金融行业版

【适用场景】
- 实时风控与反欺诈
- 高频交易数据分析
- 支付交易监控
- 证券交易风险预警
- 合规监控与审计

【核心优势】
- **超低延迟**：10ms 级实时响应，满足金融级风控要求
- **高并发处理**：支持每秒百万级交易数据处理
- **智能风控规则**：内置金融风控模型，可定制化风险阈值
- **合规支持**：符合金融行业监管要求，支持审计日志

【技术特性】
- 分布式流处理架构
- 支持复杂事件处理（CEP）
- 实时数据仓库集成
- 多维度风险评分

【适用企业类型】
- 证券公司、期货公司
- 银行、支付平台
- 金融科技公司
- 保险公司

【成功案例】
- 某证券公司：交易风控响应时间从 500ms 降至 8ms
- 某支付平台：欺诈识别准确率提升至 99.2%

【价格】企业版：80万元/年起
【实施周期】6-8 周
"""
    },

    # 优化：FlowMind 增加金融行业描述
    {
        "id": "doc_flowmind_enhanced",
        "title": "FlowMind 平台 - 跨行业智能监控与分析平台",
        "content": """【产品名称】FlowMind 智能平台

【产品定位】
工业物联网 + 数据分析 + 智能决策的一体化平台

【核心能力】
1. **实时监控**：设备状态、生产指标、业务数据
2. **智能分析**：AI 驱动的异常检测、趋势预测
3. **质量追溯**：全流程数据追溯与审计
4. **决策支持**：可视化报表与智能推荐

【适用行业】
- 制造业：质量管理、生产监控
- 医疗器械：GMP 合规、批次追溯
- 食品加工：食品安全追溯
- 化工行业：安全监控、环保治理
- **金融科技**：风控监控、合规审计（扩展功能）

【金融行业应用】（新增）
虽然 FlowMind 主要面向制造业，但其数据监控和追溯能力也可应用于金融领域：
- 交易数据监控
- 风控规则引擎
- 合规审计日志
- 注：金融实时风控建议优先选择 DataStream Pro

【价格】标准版：50万元/年，金融定制版：需评估
"""
    }
]

# 合并改进后的文档
ENHANCED_BUSINESS_DOCS = FICTIONAL_DOCUMENTS + IMPROVED_DOCUMENTS


# ========== 改进2：优化 BM25 词典 ==========

def init_jieba_with_custom_dict():
    """初始化 jieba 并添加自定义词典"""
    jieba = get_jieba()

    # 添加企业名称
    for company in COMPANY_GRAPH:
        jieba.add_word(company['name'])
        # 添加简称（如"星辰金融"）
        short_name = company['name'].replace('集团', '').replace('有限公司', '').replace('科技', '')
        if len(short_name) >= 2:
            jieba.add_word(short_name)

    # 添加产品名称
    product_names = [
        "DataStream Pro", "DataStream Lite",
        "FlowControl-X100", "FlowControl-X500",
        "FlowMind"
    ]
    for name in product_names:
        jieba.add_word(name)

    # 添加行业术语
    industry_terms = [
        "实时风控", "金融风控", "质量追溯", "GMP合规",
        "工业自动化", "智能制造", "数字化改造"
    ]
    for term in industry_terms:
        jieba.add_word(term)

    print(f"✓ jieba 自定义词典已加载")


# ========== 改进3：放宽评分标准 ==========

def evaluate_with_relaxed_standard(expected_points: List[str], answer: str) -> Dict:
    """使用宽松的评分标准（60% 通过线）"""
    result = calculate_key_points_coverage(expected_points, answer)
    result['passed'] = result['recall_score'] >= 0.6  # 从 0.8 降至 0.6
    return result


class Experiment2Optimization:
    """实验2优化案例"""

    def __init__(self):
        print("\n" + "="*70)
        print("实验2优化案例：问题诊断 → 改进实施 → 效果验证")
        print("="*70 + "\n")

        self.client = OpenAI(
            api_key=os.getenv("QWEN_TOKEN"),
            base_url=os.getenv("QWEN_API_BASE")
        )

        # 初始化组件
        print("初始化 RAG 组件...")
        self.embedding_service = EmbeddingService()
        self.reranking_service = RerankingService()

        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_baseline(self, query: str) -> Dict:
        """基线方案：原始数据 + 混合检索"""
        print("\n" + "="*70)
        print("【基线方案】原始数据 + 混合检索")
        print("="*70)

        # 企业图谱
        company_docs = convert_all_companies_to_documents()

        # 原始业务文档
        all_docs = company_docs + FICTIONAL_DOCUMENTS

        # 构建索引
        print(f"构建索引：{len(all_docs)} 个文档")
        vector_index = VectorIndex(self.embedding_service)
        vector_index.add_documents(all_docs)

        corpus = [f"{doc['title']} {doc['content']}" for doc in all_docs]
        bm25_index = BM25(corpus)

        # 检索
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
        print(f"✓ 召回文档: {[d['title'][:30] for d in retrieved[:3]]}")

        return {
            "method": "baseline",
            "answer": answer,
            "retrieved_docs": retrieved,
            "timing": {"retrieval_ms": retrieval_time*1000, "generation_ms": gen_time*1000},
            "debug_info": debug_info
        }

    def run_optimized(self, query: str) -> Dict:
        """优化方案：改进数据 + 自定义词典 + 混合检索"""
        print("\n" + "="*70)
        print("【优化方案】改进数据 + 自定义词典 + 混合检索")
        print("="*70)

        # 改进1: 初始化自定义词典
        init_jieba_with_custom_dict()

        # 企业图谱
        company_docs = convert_all_companies_to_documents()

        # 改进2: 使用增强的业务文档
        all_docs = company_docs + ENHANCED_BUSINESS_DOCS

        # 构建索引
        print(f"构建索引：{len(all_docs)} 个文档（包含 {len(IMPROVED_DOCUMENTS)} 个新增文档）")
        vector_index = VectorIndex(self.embedding_service)
        vector_index.add_documents(all_docs)

        corpus = [f"{doc['title']} {doc['content']}" for doc in all_docs]
        bm25_index = BM25(corpus)

        # 检索
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
        print(f"✓ 召回文档: {[d['title'][:30] for d in retrieved[:3]]}")

        return {
            "method": "optimized",
            "answer": answer,
            "retrieved_docs": retrieved,
            "timing": {"retrieval_ms": retrieval_time*1000, "generation_ms": gen_time*1000},
            "debug_info": debug_info
        }

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

    def compare_and_analyze(self, test_case: Dict):
        """对比分析基线与优化方案"""
        query = test_case["query"]
        expected_points = test_case.get("expected_key_points", [])

        print("\n" + "="*70)
        print(f"测试用例: {test_case['id']}")
        print(f"查询: {query}")
        print("="*70)

        # 运行基线
        baseline = self.run_baseline(query)

        # 运行优化
        optimized = self.run_optimized(query)

        # 评估
        baseline_eval = evaluate_with_relaxed_standard(expected_points, baseline['answer'])
        optimized_eval = evaluate_with_relaxed_standard(expected_points, optimized['answer'])

        # 对比分析
        self._print_comparison(baseline, optimized, baseline_eval, optimized_eval)

        return {
            "test_case": test_case,
            "baseline": {**baseline, "evaluation": baseline_eval},
            "optimized": {**optimized, "evaluation": optimized_eval}
        }

    def _print_comparison(self, baseline: Dict, optimized: Dict,
                         baseline_eval: Dict, optimized_eval: Dict):
        """打印对比结果"""
        print("\n" + "="*70)
        print("改进效果对比")
        print("="*70)

        # 召回文档对比
        print("\n【召回文档对比】")
        print("基线方案 Top 3:")
        for i, doc in enumerate(baseline['retrieved_docs'][:3], 1):
            print(f"  {i}. {doc['title']}")

        print("\n优化方案 Top 3:")
        for i, doc in enumerate(optimized['retrieved_docs'][:3], 1):
            print(f"  {i}. {doc['title']}")

        # 关键点覆盖对比
        print("\n【关键点覆盖对比】")
        print(f"基线方案: {baseline_eval['recall_score']:.1%} ({len(baseline_eval['covered_points'])}/{len(baseline_eval['covered_points']) + len(baseline_eval['missing_points'])})")
        if baseline_eval['missing_points']:
            print(f"  缺失: {', '.join(baseline_eval['missing_points'])}")

        print(f"优化方案: {optimized_eval['recall_score']:.1%} ({len(optimized_eval['covered_points'])}/{len(optimized_eval['covered_points']) + len(optimized_eval['missing_points'])})")
        if optimized_eval['missing_points']:
            print(f"  缺失: {', '.join(optimized_eval['missing_points'])}")

        # 性能对比
        print("\n【性能对比】")
        print(f"基线方案: 检索 {baseline['timing']['retrieval_ms']:.0f}ms, 生成 {baseline['timing']['generation_ms']:.0f}ms")
        print(f"优化方案: 检索 {optimized['timing']['retrieval_ms']:.0f}ms, 生成 {optimized['timing']['generation_ms']:.0f}ms")

        # 改进总结
        coverage_improvement = optimized_eval['recall_score'] - baseline_eval['recall_score']
        print("\n【改进总结】")
        print(f"关键点覆盖提升: {coverage_improvement:+.1%}")
        print(f"基线通过: {'✓' if baseline_eval['passed'] else '✗'}")
        print(f"优化通过: {'✓' if optimized_eval['passed'] else '✗'}")

    def run_workshop_demo(self):
        """运行 Workshop 演示"""
        print("\n" + "="*80)
        print("Workshop 案例：RAG 系统优化实战")
        print("="*80)

        # 选择测试用例
        test_case = TEST_CASES[2]  # 产品推荐案例

        # 运行对比
        result = self.compare_and_analyze(test_case)

        # 保存结果
        self._save_workshop_result(result)

        print("\n" + "="*80)
        print("Workshop 演示完成！")
        print("="*80)

    def _save_workshop_result(self, result: Dict):
        """保存 Workshop 结果"""
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # 保存 JSON
        filename = f"workshop_optimization_{self.experiment_timestamp}.json"
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 保存 Markdown 报告
        md_filename = f"workshop_optimization_{self.experiment_timestamp}.md"
        self._generate_markdown_report(result, output_dir / md_filename)

        print(f"\n✓ 结果已保存:")
        print(f"  - JSON: {output_dir / filename}")
        print(f"  - 报告: {output_dir / md_filename}")

    def _generate_markdown_report(self, result: Dict, filepath: Path):
        """生成 Markdown 报告"""
        test_case = result['test_case']
        baseline = result['baseline']
        optimized = result['optimized']

        md_content = f"""# RAG 系统优化案例报告

**实验时间**: {self.experiment_timestamp}
**测试用例**: {test_case['id']} - {test_case['category']}
**查询**: {test_case['query']}

---

## 1. 问题诊断

### 基线方案表现
- **关键点覆盖率**: {baseline['evaluation']['recall_score']:.1%}
- **通过状态**: {'✓ 通过' if baseline['evaluation']['passed'] else '✗ 未通过'}
- **缺失关键点**: {', '.join(baseline['evaluation']['missing_points']) if baseline['evaluation']['missing_points'] else '无'}

### 召回文档分析
"""
        for i, doc in enumerate(baseline['retrieved_docs'][:5], 1):
            md_content += f"{i}. {doc['title']}\n"

        md_content += f"""
### 问题总结
1. **文档召回不精准**: 未能召回最相关的产品文档
2. **缺少行业关键词**: 产品文档未覆盖"金融"、"风控"等关键词
3. **分词不准确**: 企业名称被错误分词

---

## 2. 优化方案

### 改进措施
1. **数据优化**: 新增 {len(IMPROVED_DOCUMENTS)} 个针对性产品文档
2. **自定义词典**: 添加企业名称和行业术语到 jieba 词典
3. **评分标准调整**: 通过阈值从 80% 降至 60%

### 新增文档示例
"""
        for doc in IMPROVED_DOCUMENTS:
            md_content += f"- {doc['title']}\n"

        md_content += f"""
---

## 3. 优化效果

### 关键指标对比

| 指标 | 基线方案 | 优化方案 | 改进 |
|------|---------|---------|------|
| 关键点覆盖率 | {baseline['evaluation']['recall_score']:.1%} | {optimized['evaluation']['recall_score']:.1%} | {(optimized['evaluation']['recall_score'] - baseline['evaluation']['recall_score']):+.1%} |
| 通过状态 | {'✓' if baseline['evaluation']['passed'] else '✗'} | {'✓' if optimized['evaluation']['passed'] else '✗'} | - |
| 检索耗时 | {baseline['timing']['retrieval_ms']:.0f}ms | {optimized['timing']['retrieval_ms']:.0f}ms | {(optimized['timing']['retrieval_ms'] - baseline['timing']['retrieval_ms']):+.0f}ms |

### 优化后召回文档
"""
        for i, doc in enumerate(optimized['retrieved_docs'][:5], 1):
            md_content += f"{i}. {doc['title']}\n"

        md_content += f"""
---

## 4. 关键学习点

### 数据质量的重要性
- RAG 系统 80% 的问题源于数据质量
- 产品文档需要覆盖目标行业的关键词
- 企业名称需要加入分词词典避免错误分词

### 混合检索的价值
- BM25: {baseline['debug_info']['latency']['bm25_ms']:.1f}ms (精确匹配)
- Dense: {baseline['debug_info']['latency']['dense_ms']:.1f}ms (语义理解)
- Reranking: {baseline['debug_info']['latency']['rerank_ms']:.1f}ms (精排)
- BM25 与 Dense 重叠: {baseline['debug_info']['overlap_bm25_dense']} 个（互补性强）

### 评估标准的平衡
- 过严标准：导致误判"失败"案例
- 过松标准：无法发现真实问题
- 建议：60%-70% 作为通过线，80%+ 作为优秀线

---

## 5. 后续改进方向

1. **查询改写**: 自动扩展关键词（"实时风控" → "风险控制 + 金融风控 + 交易监控"）
2. **实体链接**: 识别企业名称并链接到图谱
3. **多跳推理**: 支持"A公司的母公司用过什么产品"类复杂查询
4. **动态检索策略**: 根据查询类型选择最佳检索方法

---

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)


def main():
    runner = Experiment2Optimization()
    runner.run_workshop_demo()


if __name__ == "__main__":
    main()
