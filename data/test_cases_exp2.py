"""
实验2测试用例数据
混合 RAG 检索与企业图谱融合测试

基于实验2设计文档的第3节：测试用例设计
"""

from typing import List, Dict

# ========== 测试用例定义 ==========

TEST_CASES: List[Dict] = [
    # 测试用例 1：企业背景查询（图谱检索）
    {
        "id": "fusion_test_001",
        "category": "企业背景查询",
        "query": "鼎盛科技是做什么的？公司规模多大？",
        "expected_sources": {
            "company_profile": True,  # 企业档案
            "business_docs": False     # 不需要业务文档
        },
        "expected_key_points": [
            "智能制造",
            "2015",
            "5000万",
            "500人",
            "李明"
        ],
        "expected_bm25_advantage": True,  # BM25 应精确匹配 "鼎盛科技"
        "description": "测试企业基本信息查询，验证 BM25 对企业名称的精确匹配能力"
    },

    # 测试用例 2：合作历史查询（图谱 + 关系）
    {
        "id": "fusion_test_002",
        "category": "合作历史查询",
        "query": "鼎盛科技之前有没有用过我们的产品？效果怎么样？",
        "expected_sources": {
            "project_history": True,   # 合作案例
            "company_profile": True,   # 企业基本信息
            "business_docs": True      # 产品文档
        },
        "expected_key_points": [
            "2023",
            "DataStream Lite",
            "工厂数字化改造",
            "35%"
        ],
        "description": "测试合作历史查询，验证多源融合能力"
    },

    # 测试用例 3：产品推荐（图谱 + 文档融合）
    {
        "id": "fusion_test_003",
        "category": "产品推荐",
        "query": "星辰金融集团想做实时风控，应该推荐哪个产品？",
        "expected_sources": {
            "company_profile": True,     # 了解客户（金融行业）
            "company_needs": True,       # 痛点（实时风控延迟高）
            "business_docs": True        # 产品文档
        },
        "expected_key_points": [
            "金融",
            "DataStream Pro",
            "实时",
            "风控"
        ],
        "expected_fusion_value": True,  # 需要融合企业信息 + 产品信息
        "description": "测试产品推荐场景，验证企业痛点与产品特性匹配"
    },

    # 测试用例 4：关系查询（图谱推理）
    {
        "id": "fusion_test_004",
        "category": "关系查询",
        "query": "鼎盛科技的母公司是谁？投资方有哪些？",
        "expected_sources": {
            "company_relations": True,  # 企业关系
            "company_profile": False
        },
        "expected_key_points": [
            "鼎盛集团",
            "红杉资本",
            "IDG"
        ],
        "expected_bm25_advantage": True,  # 需要精确匹配公司名称
        "description": "测试企业关系查询，验证图谱关系信息检索"
    },

    # 测试用例 5：竞品客户识别（陷阱测试）
    {
        "id": "fusion_test_005",
        "category": "竞品客户识别",
        "query": "恒通电子商务之前有没有合作过？",
        "expected_sources": {
            "project_history": True,
            "company_relations": True,
            "company_needs": True
        },
        "expected_behavior": "识别出该客户使用竞品，谨慎推荐或说明差异化优势",
        "expected_key_points": [
            "恒通电子商务",
            "竞品",
            "未合作"
        ],
        "description": "测试竞品客户识别，验证 RAG 能否识别敏感信息"
    },

    # 测试用例 6：混合检索效果验证
    {
        "id": "fusion_test_006",
        "category": "混合检索效果验证",
        "query": "DataStream Pro 适合哪些行业？",
        "baseline_a": "仅 BM25 检索",
        "baseline_b": "仅 Dense 向量检索",
        "proposed": "BM25 + Dense 混合",
        "expected_result": "混合检索应召回更全面（产品文档 + 行业案例 + 客户档案）",
        "expected_key_points": [
            "DataStream Pro",
            "金融",
            "实时"
        ],
        "description": "对比测试，验证混合检索相比单一检索的优势"
    },

    # 测试用例 7：痛点匹配推荐
    {
        "id": "fusion_test_007",
        "category": "痛点匹配推荐",
        "query": "云帆新能源有生产数据孤岛的问题，推荐什么方案？",
        "expected_sources": {
            "company_needs": True,
            "business_docs": True
        },
        "expected_key_points": [
            "云帆新能源",
            "数据孤岛",
            "FlowMind"
        ],
        "description": "测试痛点精准匹配，验证语义理解能力"
    },

    # 测试用例 8：多企业对比查询
    {
        "id": "fusion_test_008",
        "category": "多企业对比查询",
        "query": "明远物流科技和晨曦食品加工都用过什么产品？",
        "expected_sources": {
            "project_history": True,
            "company_profile": True
        },
        "expected_key_points": [
            "明远物流科技",
            "FlowControl-X100",
            "晨曦食品加工"
        ],
        "description": "测试多实体识别与信息聚合"
    }
]


# ========== 评分函数 ==========

def calculate_key_points_coverage(expected_key_points: List[str],
                                   actual_answer: str) -> Dict:
    """
    计算关键点覆盖率

    Args:
        expected_key_points: 期望包含的关键点列表
        actual_answer: 实际生成的回答

    Returns:
        {
            "recall_score": 0-1,  # 召回率
            "covered_points": [],  # 覆盖的关键点
            "missing_points": []   # 遗漏的关键点
        }
    """
    covered = []
    missing = []

    for point in expected_key_points:
        if point in actual_answer:
            covered.append(point)
        else:
            missing.append(point)

    recall_score = len(covered) / len(expected_key_points) if expected_key_points else 0

    return {
        "recall_score": recall_score,
        "covered_points": covered,
        "missing_points": missing
    }


def check_source_coverage(expected_sources: Dict[str, bool],
                          retrieved_docs: List[Dict]) -> Dict:
    """
    检查是否召回了期望的数据源

    Args:
        expected_sources: {"company_profile": True, "business_docs": False}
        retrieved_docs: 检索到的文档列表

    Returns:
        {
            "passed": True/False,
            "actual_sources": ["company_profile", "project_history"],
            "missing_sources": []
        }
    """
    actual_sources = set([doc.get('type') for doc in retrieved_docs])
    required_sources = [k for k, v in expected_sources.items() if v]
    missing_sources = [s for s in required_sources if s not in actual_sources]

    return {
        "passed": len(missing_sources) == 0,
        "actual_sources": list(actual_sources),
        "missing_sources": missing_sources
    }


def evaluate_test_case(test_case: Dict, actual_answer: str, retrieved_docs: List[Dict]) -> Dict:
    """
    评估单个测试用例

    Args:
        test_case: 测试用例
        actual_answer: 实际生成的回答
        retrieved_docs: 检索到的文档

    Returns:
        评分结果字典
    """
    # 关键点覆盖
    key_points_result = calculate_key_points_coverage(
        test_case.get("expected_key_points", []),
        actual_answer
    )

    # 数据源覆盖
    source_result = check_source_coverage(
        test_case.get("expected_sources", {}),
        retrieved_docs
    )

    # 综合判断
    passed = (
        key_points_result["recall_score"] >= 0.8 and
        source_result["passed"]
    )

    return {
        "test_id": test_case["id"],
        "category": test_case["category"],
        "passed": passed,
        "key_points_coverage": key_points_result,
        "source_coverage": source_result,
        "answer_length": len(actual_answer)
    }


# ========== 统计信息 ==========

def get_test_cases_summary():
    """获取测试用例统计信息"""
    categories = {}
    for tc in TEST_CASES:
        cat = tc["category"]
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_test_cases": len(TEST_CASES),
        "categories": categories,
        "has_bm25_advantage_tests": sum(1 for tc in TEST_CASES if tc.get("expected_bm25_advantage")),
        "has_fusion_value_tests": sum(1 for tc in TEST_CASES if tc.get("expected_fusion_value"))
    }


if __name__ == "__main__":
    import json
    summary = get_test_cases_summary()
    print("=" * 60)
    print("实验2测试用例统计")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n测试用例列表:")
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"{i}. [{tc['id']}] {tc['category']}: {tc['query']}")
    print("=" * 60)
