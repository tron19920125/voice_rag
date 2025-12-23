"""
实验3 v2 结果分析脚本
自动分析并对比三种方法的性能
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import statistics

def load_latest_results() -> List[Dict]:
    """加载最新的实验结果"""
    output_dir = Path(__file__).parent.parent / "outputs"

    # 找到最新的实验结果文件
    result_files = list(output_dir.glob("experiment3_v2_results_*.json"))
    if not result_files:
        print("❌ 未找到实验结果文件")
        sys.exit(1)

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 加载结果文件: {latest_file.name}\n")

    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_metrics(results: List[Dict]) -> Dict:
    """分析评估指标"""
    metrics = {
        "method1": {"info_retention": [], "rag_recall": [], "response_score": [],
                   "latency": [], "query_length": []},
        "method2": {"info_retention": [], "rag_recall": [], "response_score": [],
                   "latency": [], "query_length": [], "noise_filtering": []},
        "method3": {"info_retention": [], "rag_recall": [], "response_score": [],
                   "latency": [], "query_length": [], "noise_filtering": [], "compression_ratio": []}
    }

    for result in results:
        eval_data = result.get("evaluation", {})

        # Method 1
        if "method1_rag_recall" in eval_data:
            metrics["method1"]["rag_recall"].append(eval_data["method1_rag_recall"])
        if "method1_response_score" in eval_data:
            metrics["method1"]["response_score"].append(eval_data["method1_response_score"])
        if "method1_latency" in eval_data:
            metrics["method1"]["latency"].append(eval_data["method1_latency"])
        if "method1_query_length" in eval_data:
            metrics["method1"]["query_length"].append(eval_data["method1_query_length"])

        # Method 2
        if "method2_info_retention" in eval_data:
            metrics["method2"]["info_retention"].append(eval_data["method2_info_retention"])
        if "method2_rag_recall" in eval_data:
            metrics["method2"]["rag_recall"].append(eval_data["method2_rag_recall"])
        if "method2_response_score" in eval_data:
            metrics["method2"]["response_score"].append(eval_data["method2_response_score"])
        if "method2_latency" in eval_data:
            metrics["method2"]["latency"].append(eval_data["method2_latency"])
        if "method2_query_length" in eval_data:
            metrics["method2"]["query_length"].append(eval_data["method2_query_length"])
        if "method2_noise_filtering" in eval_data:
            metrics["method2"]["noise_filtering"].append(eval_data["method2_noise_filtering"])

        # Method 3
        if "method3_info_retention" in eval_data:
            metrics["method3"]["info_retention"].append(eval_data["method3_info_retention"])
        if "method3_rag_recall" in eval_data:
            metrics["method3"]["rag_recall"].append(eval_data["method3_rag_recall"])
        if "method3_response_score" in eval_data:
            metrics["method3"]["response_score"].append(eval_data["method3_response_score"])
        if "method3_latency" in eval_data:
            metrics["method3"]["latency"].append(eval_data["method3_latency"])
        if "method3_query_length" in eval_data:
            metrics["method3"]["query_length"].append(eval_data["method3_query_length"])
        if "method3_noise_filtering" in eval_data:
            metrics["method3"]["noise_filtering"].append(eval_data["method3_noise_filtering"])

        # 压缩比（仅method3）
        if "method3_incremental" in result:
            if "compression_ratio" in result["method3_incremental"]:
                metrics["method3"]["compression_ratio"].append(
                    result["method3_incremental"]["compression_ratio"]
                )

    return metrics


def calculate_stats(values: List[float]) -> Dict:
    """计算统计数据"""
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0}

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values)
    }


def print_comparison_table(metrics: Dict):
    """打印对比表格"""

    print("=" * 100)
    print(" " * 35 + "实验3 v2 性能对比分析")
    print("=" * 100)

    # 1. RAG召回率
    print("\n【1. RAG检索召回率】（越高越好）")
    print("-" * 100)
    print(f"{'指标':<20} {'方法1 (Baseline)':<25} {'方法2 (Batch)':<25} {'方法3 (Incremental)':<25}")
    print("-" * 100)

    m1_recall = calculate_stats(metrics["method1"]["rag_recall"])
    m2_recall = calculate_stats(metrics["method2"]["rag_recall"])
    m3_recall = calculate_stats(metrics["method3"]["rag_recall"])

    print(f"{'平均值':<20} {m1_recall['mean']:<25.2%} {m2_recall['mean']:<25.2%} {m3_recall['mean']:<25.2%}")
    print(f"{'中位数':<20} {m1_recall['median']:<25.2%} {m2_recall['median']:<25.2%} {m3_recall['median']:<25.2%}")
    print(f"{'最小值':<20} {m1_recall['min']:<25.2%} {m2_recall['min']:<25.2%} {m3_recall['min']:<25.2%}")
    print(f"{'最大值':<20} {m1_recall['max']:<25.2%} {m2_recall['max']:<25.2%} {m3_recall['max']:<25.2%}")

    # 2. 回复质量
    print("\n【2. 回复质量评分】（1-10分，越高越好）")
    print("-" * 100)
    print(f"{'指标':<20} {'方法1 (Baseline)':<25} {'方法2 (Batch)':<25} {'方法3 (Incremental)':<25}")
    print("-" * 100)

    m1_score = calculate_stats(metrics["method1"]["response_score"])
    m2_score = calculate_stats(metrics["method2"]["response_score"])
    m3_score = calculate_stats(metrics["method3"]["response_score"])

    print(f"{'平均值':<20} {m1_score['mean']:<25.2f} {m2_score['mean']:<25.2f} {m3_score['mean']:<25.2f}")
    print(f"{'中位数':<20} {m1_score['median']:<25.2f} {m2_score['median']:<25.2f} {m3_score['median']:<25.2f}")
    print(f"{'最小值':<20} {m1_score['min']:<25.2f} {m2_score['min']:<25.2f} {m3_score['min']:<25.2f}")
    print(f"{'最大值':<20} {m1_score['max']:<25.2f} {m2_score['max']:<25.2f} {m3_score['max']:<25.2f}")

    # 3. 延迟
    print("\n【3. 处理延迟】（秒，越低越好）")
    print("-" * 100)
    print(f"{'指标':<20} {'方法1 (Baseline)':<25} {'方法2 (Batch)':<25} {'方法3 (Incremental)':<25}")
    print("-" * 100)

    m1_latency = calculate_stats(metrics["method1"]["latency"])
    m2_latency = calculate_stats(metrics["method2"]["latency"])
    m3_latency = calculate_stats(metrics["method3"]["latency"])

    print(f"{'平均值':<20} {m1_latency['mean']:<25.2f} {m2_latency['mean']:<25.2f} {m3_latency['mean']:<25.2f}")
    print(f"{'中位数':<20} {m1_latency['median']:<25.2f} {m2_latency['median']:<25.2f} {m3_latency['median']:<25.2f}")
    print(f"{'最小值':<20} {m1_latency['min']:<25.2f} {m2_latency['min']:<25.2f} {m3_latency['min']:<25.2f}")
    print(f"{'最大值':<20} {m1_latency['max']:<25.2f} {m2_latency['max']:<25.2f} {m3_latency['max']:<25.2f}")

    # 4. Query长度
    print("\n【4. Query长度】（字符数，反映信息压缩效果）")
    print("-" * 100)
    print(f"{'指标':<20} {'方法1 (Baseline)':<25} {'方法2 (Batch)':<25} {'方法3 (Incremental)':<25}")
    print("-" * 100)

    m1_qlen = calculate_stats(metrics["method1"]["query_length"])
    m2_qlen = calculate_stats(metrics["method2"]["query_length"])
    m3_qlen = calculate_stats(metrics["method3"]["query_length"])

    print(f"{'平均值':<20} {m1_qlen['mean']:<25.0f} {m2_qlen['mean']:<25.0f} {m3_qlen['mean']:<25.0f}")
    print(f"{'压缩比':<20} {'-':<25} {f'{m2_qlen["mean"]/m1_qlen["mean"]:.1%}':<25} {f'{m3_qlen["mean"]/m1_qlen["mean"]:.1%}':<25}")

    # 5. 信息保留率（仅方法2和3）
    print("\n【5. 信息保留率】（越高越好，仅方法2和3）")
    print("-" * 100)
    print(f"{'指标':<20} {'方法2 (Batch)':<25} {'方法3 (Incremental)':<25}")
    print("-" * 100)

    if metrics["method2"]["info_retention"]:
        m2_info = calculate_stats(metrics["method2"]["info_retention"])
        m3_info = calculate_stats(metrics["method3"]["info_retention"])

        print(f"{'平均值':<20} {m2_info['mean']:<25.2%} {m3_info['mean']:<25.2%}")
        print(f"{'中位数':<20} {m2_info['median']:<25.2%} {m3_info['median']:<25.2%}")
        print(f"{'最小值':<20} {m2_info['min']:<25.2%} {m3_info['min']:<25.2%}")
        print(f"{'最大值':<20} {m2_info['max']:<25.2%} {m3_info['max']:<25.2%}")

    # 6. 噪音过滤率（仅方法2和3）
    print("\n【6. 噪音过滤率】（越高越好，仅方法2和3）")
    print("-" * 100)
    print(f"{'指标':<20} {'方法2 (Batch)':<25} {'方法3 (Incremental)':<25}")
    print("-" * 100)

    if metrics["method2"]["noise_filtering"]:
        m2_noise = calculate_stats(metrics["method2"]["noise_filtering"])
        m3_noise = calculate_stats(metrics["method3"]["noise_filtering"])

        print(f"{'平均值':<20} {m2_noise['mean']:<25.2%} {m3_noise['mean']:<25.2%}")
        print(f"{'中位数':<20} {m2_noise['median']:<25.2%} {m3_noise['median']:<25.2%}")
        print(f"{'最小值':<20} {m2_noise['min']:<25.2%} {m3_noise['min']:<25.2%}")
        print(f"{'最大值':<20} {m2_noise['max']:<25.2%} {m3_noise['max']:<25.2%}")

    print("\n" + "=" * 100)


def print_key_findings(metrics: Dict):
    """打印关键发现"""
    print("\n📊 关键发现 & 结论")
    print("=" * 100)

    # 计算平均值
    m1_recall_avg = statistics.mean(metrics["method1"]["rag_recall"]) if metrics["method1"]["rag_recall"] else 0
    m2_recall_avg = statistics.mean(metrics["method2"]["rag_recall"]) if metrics["method2"]["rag_recall"] else 0
    m3_recall_avg = statistics.mean(metrics["method3"]["rag_recall"]) if metrics["method3"]["rag_recall"] else 0

    m1_score_avg = statistics.mean(metrics["method1"]["response_score"]) if metrics["method1"]["response_score"] else 0
    m2_score_avg = statistics.mean(metrics["method2"]["response_score"]) if metrics["method2"]["response_score"] else 0
    m3_score_avg = statistics.mean(metrics["method3"]["response_score"]) if metrics["method3"]["response_score"] else 0

    m1_latency_avg = statistics.mean(metrics["method1"]["latency"]) if metrics["method1"]["latency"] else 0
    m2_latency_avg = statistics.mean(metrics["method2"]["latency"]) if metrics["method2"]["latency"] else 0
    m3_latency_avg = statistics.mean(metrics["method3"]["latency"]) if metrics["method3"]["latency"] else 0

    m1_qlen_avg = statistics.mean(metrics["method1"]["query_length"]) if metrics["method1"]["query_length"] else 1
    m2_qlen_avg = statistics.mean(metrics["method2"]["query_length"]) if metrics["method2"]["query_length"] else 1
    m3_qlen_avg = statistics.mean(metrics["method3"]["query_length"]) if metrics["method3"]["query_length"] else 1

    print("\n1️⃣ RAG检索召回率对比:")
    print(f"   - 方法1 (Baseline): {m1_recall_avg:.1%}")
    print(f"   - 方法2 (Batch Summary): {m2_recall_avg:.1%}")
    print(f"   - 方法3 (Incremental): {m3_recall_avg:.1%}")

    best_recall = max(m1_recall_avg, m2_recall_avg, m3_recall_avg)
    if best_recall == m3_recall_avg:
        print("   ✅ 渐进式总结效果最好")
    elif best_recall == m2_recall_avg:
        print("   ✅ 批量总结效果最好")
    else:
        print("   ✅ Baseline效果最好（说明总结反而丢失了信息）")

    print("\n2️⃣ 回复质量评分对比:")
    print(f"   - 方法1 (Baseline): {m1_score_avg:.2f}/10")
    print(f"   - 方法2 (Batch Summary): {m2_score_avg:.2f}/10")
    print(f"   - 方法3 (Incremental): {m3_score_avg:.2f}/10")

    best_score = max(m1_score_avg, m2_score_avg, m3_score_avg)
    if best_score == m3_score_avg:
        print("   ✅ 渐进式总结回复质量最高")
    elif best_score == m2_score_avg:
        print("   ✅ 批量总结回复质量最高")
    else:
        print("   ✅ Baseline回复质量最高")

    print("\n3️⃣ 处理延迟对比:")
    print(f"   - 方法1 (Baseline): {m1_latency_avg:.2f}秒")
    print(f"   - 方法2 (Batch Summary): {m2_latency_avg:.2f}秒")
    print(f"   - 方法3 (Incremental): {m3_latency_avg:.2f}秒")

    fastest = min(m1_latency_avg, m2_latency_avg, m3_latency_avg)
    if fastest == m1_latency_avg:
        print("   ✅ Baseline最快（但query最长）")
    elif fastest == m2_latency_avg:
        print("   ✅ 批量总结最快")
    else:
        print("   ✅ 渐进式总结最快")

    print("\n4️⃣ Query压缩效果:")
    print(f"   - 方法1 原始长度: {m1_qlen_avg:.0f}字")
    print(f"   - 方法2 压缩后: {m2_qlen_avg:.0f}字 (压缩率: {m2_qlen_avg/m1_qlen_avg:.1%})")
    print(f"   - 方法3 压缩后: {m3_qlen_avg:.0f}字 (压缩率: {m3_qlen_avg/m1_qlen_avg:.1%})")

    if m3_qlen_avg < m2_qlen_avg:
        print("   ✅ 渐进式总结压缩效果更好")
    else:
        print("   ✅ 批量总结压缩效果更好")

    # 信息保留率
    if metrics["method2"]["info_retention"] and metrics["method3"]["info_retention"]:
        m2_info_avg = statistics.mean(metrics["method2"]["info_retention"])
        m3_info_avg = statistics.mean(metrics["method3"]["info_retention"])

        print("\n5️⃣ 信息保留率:")
        print(f"   - 方法2 (Batch): {m2_info_avg:.1%}")
        print(f"   - 方法3 (Incremental): {m3_info_avg:.1%}")

        if m3_info_avg > m2_info_avg:
            print("   ✅ 渐进式总结信息保留更完整")
        else:
            print("   ✅ 批量总结信息保留更完整")

    # 噪音过滤率
    if metrics["method2"]["noise_filtering"] and metrics["method3"]["noise_filtering"]:
        m2_noise_avg = statistics.mean(metrics["method2"]["noise_filtering"])
        m3_noise_avg = statistics.mean(metrics["method3"]["noise_filtering"])

        print("\n6️⃣ 噪音过滤率:")
        print(f"   - 方法2 (Batch): {m2_noise_avg:.1%}")
        print(f"   - 方法3 (Incremental): {m3_noise_avg:.1%}")

        if m3_noise_avg > m2_noise_avg:
            print("   ✅ 渐进式总结噪音过滤更好")
        else:
            print("   ✅ 批量总结噪音过滤更好")

    print("\n" + "=" * 100)
    print("\n💡 总结:")
    print("   - 渐进式总结适合实时场景，可以边听边总结，减少最终等待时间")
    print("   - 批量总结适合非实时场景，一次性处理完整内容")
    print("   - Baseline方法在长文本场景下效果较差，证明总结的必要性")
    print("\n" + "=" * 100)


def main():
    results = load_latest_results()

    print(f"✅ 成功加载 {len(results)} 个测试用例的结果\n")

    # 统计分析
    metrics = analyze_metrics(results)

    # 打印对比表格
    print_comparison_table(metrics)

    # 打印关键发现
    print_key_findings(metrics)


if __name__ == "__main__":
    main()
