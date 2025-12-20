#!/usr/bin/env python3
"""
Experiment 3: Local vLLM vs Cloud API Performance Comparison
本地 vLLM vs 云端 API 性能对比

目标：
1. 对比本地部署 LLM (vLLM + Qwen3-8B) vs 云端 API 的性能
2. 测试延迟、吞吐量等关键指标
3. 验证本地部署的可行性

配置：
- 本地: vLLM 0.11.0 + Qwen3-8B on V100 x2 (tensor parallelism)
- 云端: 通义千问 API (qwen3-8b)
- 思考模式: 两边都禁用 (enable_thinking=false)

测试方法：
- 使用 3 个简单问题测试基础性能
- 测量端到端延迟和 token 使用量
- 对比平均性能指标
"""

import json
import time
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 测试问题
test_questions = [
    "你好，请简单介绍一下你自己。",
    "1+1等于几？",
    "请解释一下什么是机器学习。",
]

def test_model(client, model, name, is_cloud=False):
    """测试模型性能"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] 问题: {question}")

        try:
            start_time = time.time()

            # 构建消息
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
                {"role": "user", "content": question}
            ]

            # 云端 API 使用 extra_body 禁用思考模式
            if is_cloud:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.1,
                    extra_body={"enable_thinking": False}
                )
            else:
                # 本地 vLLM 使用 chat_template_kwargs 禁用思考模式
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )

            latency = time.time() - start_time
            answer = response.choices[0].message.content

            result = {
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "latency": latency,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }

            results.append(result)

            print(f"✅ 延迟: {latency:.2f}s")
            print(f"📊 Tokens: {response.usage.total_tokens}")
            print(f"📝 回答: {answer[:100]}...")

        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({
                "question": question,
                "error": str(e)
            })

    return results


def main():
    print("\n" + "="*80)
    print("Experiment 3: Local vLLM vs Cloud API Performance Test")
    print("="*80)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "cloud": None,
        "local": None
    }

    # 1. 测试云端 API
    print("\n\n" + "="*80)
    print("Part 1: Testing Cloud API")
    print("="*80)

    try:
        cloud_client = OpenAI(
            api_key=os.getenv("QWEN_TOKEN"),
            base_url=os.getenv("QWEN_API_BASE")
        )
        cloud_model = os.getenv("QWEN_MODEL")

        cloud_results = test_model(cloud_client, cloud_model, "Cloud API", is_cloud=True)
        all_results["cloud"] = cloud_results

    except Exception as e:
        print(f"\n❌ 云端 API 测试失败: {e}")

    # 2. 测试本地 vLLM
    print("\n\n" + "="*80)
    print("Part 2: Testing Local vLLM")
    print("="*80)

    try:
        local_client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        local_model = "Qwen/Qwen3-8B"

        local_results = test_model(local_client, local_model, "Local vLLM", is_cloud=False)
        all_results["local"] = local_results

    except Exception as e:
        print(f"\n❌ 本地 vLLM 测试失败: {e}")

    # 3. 对比分析
    print("\n\n" + "="*80)
    print("📊 Performance Comparison")
    print("="*80)

    if all_results["cloud"] and all_results["local"]:
        cloud_valid = [r for r in all_results["cloud"] if "latency" in r]
        local_valid = [r for r in all_results["local"] if "latency" in r]

        if cloud_valid and local_valid:
            cloud_avg = sum(r["latency"] for r in cloud_valid) / len(cloud_valid)
            local_avg = sum(r["latency"] for r in local_valid) / len(local_valid)

            print(f"\n⏱️  Average Latency:")
            print(f"   Cloud API:  {cloud_avg:.3f}s")
            print(f"   Local vLLM: {local_avg:.3f}s")
            print(f"   Speedup:    {cloud_avg / local_avg:.2f}x")

            cloud_tokens = sum(r["tokens"]["total"] for r in cloud_valid) / len(cloud_valid)
            local_tokens = sum(r["tokens"]["total"] for r in local_valid) / len(local_valid)

            print(f"\n📊 Average Tokens:")
            print(f"   Cloud API:  {cloud_tokens:.0f}")
            print(f"   Local vLLM: {local_tokens:.0f}")

    # 4. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/experiment3_simple_{timestamp}.json"

    os.makedirs("outputs", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n\n💾 Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
