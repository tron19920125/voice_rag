#!/usr/bin/env python3
"""
API Token 验证脚本
测试所有配置的 API token 是否可用
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import requests

# 加载环境变量
load_dotenv()

def test_qwen_api():
    """测试 Qwen LLM API"""
    print("\n" + "="*70)
    print("【测试 1】Qwen LLM API")
    print("="*70)

    api_base = os.getenv("QWEN_API_BASE")
    model = os.getenv("QWEN_MODEL")
    token = os.getenv("QWEN_TOKEN")

    if not all([api_base, model, token]):
        print("❌ 配置缺失：请检查 .env 中的 QWEN_API_BASE, QWEN_MODEL, QWEN_TOKEN")
        return False

    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"Token: {token[:10]}...")

    try:
        client = OpenAI(
            api_key=token,
            base_url=api_base
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": "请用一句话介绍西门子。"}
            ],
            max_tokens=50,
            extra_body={"enable_thinking": False}
        )

        answer = response.choices[0].message.content
        print(f"\n✅ Qwen API 测试成功")
        print(f"回答: {answer}")
        return True

    except Exception as e:
        print(f"\n❌ Qwen API 测试失败: {str(e)}")
        return False


def test_embedding_api():
    """测试 Embedding API"""
    print("\n" + "="*70)
    print("【测试 2】Embedding API")
    print("="*70)

    url = os.getenv("EMBEDDING_URL")
    model = os.getenv("EMBEDDING_MODEL")
    token = os.getenv("EMBEDDING_TOKEN")

    if not all([url, model, token]):
        print("❌ 配置缺失：请检查 .env 中的 EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_TOKEN")
        return False

    print(f"URL: {url}")
    print(f"Model: {model}")
    print(f"Token: {token[:10]}...")

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "input": ["西门子自动化解决方案"],
            "encoding_format": "float"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        embedding = data["data"][0]["embedding"]

        print(f"\n✅ Embedding API 测试成功")
        print(f"向量维度: {len(embedding)}")
        print(f"向量前5个值: {embedding[:5]}")
        return True

    except Exception as e:
        print(f"\n❌ Embedding API 测试失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text[:200]}")
        return False


def test_reranking_api():
    """测试 Reranking API"""
    print("\n" + "="*70)
    print("【测试 3】Reranking API")
    print("="*70)

    url = os.getenv("RERANKING_URL")
    model = os.getenv("RERANKING_MODEL")
    token = os.getenv("RERANKING_TOKEN")

    if not all([url, model, token]):
        print("❌ 配置缺失：请检查 .env 中的 RERANKING_URL, RERANKING_MODEL, RERANKING_TOKEN")
        return False

    print(f"URL: {url}")
    print(f"Model: {model}")
    print(f"Token: {token[:10]}...")

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "query": "生产效率提升方案",
            "documents": [
                "西门子 S7 系列 PLC 产品介绍",
                "工业自动化解决方案",
                "数字化转型案例"
            ],
            "top_n": 2
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        print(f"\n✅ Reranking API 测试成功")
        print(f"返回结果数: {len(results)}")
        if results:
            print("排序结果:")
            for i, item in enumerate(results, 1):
                print(f"  {i}. 文档索引={item.get('index')}, 分数={item.get('relevance_score', 0):.4f}")
        return True

    except Exception as e:
        print(f"\n❌ Reranking API 测试失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text[:200]}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("API Token 验证脚本")
    print("="*70)

    # 检查 .env 文件
    if not os.path.exists(".env"):
        print("\n❌ 错误：.env 文件不存在")
        print("请根据 .env.example 创建 .env 文件并配置相关 token")
        sys.exit(1)

    print("\n✓ .env 文件已找到")

    # 运行所有测试
    results = {
        "Qwen LLM": test_qwen_api(),
        "Embedding": test_embedding_api(),
        "Reranking": test_reranking_api()
    }

    # 输出总结
    print("\n" + "="*70)
    print("【测试总结】")
    print("="*70)

    for service, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{service:15s}: {status}")

    total_success = sum(results.values())
    total_tests = len(results)

    print(f"\n通过率: {total_success}/{total_tests}")

    if all(results.values()):
        print("\n🎉 所有 API 测试通过！可以开始实验开发。")
        sys.exit(0)
    else:
        print("\n⚠️  部分 API 测试失败，请检查配置和 token。")
        sys.exit(1)


if __name__ == "__main__":
    main()
