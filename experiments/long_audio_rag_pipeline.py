"""
长语音输入的RAG处理流程
包含：结构化提取、多查询检索、响应生成
"""

import json
import time
from typing import Dict, List, Optional
from openai import OpenAI
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_utils import VectorIndex, EmbeddingService, RerankService, BM25Index


# ========== Prompt模板 ==========

SUMMARIZATION_PROMPT = """你是一个专业的语音助手，用户刚刚进行了一段较长的语音输入。
请分析用户的输入，提取关键信息。

用户输入：
{user_input}

请按照以下格式返回JSON（不要有markdown代码块标记）：
{{
  "main_intent": "主要意图（产品咨询/技术咨询/客户查询/需求描述等）",
  "key_points": ["关键点1", "关键点2", "关键点3"],
  "entities": ["提到的实体1", "实体2"],
  "constraints": ["预算/时间/技术约束"],
  "is_multi_question": false,
  "concise_query": "简化后的查询语句（50-100字）"
}}

注意：
1. 过滤口语化表达（嗯、那个、就是、怎么说呢等）
2. 保留核心信息（数字、专有名词、关键需求）
3. 如果有多个独立问题，设置is_multi_question为true，并在key_points中列出
4. concise_query要简洁但包含所有关键信息
"""

MULTI_QUERY_DECOMPOSITION_PROMPT = """用户提出了一个包含多个问题的查询。
请将其分解为多个独立的子查询。

用户输入：
{user_input}

关键点：
{key_points}

请返回JSON格式（不要有markdown代码块标记）：
{{
  "sub_queries": [
    "子查询1",
    "子查询2",
    "子查询3"
  ]
}}

要求：
1. 每个子查询要独立完整
2. 保留原文中的关键实体和约束
3. 按重要性排序
"""

RESPONSE_GENERATION_PROMPT = """你是一个专业的企业服务顾问。用户刚刚进行了一段语音咨询，你需要基于检索到的相关信息给出专业的回复。

用户原始输入：
{original_text}

提取的关键信息：
- 主要意图：{main_intent}
- 关键点：{key_points}
- 提到的实体：{entities}
- 约束条件：{constraints}

检索到的相关信息：
{rag_context}

请给出专业、准确的回复：
1. 直接回答用户的核心问题
2. 针对每个关键点都要有回应
3. 引用检索到的具体信息（客户案例、产品功能等）
4. 如果信息不足以回答某些问题，要诚实说明
5. 语气专业但友好

回复：
"""


# ========== 核心处理类 ==========

class LongAudioRAGPipeline:
    """长语音输入的RAG处理流程"""

    def __init__(
        self,
        llm_client: OpenAI,
        vector_index: VectorIndex,
        bm25_index: Optional[BM25Index] = None,
        rerank_service: Optional[RerankService] = None,
        model_name: str = "Qwen/Qwen3-8B"
    ):
        self.llm = llm_client
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.rerank_service = rerank_service
        self.model_name = model_name

    def process(self, long_text: str, top_k: int = 5) -> Dict:
        """
        处理长文本输入

        Args:
            long_text: 用户的长文本输入
            top_k: 返回的文档数量

        Returns:
            包含响应、结构化信息、时间统计的字典
        """
        start_time = time.time()
        result = {
            "original_text": long_text,
            "original_length": len(long_text)
        }

        # 1. 结构化提取
        print("正在提取结构化信息...")
        extract_start = time.time()
        try:
            structured = self.extract_structure(long_text)
            result["structured_query"] = structured
            result["timing_extract"] = time.time() - extract_start
        except Exception as e:
            print(f"结构化提取失败: {e}")
            result["error"] = str(e)
            return result

        # 2. RAG检索
        print(f"正在检索（多问题模式: {structured.get('is_multi_question', False)}）...")
        rag_start = time.time()
        try:
            if structured.get("is_multi_question", False):
                # 多问题：分解并检索
                queries = self.decompose_queries(long_text, structured["key_points"])
                rag_results = self.multi_query_retrieval(queries, top_k=top_k)
                result["sub_queries"] = queries
            else:
                # 单问题：直接检索
                rag_results = self.single_query_retrieval(
                    structured["concise_query"],
                    top_k=top_k
                )

            result["rag_results"] = rag_results
            result["timing_rag"] = time.time() - rag_start
        except Exception as e:
            print(f"RAG检索失败: {e}")
            result["rag_error"] = str(e)
            result["rag_results"] = []

        # 3. LLM生成回复
        print("正在生成回复...")
        gen_start = time.time()
        try:
            response = self.generate_response(
                original_text=long_text,
                structured=structured,
                rag_results=result["rag_results"]
            )
            result["final_response"] = response
            result["timing_generate"] = time.time() - gen_start
        except Exception as e:
            print(f"回复生成失败: {e}")
            result["generation_error"] = str(e)
            result["final_response"] = ""

        # 4. 总时间
        result["timing_total"] = time.time() - start_time

        return result

    def extract_structure(self, text: str) -> Dict:
        """使用LLM提取结构化信息"""
        prompt = SUMMARIZATION_PROMPT.format(user_input=text)

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        content = response.choices[0].message.content.strip()

        # 移除可能的markdown代码块标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    def decompose_queries(self, text: str, key_points: List[str]) -> List[str]:
        """将多问题分解为多个子查询"""
        prompt = MULTI_QUERY_DECOMPOSITION_PROMPT.format(
            user_input=text,
            key_points="\n".join(f"- {p}" for p in key_points)
        )

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        content = response.choices[0].message.content.strip()

        # 移除markdown标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        result = json.loads(content.strip())
        return result["sub_queries"]

    def single_query_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """单查询检索"""
        # 向量检索
        vector_results = self.vector_index.search(query, top_k=top_k * 2)

        # 如果有BM25，进行混合检索
        if self.bm25_index:
            bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

            # 合并去重
            all_results = {}
            for doc in vector_results:
                all_results[doc["id"]] = doc
            for doc in bm25_results:
                if doc["id"] not in all_results:
                    all_results[doc["id"]] = doc

            results = list(all_results.values())
        else:
            results = vector_results

        # Rerank
        if self.rerank_service and len(results) > top_k:
            results = self.rerank_service.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def multi_query_retrieval(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """多查询检索并去重"""
        all_results = {}
        seen_ids = set()

        # 为每个查询检索
        for query in queries:
            results = self.single_query_retrieval(query, top_k=3)
            for doc in results:
                if doc["id"] not in seen_ids:
                    all_results[doc["id"]] = doc
                    seen_ids.add(doc["id"])

        results = list(all_results.values())

        # 使用第一个查询进行rerank
        if self.rerank_service and len(results) > top_k:
            results = self.rerank_service.rerank(queries[0], results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def generate_response(
        self,
        original_text: str,
        structured: Dict,
        rag_results: List[Dict]
    ) -> str:
        """生成最终回复"""
        # 构建RAG上下文
        rag_context = ""
        for i, doc in enumerate(rag_results, 1):
            rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
            rag_context += f"{doc.get('content', '无内容')}\n"

        if not rag_context:
            rag_context = "（未检索到相关信息）"

        # 构建prompt
        prompt = RESPONSE_GENERATION_PROMPT.format(
            original_text=original_text,
            main_intent=structured.get("main_intent", "未知"),
            key_points=", ".join(structured.get("key_points", [])),
            entities=", ".join(structured.get("entities", [])),
            constraints=", ".join(structured.get("constraints", [])),
            rag_context=rag_context
        )

        # 调用LLM生成
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # 测试代码
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # 初始化服务
    llm_client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    embedding_service = EmbeddingService()
    vector_index = VectorIndex(embedding_service)

    # 添加测试文档
    test_docs = [
        {
            "id": "doc1",
            "title": "中国银行合作案例",
            "content": "我们与中国银行合作开发了供应链金融平台，项目周期4个月，投入500万..."
        }
    ]
    vector_index.add_documents(test_docs)

    # 创建pipeline
    pipeline = LongAudioRAGPipeline(
        llm_client=llm_client,
        vector_index=vector_index
    )

    # 测试
    test_text = "你好，我想咨询一下你们公司的产品。我们是一家金融科技公司，听说你们给中国银行做过项目..."

    result = pipeline.process(test_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
