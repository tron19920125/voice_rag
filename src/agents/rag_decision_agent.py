"""
RAG需求判断Agent
基于用户问题和对话上下文，判断是否需要查询知识库
"""

import logging
from typing import List, Dict
from src.llm.qwen_service import QwenService
from src.config.settings import settings

logger = logging.getLogger(__name__)


class RAGDecisionAgent:
    """RAG需求判断Agent"""

    def __init__(self, sub_llm_service=None):
        """
        初始化，使用sub_model（qwen3-8b）提供快速判断

        Args:
            sub_llm_service: 辅助LLM服务（可选，未提供时自动创建）
        """
        if sub_llm_service is None:
            # 向后兼容：自动创建
            self.sub_llm = QwenService(
                api_base=settings.qwen.api_base,
                model=settings.qwen.sub_model,
                token=settings.qwen.token,
                temperature=0.1,
                is_local_vllm=False,
            )
        else:
            self.sub_llm = sub_llm_service

        self.system_prompt = """你是一个智能客服系统的决策助手。
任务：判断用户问题是否需要查询知识库（RAG）。

规则：
- 闲聊、问候、感谢、再见 -> 0
- 询问具体业务、产品、政策、流程、技术问题 -> 1
- 追问上文提到的内容（如"那XX呢"、"还有吗"） -> 看上文是否用了RAG，是则1
- 模糊的通用问题（"怎么办"、"能否帮我"） -> 1

只输出0或1，不要解释。"""

    def should_use_rag(
        self,
        user_query: str,
        recent_messages: List[Dict[str, str]] = None
    ) -> bool:
        """
        判断是否需要RAG检索

        Args:
            user_query: 用户问题
            recent_messages: 最近的对话历史（建议最多6条消息，即3轮对话）

        Returns:
            True: 需要RAG
            False: 不需要RAG
        """
        # 构建上下文（如果有历史对话）
        context = ""
        if recent_messages and len(recent_messages) > 0:
            context = "最近对话:\n"
            for msg in recent_messages[-6:]:  # 只看最近3轮（6条消息）
                role = "用户" if msg["role"] == "user" else "助手"
                content = msg["content"][:80]  # 截取前80字符
                context += f"{role}: {content}...\n"
            context += "\n"

        # 构建判断消息
        user_message = f"{context}当前问题: {user_query}\n\n是否需要RAG? (0或1)"

        try:
            logger.debug(f"RAG判断 - 用户问题: {user_query[:50]}")

            # 调用sub_model判断
            response = self.sub_llm.chat(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=self.system_prompt
            )

            # 解析结果（提取0或1）
            result = response.strip()
            logger.debug(f"RAG判断 - 模型返回: {result}")

            if "1" in result:
                logger.info("RAG判断: 需要检索")
                return True
            elif "0" in result:
                logger.info("RAG判断: 不需要检索")
                return False
            else:
                # 默认策略：不确定时使用RAG（保守策略）
                logger.warning(f"RAG判断返回异常: {result}，默认使用RAG")
                return True

        except Exception as e:
            logger.error(f"RAG判断失败: {str(e)}，默认使用RAG")
            return True  # 异常时默认使用RAG
