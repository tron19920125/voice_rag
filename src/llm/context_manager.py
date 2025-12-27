"""
上下文管理器
支持阈值触发的自动压缩 + 保留最近N轮对话
"""

import logging
from typing import List, Dict, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ContextManager:
    """对话上下文管理器（压缩 + 截断）"""

    def __init__(
        self,
        llm_service,
        token_threshold: int = 4000,
        keep_recent_turns: int = 2,
        compression_model: Optional[str] = None,
    ):
        """
        初始化上下文管理器

        Args:
            llm_service: LLM服务实例（用于压缩）
            token_threshold: 触发压缩的token阈值
            keep_recent_turns: 保留最近N轮对话
            compression_model: 压缩使用的模型（可选）
        """
        self.llm_service = llm_service
        self.token_threshold = token_threshold
        self.keep_recent_turns = keep_recent_turns
        self.compression_model = compression_model
        self.executor = ThreadPoolExecutor(max_workers=1)

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本token数量

        简单规则：
        - 中文：1字 ≈ 2 tokens
        - 英文/数字：1词 ≈ 1.3 tokens
        - 标点符号：1个 ≈ 1 token

        Args:
            text: 文本内容

        Returns:
            估算的token数量
        """
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        english_words = len([w for w in text.split() if w.isalpha()])
        other_chars = len(text) - chinese_chars

        tokens = chinese_chars * 2 + english_words * 1.3 + other_chars * 0.5
        return int(tokens)

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的总token数

        Args:
            messages: 消息列表

        Returns:
            总token数
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            # 角色标记额外token
            total += self.estimate_tokens(content) + 10
        return total

    def should_compress(self, messages: List[Dict[str, str]]) -> bool:
        """
        判断是否需要压缩

        Args:
            messages: 当前消息列表

        Returns:
            是否需要压缩
        """
        total_tokens = self.count_messages_tokens(messages)
        return total_tokens > self.token_threshold

    def split_recent_and_old(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        分离最近N轮对话和旧对话

        Args:
            messages: 消息列表

        Returns:
            (recent_messages, old_messages) 元组
        """
        if len(messages) <= self.keep_recent_turns * 2:
            return messages, []

        # 保留最近N轮（每轮包含user + assistant）
        split_index = -(self.keep_recent_turns * 2)
        old_messages = messages[:split_index]
        recent_messages = messages[split_index:]

        return recent_messages, old_messages

    def compress_messages(
        self, messages: List[Dict[str, str]]
    ) -> str:
        """
        使用LLM压缩旧对话

        Args:
            messages: 需要压缩的消息列表

        Returns:
            压缩后的摘要文本
        """
        if not messages:
            return ""

        # 构建压缩提示
        conversation_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]
        )

        compression_prompt = f"""
请将以下对话历史总结为简短的摘要（100-200字），保留关键信息和上下文：

{conversation_text}

摘要：
""".strip()

        try:
            # 使用LLM压缩（非流式）
            summary = self.llm_service.chat(
                messages=[{"role": "user", "content": compression_prompt}],
                system_prompt="你是一个专业的对话摘要助手，能够提取关键信息并生成简洁的摘要。",
            )

            logger.info(f"压缩完成: {len(conversation_text)} 字 -> {len(summary)} 字")
            return summary.strip()

        except Exception as e:
            logger.error(f"压缩失败: {str(e)}")
            # 降级方案：简单截断
            return conversation_text[:200] + "..."

    async def compress_messages_async(
        self, messages: List[Dict[str, str]]
    ) -> str:
        """
        异步压缩旧对话

        Args:
            messages: 需要压缩的消息列表

        Returns:
            压缩后的摘要文本
        """
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            self.executor, self.compress_messages, messages
        )
        return summary

    def manage_context(
        self, messages: List[Dict[str, str]], force_compress: bool = False
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        管理上下文（同步版本）

        Args:
            messages: 当前消息列表
            force_compress: 强制压缩

        Returns:
            (managed_messages, summary) 元组
            - managed_messages: 处理后的消息列表（最近N轮）
            - summary: 压缩摘要（如果有）
        """
        if not force_compress and not self.should_compress(messages):
            return messages, None

        # 分离最近和旧对话
        recent_messages, old_messages = self.split_recent_and_old(messages)

        if not old_messages:
            return recent_messages, None

        # 压缩旧对话
        logger.info(f"开始压缩: {len(old_messages)} 条旧消息")
        summary = self.compress_messages(old_messages)

        return recent_messages, summary

    async def manage_context_async(
        self, messages: List[Dict[str, str]], force_compress: bool = False
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        管理上下文（异步版本）

        Args:
            messages: 当前消息列表
            force_compress: 强制压缩

        Returns:
            (managed_messages, summary) 元组
        """
        if not force_compress and not self.should_compress(messages):
            return messages, None

        # 分离最近和旧对话
        recent_messages, old_messages = self.split_recent_and_old(messages)

        if not old_messages:
            return recent_messages, None

        # 异步压缩旧对话
        logger.info(f"开始异步压缩: {len(old_messages)} 条旧消息")
        summary = await self.compress_messages_async(old_messages)

        return recent_messages, summary

    def build_context_with_summary(
        self,
        recent_messages: List[Dict[str, str]],
        summary: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        构建带摘要的上下文

        Args:
            recent_messages: 最近的消息
            summary: 历史摘要

        Returns:
            完整的消息列表
        """
        if not summary:
            return recent_messages

        # 将摘要插入到最前面作为system消息
        return [
            {
                "role": "system",
                "content": f"历史对话摘要：\n{summary}",
            }
        ] + recent_messages
