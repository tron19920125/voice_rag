"""
Qwen LLM 服务
支持流式调用和RAG上下文注入
"""

import logging
import json
from typing import List, Dict, Iterator, Optional
import requests

logger = logging.getLogger(__name__)


class QwenService:
    """Qwen LLM服务（流式调用）"""

    def __init__(
        self,
        api_base: str,
        model: str,
        token: str,
        temperature: float = 0.7,
    ):
        """
        初始化Qwen服务

        Args:
            api_base: API基础URL
            model: 模型名称（如qwen-plus）
            token: API令牌
            temperature: 温度参数（0-1）
        """
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.token = token
        self.temperature = temperature

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        流式对话

        Args:
            messages: 对话历史，格式：[{"role": "user", "content": "..."}]
            system_prompt: 系统提示词（可选）

        Yields:
            生成的文本片段
        """
        # 构建完整messages
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        # 构建请求
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            # 发送流式请求
            url = f"{self.api_base}/chat/completions"
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            # 解析流式响应
            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                # 跳过data:前缀
                if line.startswith("data: "):
                    line = line[6:]

                # 结束标记
                if line == "[DONE]":
                    break

                # 解析JSON
                try:
                    chunk = json.loads(line)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")

                    if content:
                        yield content

                except json.JSONDecodeError:
                    continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Qwen API调用失败: {str(e)}")
            raise RuntimeError(f"LLM调用失败: {str(e)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        非流式对话（一次性返回完整结果）

        Args:
            messages: 对话历史
            system_prompt: 系统提示词（可选）

        Returns:
            完整响应文本
        """
        # 构建完整messages
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        # 构建请求
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "stream": False,
        }

        try:
            url = f"{self.api_base}/chat/completions"
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"Qwen API调用失败: {str(e)}")
            raise RuntimeError(f"LLM调用失败: {str(e)}")

    def chat_with_rag(
        self,
        query: str,
        rag_context: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        stream: bool = True,
    ):
        """
        带RAG上下文的对话

        Args:
            query: 用户查询
            rag_context: RAG检索的上下文
            messages: 对话历史
            system_prompt: 系统提示词（可选）
            stream: 是否流式输出

        Returns:
            流式：Iterator[str]
            非流式：str
        """
        # 构建带RAG的用户消息
        enhanced_query = f"{rag_context}\n\n用户问题：{query}"

        # 添加到消息列表
        new_messages = messages + [{"role": "user", "content": enhanced_query}]

        # 调用对应方法
        if stream:
            return self.chat_stream(new_messages, system_prompt=system_prompt)
        else:
            return self.chat(new_messages, system_prompt=system_prompt)
