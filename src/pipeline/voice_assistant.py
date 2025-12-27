"""
语音助手主流程编排
整合 STT + RAG + LLM + TTS + 上下文管理
"""

import logging
import re
import time
from typing import List, Dict, Optional, Callable
from threading import Event, Thread
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """语音助手主控制器"""

    def __init__(
        self,
        stt_service,
        tts_service,
        rag_searcher,
        llm_service,
        context_manager,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化语音助手

        Args:
            stt_service: STT服务
            tts_service: TTS服务
            rag_searcher: RAG检索器
            llm_service: LLM服务
            context_manager: 上下文管理器
            system_prompt: 系统提示词
        """
        self.stt = stt_service
        self.tts = tts_service
        self.rag = rag_searcher
        self.llm = llm_service
        self.context_mgr = context_manager

        self.system_prompt = system_prompt or self._default_system_prompt()

        # 对话状态
        self.messages: List[Dict[str, str]] = []
        self.is_running = False
        self.is_processing = False

        # TTS流式播放队列
        self.tts_queue = Queue()
        self.tts_worker_thread: Optional[Thread] = None
        self.is_tts_playing = False

        # 临时变量（用于处理单轮对话）
        self._current_user_text = None
        self._current_assistant_text = ""
        self._recognition_complete_event = Event()

        # 回调函数
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_assistant_response: Optional[Callable[[str], None]] = None
        self.on_rag_retrieved: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """
你是一个专业的智能客服助手，具备以下特点：
1. 友好、耐心、专业
2. 基于提供的知识库内容准确回答问题
3. 如果知识库中没有相关信息，诚实告知用户
4. 回答简洁明了，避免冗长
5. 对于直接问答，直接给出答案
""".strip()

    def start(self):
        """启动语音助手"""
        if self.is_running:
            logger.warning("语音助手已在运行")
            return

        logger.info("启动语音助手...")
        self.is_running = True

        # 启动STT连续识别
        try:
            self.stt.start_continuous_recognition(
                on_recognizing=self._on_stt_recognizing,
                on_recognized=self._on_stt_recognized,
                on_session_started=self._on_stt_session_started,
                on_session_stopped=self._on_stt_session_stopped,
                on_canceled=self._on_stt_canceled,
            )
        except Exception as e:
            logger.error(f"启动STT失败: {str(e)}")
            self.is_running = False
            if self.on_error:
                self.on_error(f"启动失败: {str(e)}")
            raise

    def stop(self):
        """停止语音助手"""
        if not self.is_running:
            return

        logger.info("停止语音助手...")
        self.is_running = False

        try:
            self.stt.stop_continuous_recognition()
        except Exception as e:
            logger.error(f"停止STT失败: {str(e)}")

        # 停止TTS worker
        self._stop_tts_worker()

    def _start_tts_worker(self):
        """启动TTS工作线程"""
        if self.tts_worker_thread and self.tts_worker_thread.is_alive():
            return

        self.is_tts_playing = True
        self.tts_worker_thread = Thread(target=self._tts_worker_loop, daemon=True)
        self.tts_worker_thread.start()
        logger.debug("TTS worker线程已启动")

    def _stop_tts_worker(self):
        """停止TTS工作线程"""
        if not self.tts_worker_thread:
            return

        self.is_tts_playing = False
        # 清空队列
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except Empty:
                break

        # 发送停止信号
        self.tts_queue.put(None)
        logger.debug("TTS worker线程已停止")

    def _tts_worker_loop(self):
        """TTS工作线程主循环"""
        while self.is_tts_playing:
            try:
                # 从队列获取句子
                sentence = self.tts_queue.get(timeout=0.5)

                # None表示结束信号
                if sentence is None:
                    logger.debug("TTS队列结束")
                    break

                # 合成并播放
                logger.debug(f"TTS合成: {sentence[:50]}...")
                self.tts.synthesize_and_play(
                    sentence,
                    on_canceled=lambda reason: logger.error(f"TTS失败: {reason}"),
                )

            except Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker错误: {str(e)}")

        # 播放完成后增加安全延迟（确保音频完全播放）
        logger.debug("等待音频播放完毕...")
        time.sleep(1.0)

        # 重置标志
        self.is_tts_playing = False
        logger.debug("TTS worker循环结束，已重置播放标志")

    def _on_stt_recognizing(self, text: str):
        """STT部分识别结果"""
        logger.debug(f"识别中: {text}")

    def _on_stt_recognized(self, text: str):
        """STT最终识别结果"""
        # 阻止在处理期间或TTS播放期间的识别
        if not self.is_running or self.is_processing or self.is_tts_playing:
            logger.debug(f"忽略STT输入（正在处理或播放中）: {text}")
            return

        logger.info(f"用户: {text}")

        # 检查退出命令
        exit_keywords = ["退出", "再见", "结束对话", "拜拜"]
        if any(keyword in text for keyword in exit_keywords):
            logger.info("用户请求退出")
            self.stop()
            return

        # 触发用户语音回调
        if self.on_user_speech:
            self.on_user_speech(text)

        # 处理用户输入（异步）
        self._current_user_text = text
        self._recognition_complete_event.set()

        # 启动处理线程
        Thread(target=self._process_user_input, args=(text,), daemon=True).start()

    def _on_stt_session_started(self):
        """STT会话启动"""
        logger.info("语音识别会话已启动")

    def _on_stt_session_stopped(self):
        """STT会话停止"""
        logger.info("语音识别会话已停止")

    def _on_stt_canceled(self, reason: str):
        """STT取消/错误"""
        logger.error(f"STT错误: {reason}")
        if self.on_error:
            self.on_error(f"语音识别错误: {reason}")

    def _process_user_input(self, user_text: str):
        """
        处理用户输入（核心流程）

        流程：
        1. RAG检索
        2. 构建上下文
        3. LLM生成（流式）
        4. TTS播放
        5. 更新历史
        6. 压缩上下文
        """
        if self.is_processing:
            logger.warning("正在处理中，请稍候...")
            return

        self.is_processing = True
        self._current_assistant_text = ""

        try:
            # ===== 步骤1: RAG检索 =====
            logger.info("[1/5] RAG检索中...")
            rag_result = self.rag.search(user_text)

            if self.on_rag_retrieved:
                self.on_rag_retrieved(rag_result)

            # ===== 步骤2: 判断直接回答还是RAG生成 =====
            if rag_result["type"] == "direct_answer":
                # 高置信度QA，直接返回答案
                logger.info(f"[2/5] 直接回答（置信度: {rag_result['confidence']:.2f}）")
                assistant_text = rag_result["answer"]
                self._current_assistant_text = assistant_text

                # ===== 步骤3: TTS播放（仅直接回答需要）=====
                logger.info("[3/5] 播放回答...")
                self._play_response(assistant_text)

            else:
                # 需要LLM生成
                logger.info("[2/5] LLM生成回答中...")

                # 构建RAG上下文
                rag_context = self._build_rag_context(rag_result)

                # 管理上下文（检查是否需要压缩）
                managed_messages, summary = self.context_mgr.manage_context(
                    self.messages
                )

                # 构建完整上下文
                if summary:
                    managed_messages = self.context_mgr.build_context_with_summary(
                        managed_messages, summary
                    )

                # LLM流式生成（内部已实时TTS播放）
                logger.info("[3/5] 流式生成并播放...")
                assistant_text = self._generate_with_llm(
                    user_text, rag_context, managed_messages
                )
                self._current_assistant_text = assistant_text

            # ===== 步骤4: 更新对话历史 =====
            logger.info("[4/5] 更新对话历史...")
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": assistant_text})

            # ===== 步骤5: 异步压缩上下文（如果需要）=====
            if self.context_mgr.should_compress(self.messages):
                logger.info("[5/5] 触发异步上下文压缩...")
                Thread(target=self._compress_context_async, daemon=True).start()

            # 等待TTS播放完成（防止回音）
            logger.info("等待TTS播放完成...")
            while self.is_tts_playing:
                time.sleep(0.1)

            logger.info("处理完成")

        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            if self.on_error:
                self.on_error(f"处理失败: {str(e)}")

        finally:
            self.is_processing = False

    def _build_rag_context(self, rag_result: Dict) -> str:
        """构建RAG上下文"""
        if rag_result["type"] != "rag_context" or not rag_result.get("docs"):
            return ""

        context_parts = ["以下是相关的知识库内容：\n"]

        for i, doc in enumerate(rag_result["docs"], 1):
            context_parts.append(f"【文档 {i}】{doc['title']}")
            context_parts.append(doc["content"])
            context_parts.append("")

        context_parts.append("请基于以上内容回答用户问题。如果内容中没有相关信息，请如实告知。")

        return "\n".join(context_parts)

    def _generate_with_llm(
        self, user_text: str, rag_context: str, messages: List[Dict[str, str]]
    ) -> str:
        """使用LLM生成回答（流式 + 实时TTS）"""
        # 构建增强的用户消息
        if rag_context:
            enhanced_query = f"{rag_context}\n\n用户问题：{user_text}"
        else:
            enhanced_query = user_text

        # 添加到消息列表
        current_messages = messages + [{"role": "user", "content": enhanced_query}]

        # 启动TTS worker
        self._start_tts_worker()

        # 流式生成 + 分句处理
        sentence_buffer = ""
        full_response = ""

        for chunk in self.llm.chat_stream(
            current_messages, system_prompt=self.system_prompt
        ):
            full_response += chunk
            sentence_buffer += chunk

            # 检测强句子边界（。！？\n）
            strong_parts = re.split(r'([。！？\n])', sentence_buffer)

            if len(strong_parts) > 1:
                # 有强分隔符，按强分隔符分句
                for i in range(0, len(strong_parts) - 1, 2):
                    if i + 1 < len(strong_parts):
                        sentence = strong_parts[i] + strong_parts[i+1]
                        if sentence.strip():
                            self.tts_queue.put(sentence.strip())
                            logger.debug(f"TTS队列(强分): {sentence.strip()[:30]}...")

                # 保留未完成的部分
                sentence_buffer = strong_parts[-1] if len(strong_parts) % 2 == 1 else ""

            elif len(sentence_buffer) > 40:
                # 无强分隔符但buffer过长，按弱分隔符（，、；：）分句
                weak_parts = re.split(r'([，、；：])', sentence_buffer)
                if len(weak_parts) > 1:
                    for i in range(0, len(weak_parts) - 1, 2):
                        if i + 1 < len(weak_parts):
                            sentence = weak_parts[i] + weak_parts[i+1]
                            if sentence.strip():
                                self.tts_queue.put(sentence.strip())
                                logger.debug(f"TTS队列(弱分): {sentence.strip()[:30]}...")

                    # 保留未完成的部分
                    sentence_buffer = weak_parts[-1] if len(weak_parts) % 2 == 1 else ""

        # 处理剩余文本
        if sentence_buffer.strip():
            self.tts_queue.put(sentence_buffer.strip())
            logger.debug(f"TTS队列(剩余): {sentence_buffer.strip()[:30]}...")

        # 发送结束信号
        self.tts_queue.put(None)

        logger.info(f"助手: {full_response}")

        # 触发回调
        if self.on_assistant_response:
            self.on_assistant_response(full_response)

        return full_response

    def _play_response(self, text: str):
        """播放回答"""
        try:
            self.tts.synthesize_and_play(
                text,
                on_synthesis_started=lambda: logger.debug("TTS开始"),
                on_synthesis_completed=lambda: logger.debug("TTS完成"),
                on_canceled=lambda reason: logger.error(f"TTS失败: {reason}"),
            )
        except Exception as e:
            logger.error(f"TTS播放失败: {str(e)}")

    def _compress_context_async(self):
        """异步压缩上下文"""
        try:
            logger.info("开始异步压缩上下文...")
            managed_messages, summary = self.context_mgr.manage_context(
                self.messages, force_compress=True
            )

            if summary:
                # 更新消息列表
                self.messages = self.context_mgr.build_context_with_summary(
                    managed_messages, summary
                )
                logger.info(f"压缩完成，保留 {len(self.messages)} 条消息")

        except Exception as e:
            logger.error(f"异步压缩失败: {str(e)}")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.messages.copy()

    def clear_history(self):
        """清空对话历史"""
        self.messages.clear()
        logger.info("对话历史已清空")
