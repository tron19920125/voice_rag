#!/usr/bin/env python3
"""
语音客服系统主程序
整合 STT + RAG + LLM + TTS
"""

import logging
import sys
import signal
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.speech.stt_service import STTService
from src.speech.tts_service import TTSService
from src.knowledge.rag_searcher import RAGSearcher
from src.llm.qwen_service import QwenService
from src.llm.context_manager import ContextManager
from src.pipeline.voice_assistant import VoiceAssistant
from rag_utils import EmbeddingService, RerankingService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_signal_handler(assistant: VoiceAssistant):
    """设置信号处理器（优雅退出）"""

    def signal_handler(sig, frame):
        logger.info("\n收到退出信号，正在关闭...")
        assistant.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("语音客服系统启动中...")
    logger.info("=" * 70)

    try:
        # ===== 1. 初始化服务 =====
        logger.info("\n[1/6] 初始化语音服务...")
        stt_service = STTService(
            key=settings.azure_speech.key,
            region=settings.azure_speech.region,
            language=settings.azure_speech.language,
        )

        tts_service = TTSService(
            key=settings.azure_speech.key,
            region=settings.azure_speech.region,
            voice_name="zh-CN-XiaoxiaoNeural",
            rate=1.0,
        )

        # ===== 2. 初始化RAG服务 =====
        logger.info("[2/6] 初始化RAG检索服务...")
        embedding_service = EmbeddingService()
        reranking_service = RerankingService()

        rag_searcher = RAGSearcher(
            endpoint=settings.azure_search.endpoint,
            api_key=settings.azure_search.key,
            index_name=settings.azure_search.index_name,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
        )

        # ===== 3. 初始化LLM服务 =====
        logger.info("[3/6] 初始化LLM服务...")
        llm_service = QwenService(
            api_base=settings.qwen.api_base,
            model=settings.qwen.model,
            token=settings.qwen.token,
            temperature=settings.qwen.temperature,
        )

        # ===== 4. 初始化上下文管理器 =====
        logger.info("[4/6] 初始化上下文管理器...")
        context_manager = ContextManager(
            llm_service=llm_service,
            token_threshold=settings.context.compression_threshold,
            keep_recent_turns=settings.context.keep_recent_turns,
        )

        # ===== 5. 创建语音助手 =====
        logger.info("[5/6] 创建语音助手...")

        system_prompt = """
你是一个专业的智能客服助手，具备以下特点：
1. 友好、耐心、专业，善于倾听用户需求
2. 基于提供的知识库内容准确回答问题
3. 如果知识库中没有相关信息，诚实告知用户，不编造信息
4. 回答简洁明了，避免冗长，直击要点
5. 对于FAQ问题，直接给出答案
6. 遇到复杂问题时，会引导用户提供更多信息
7. 不要在回答中使用emoji表情符号
""".strip()

        assistant = VoiceAssistant(
            stt_service=stt_service,
            tts_service=tts_service,
            rag_searcher=rag_searcher,
            llm_service=llm_service,
            context_manager=context_manager,
            system_prompt=system_prompt,
        )

        # 设置回调
        def on_user_speech(text: str):
            print(f"\n👤 用户: {text}")

        def on_assistant_response(text: str):
            print(f"🤖 助手: {text}\n")

        def on_rag_retrieved(result: dict):
            if result["type"] == "direct_answer":
                print(f"💡 直接回答 (置信度: {result['confidence']:.2f})")
            else:
                doc_count = len(result.get("docs", []))
                print(f"📚 检索到 {doc_count} 个相关文档")

        def on_error(error: str):
            print(f"❌ 错误: {error}")

        assistant.on_user_speech = on_user_speech
        assistant.on_assistant_response = on_assistant_response
        assistant.on_rag_retrieved = on_rag_retrieved
        assistant.on_error = on_error

        # ===== 6. 启动助手 =====
        logger.info("[6/6] 启动语音助手...")
        setup_signal_handler(assistant)

        assistant.start()

        logger.info("\n" + "=" * 70)
        logger.info("✓ 语音客服系统已就绪！")
        logger.info("请开始说话...")
        logger.info("说 '退出'、'再见' 或按 Ctrl+C 结束对话")
        logger.info("=" * 70 + "\n")

        # 保持运行
        while assistant.is_running:
            try:
                import time

                time.sleep(1)
            except KeyboardInterrupt:
                break

    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        logger.info("系统已关闭")


if __name__ == "__main__":
    main()
