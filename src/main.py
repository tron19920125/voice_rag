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
from src.speech.stt_silero_service import STTSileroService
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
        # ===== 1. 初始化语音服务 =====
        logger.info("\n[1/6] 初始化语音服务...")
        logger.info(f"VAD类型: {settings.vad.type}")

        # 根据配置选择STT实现
        if settings.vad.type == "silero":
            logger.info("使用 Silero VAD + Azure STT")
            stt_service = STTSileroService(
                key=settings.azure_speech.key,
                region=settings.azure_speech.region,
                language=settings.azure_speech.language,
                sample_rate=settings.vad.sample_rate,
                vad_threshold=settings.vad.threshold,
                min_speech_duration=settings.vad.min_speech_duration,
                min_silence_duration=settings.vad.min_silence_duration,
            )
        else:
            logger.info("使用 Azure 内置 VAD")
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

        if settings.qwen.use_local_vllm:
            logger.info("使用本地vLLM服务")
            logger.info(f"  - 主模型(14B): {settings.qwen.local_vllm_14b_base}")
            logger.info(f"  - 辅助模型(8B): {settings.qwen.local_vllm_8b_base}")

            # 主LLM服务（14B，用于对话生成）
            llm_service = QwenService(
                api_base=settings.qwen.local_vllm_14b_base,
                model="Qwen/Qwen2.5-14B-Instruct",  # vllm模型名称
                token="EMPTY",
                temperature=settings.qwen.temperature,
                is_local_vllm=True,
            )

            # 辅助LLM服务（8B，用于RAG判断等轻量级任务）
            sub_llm_service = QwenService(
                api_base=settings.qwen.local_vllm_8b_base,
                model="Qwen/Qwen2.5-8B-Instruct",  # vllm模型名称
                token="EMPTY",
                temperature=settings.qwen.temperature,
                is_local_vllm=True,
            )
        else:
            logger.info("使用远程API服务")
            logger.info(f"  - 主模型: {settings.qwen.model}")
            logger.info(f"  - 辅助模型: {settings.qwen.sub_model}")

            # 主LLM服务
            llm_service = QwenService(
                api_base=settings.qwen.api_base,
                model=settings.qwen.model,
                token=settings.qwen.token,
                temperature=settings.qwen.temperature,
                is_local_vllm=False,
            )

            # 辅助LLM服务
            sub_llm_service = QwenService(
                api_base=settings.qwen.api_base,
                model=settings.qwen.sub_model,
                token=settings.qwen.token,
                temperature=settings.qwen.temperature,
                is_local_vllm=False,
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
你是TechFlow的智能客服，用自然口语回答问题。

核心要求：
1. 说人话，别太书面，就像朋友聊天一样
2. 根据知识库回答，不知道就直说"这个我不太清楚"
3. 回答简短点，别啰嗦，直接说重点
4. 不要用emoji
5. 不要用markdown格式（不要用**、#、*、-等符号），只输出纯文本口语
6. 少用符号，尽量用中文化的自然表达
   - 不要用顿号、冒号、括号、斜杠这些符号
   - 用"和"、"还有"、"另外"、"每"这样的中文连接词
7. 语气自然、亲切，但保持专业

示例风格：
好的回答：
- FlowMind是我们的工业物联网平台，可以帮您实时监控设备和分析数据
- 这个设备一套大概是五万块左右
- 培训费用是15000元每人
- 我们支持工业4.0趋势

不好的回答：
- FlowMind系统是一款集成了多种先进功能的综合性工业物联网解决方案平台...（太书面）
- **FlowMind** 是...（用了markdown粗体）
- **工业 4.0 趋势**（用了markdown粗体）
- 支持设备连接、数据采集、实时监控、数据分析（用顿号列举，不够口语）
- 价格：5万元/套（用了冒号和斜杠，太生硬）
- 15000元/人（用了斜杠，应该说"每人"）
""".strip()

        assistant = VoiceAssistant(
            stt_service=stt_service,
            tts_service=tts_service,
            rag_searcher=rag_searcher,
            llm_service=llm_service,
            context_manager=context_manager,
            system_prompt=system_prompt,
            sub_llm_service=sub_llm_service,
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
