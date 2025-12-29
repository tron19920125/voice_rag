"""
Azure Speech-to-Text 服务
支持连续识别 + 内置VAD
"""

import logging
from typing import Callable, Optional
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)


class STTService:
    """语音识别服务（Azure Speech + VAD）"""

    def __init__(self, key: str, region: str, language: str = "zh-CN"):
        """
        初始化STT服务

        Args:
            key: Azure Speech API密钥
            region: Azure区域（如eastus）
            language: 识别语言（默认zh-CN）
        """
        self.key = key
        self.region = region
        self.language = language
        self.recognizer = None
        self._is_recognizing = False

    def start_continuous_recognition(
        self,
        on_recognizing: Callable[[str], None],
        on_recognized: Callable[[str], None],
        on_session_started: Optional[Callable[[], None]] = None,
        on_session_stopped: Optional[Callable[[], None]] = None,
        on_canceled: Optional[Callable[[str], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
    ):
        """
        启动连续识别（麦克风输入 + VAD）

        Args:
            on_recognizing: 部分识别结果回调
            on_recognized: 最终识别结果回调
            on_session_started: 会话开始回调
            on_session_stopped: 会话停止回调
            on_canceled: 取消/错误回调
            on_speech_started: 语音开始回调（Azure VAD不支持，仅用于接口兼容）
        """
        if self._is_recognizing:
            logger.warning("识别已在运行中")
            return

        try:
            # 配置语音识别
            speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )
            speech_config.speech_recognition_language = self.language

            # 使用默认麦克风
            audio_config = speechsdk.AudioConfig(use_default_microphone=True)

            # 创建识别器
            self.recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # 绑定事件回调
            def _on_recognizing(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                    text = evt.result.text
                    if text.strip():
                        on_recognizing(text)

            def _on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text
                    if text.strip():
                        on_recognized(text)
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.debug("未识别到语音")

            def _on_session_started(evt):
                self._is_recognizing = True
                logger.info("语音识别会话已启动")
                if on_session_started:
                    on_session_started()

            def _on_session_stopped(evt):
                self._is_recognizing = False
                logger.info("语音识别会话已停止")
                if on_session_stopped:
                    on_session_stopped()

            def _on_canceled(evt):
                self._is_recognizing = False
                reason = f"识别取消: {evt.result.cancellation_details.reason}"
                if evt.result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_details = evt.result.cancellation_details.error_details
                    reason = f"识别错误: {error_details}"
                    logger.error(reason)
                if on_canceled:
                    on_canceled(reason)

            # 连接回调
            self.recognizer.recognizing.connect(_on_recognizing)
            self.recognizer.recognized.connect(_on_recognized)
            self.recognizer.session_started.connect(_on_session_started)
            self.recognizer.session_stopped.connect(_on_session_stopped)
            self.recognizer.canceled.connect(_on_canceled)

            # 启动连续识别
            self.recognizer.start_continuous_recognition()
            logger.info("已启动连续语音识别")

        except Exception as e:
            self._is_recognizing = False
            error_msg = f"启动识别失败: {str(e)}"
            logger.error(error_msg)
            if on_canceled:
                on_canceled(error_msg)
            raise

    def stop_continuous_recognition(self):
        """停止连续识别"""
        if not self._is_recognizing or not self.recognizer:
            logger.warning("识别未在运行")
            return

        try:
            self.recognizer.stop_continuous_recognition()
            logger.info("已停止连续语音识别")
        except Exception as e:
            logger.error(f"停止识别失败: {str(e)}")
            raise
        finally:
            self._is_recognizing = False
