"""
长语音输入模拟器
模拟VAD和ASR的流式输入场景
"""

import time
from typing import Generator, List


class LongAudioSimulator:
    """模拟长语音输入场景"""

    @staticmethod
    def simulate_streaming_input(
        text: str,
        chunk_size: int = 50,
        delay: float = 0.1
    ) -> Generator[str, None, None]:
        """
        模拟流式输入（模拟ASR实时转文本）

        Args:
            text: 完整文本
            chunk_size: 每次返回的字符数（模拟语音速度）
            delay: 每次返回的延迟（秒）

        Yields:
            文本片段
        """
        current = ""
        char_count = 0

        for char in text:
            current += char
            char_count += 1

            # 达到chunk_size或遇到句子结束标点
            if char_count >= chunk_size or char in ('。', '！', '？', '.', '!', '?'):
                yield current
                if delay > 0:
                    time.sleep(delay)
                current = ""
                char_count = 0

        # 返回剩余部分
        if current:
            yield current

    @staticmethod
    def simulate_vad_completion(text: str, pause_threshold: int = 3) -> bool:
        """
        模拟VAD检测完成（判断用户是否说完）

        Args:
            text: 当前文本
            pause_threshold: 停顿阈值（句尾标点符号数量）

        Returns:
            是否检测到长停顿（意味着用户说完了）
        """
        # 简单模拟：检查是否有句号、问号等终止符
        ending_punctuation = ['.', '。', '?', '？', '!', '！']

        # 统计结尾的终止符数量
        end_punct_count = 0
        for char in reversed(text.strip()):
            if char in ending_punctuation:
                end_punct_count += 1
            elif char.isspace():
                continue
            else:
                break

        return end_punct_count >= pause_threshold

    @staticmethod
    def collect_full_input(
        text: str,
        chunk_size: int = 50,
        show_progress: bool = False
    ) -> str:
        """
        收集完整输入（模拟等待用户说完）

        Args:
            text: 完整文本
            chunk_size: 每次处理的字符数
            show_progress: 是否显示进度

        Returns:
            完整的文本
        """
        full_text = ""

        for chunk in LongAudioSimulator.simulate_streaming_input(
            text,
            chunk_size=chunk_size,
            delay=0  # 不延迟
        ):
            full_text += chunk
            if show_progress:
                print(f"收集中... ({len(full_text)}/{len(text)} 字符)")

        return full_text

    @staticmethod
    def split_by_sentences(text: str) -> List[str]:
        """
        按句子切分文本（模拟VAD切分）

        Args:
            text: 完整文本

        Returns:
            句子列表
        """
        import re

        # 按中英文句子终止符切分
        sentences = re.split(r'([。！？.!?]+)', text)

        # 合并句子和标点
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                sentence = sentence.strip()
                if sentence:
                    result.append(sentence)

        # 处理最后一个句子（如果没有标点）
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return result


class StreamingProcessor:
    """流式处理器（模拟边听边处理的场景）"""

    def __init__(self):
        self.buffer = ""
        self.processed_chunks = []

    def add_chunk(self, chunk: str):
        """添加新的文本片段"""
        self.buffer += chunk
        self.processed_chunks.append(chunk)

    def get_current_text(self) -> str:
        """获取当前累积的文本"""
        return self.buffer

    def get_chunks(self) -> List[str]:
        """获取所有处理过的片段"""
        return self.processed_chunks

    def is_complete(self) -> bool:
        """判断是否完成"""
        return LongAudioSimulator.simulate_vad_completion(self.buffer)

    def reset(self):
        """重置缓冲区"""
        self.buffer = ""
        self.processed_chunks = []


if __name__ == "__main__":
    # 测试代码
    test_text = """
    你好，我想咨询一下你们公司的产品。我们是一家金融科技公司，最近在做数字化转型。
    我们现在有大约100人的团队。我听说你们之前给中国银行做过项目。
    我想了解一下那个项目的情况。另外，我们的预算大概在50万左右。
    """.strip()

    print("=== 测试1：流式输入 ===")
    for i, chunk in enumerate(LongAudioSimulator.simulate_streaming_input(
        test_text,
        chunk_size=30,
        delay=0.2
    ), 1):
        print(f"Chunk {i}: {chunk}")

    print("\n=== 测试2：VAD完成检测 ===")
    print(f"完整文本: {LongAudioSimulator.simulate_vad_completion(test_text)}")
    print(f"不完整: {LongAudioSimulator.simulate_vad_completion('你好，我想')}")

    print("\n=== 测试3：按句子切分 ===")
    sentences = LongAudioSimulator.split_by_sentences(test_text)
    for i, sent in enumerate(sentences, 1):
        print(f"句子 {i}: {sent}")

    print("\n=== 测试4：流式处理器 ===")
    processor = StreamingProcessor()
    for chunk in LongAudioSimulator.simulate_streaming_input(test_text, chunk_size=30, delay=0):
        processor.add_chunk(chunk)
        print(f"当前长度: {len(processor.get_current_text())}, 是否完成: {processor.is_complete()}")
