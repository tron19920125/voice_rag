"""
使用官方silero-vad库测试语音检测
"""

import torch
import wave
import numpy as np
from pathlib import Path


def read_audio_with_wave(wav_file):
    """
    使用wave库读取音频并转换为torch tensor

    Args:
        wav_file: WAV文件路径

    Returns:
        torch tensor (1D, float32, normalized to [-1, 1])
    """
    with wave.open(wav_file, 'rb') as wf:
        # 获取音频参数
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        # 读取音频数据
        audio_data = wf.readframes(n_frames)

    # 转换为numpy数组
    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

    # 如果是多声道，只取第一个声道
    if channels > 1:
        audio_int16 = audio_int16[::channels]

    # 归一化到[-1, 1]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # 转换为torch tensor
    audio_tensor = torch.from_numpy(audio_float32)

    return audio_tensor, sample_rate


def test_with_official_silero(wav_file):
    """
    使用官方silero-vad库测试

    Args:
        wav_file: WAV文件路径
    """
    print(f"\n测试文件: {wav_file}")
    print("="*70)

    try:
        # 导入silero-vad官方库
        from silero_vad import load_silero_vad, get_speech_timestamps

        print("加载Silero VAD模型（官方库）...")
        model = load_silero_vad()
        print("模型加载完成\n")

        # 读取音频（使用wave库）
        print("读取音频文件...")
        wav, sample_rate = read_audio_with_wave(wav_file)
        print(f"音频张量形状: {wav.shape}")
        print(f"采样率: {sample_rate} Hz")
        print(f"音频时长: {len(wav) / sample_rate:.2f} 秒\n")

        # 如果采样率不是16kHz，需要重采样
        if sample_rate != 16000:
            print(f"重采样: {sample_rate} Hz -> 16000 Hz")
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            wav = resampler(wav)
            sample_rate = 16000
            print(f"重采样后形状: {wav.shape}\n")

        # 检测语音时间戳
        print("检测语音时间戳...")
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=0.2,  # 与我们实现相同的阈值
            return_seconds=True,
        )

        print("\n" + "="*70)
        print("检测结果:")
        print(f"  检测到的语音段数: {len(speech_timestamps)}")

        if speech_timestamps:
            print("\n  语音时间段:")
            for i, ts in enumerate(speech_timestamps, 1):
                start = ts['start']
                end = ts['end']
                duration = end - start
                print(f"    段 {i}: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s)")
            print("\n  ✓ 成功检测到语音!")
        else:
            print("\n  ✗ 未检测到语音")

        print("="*70 + "\n")

        return speech_timestamps

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys

    # 默认测试增益后的录音
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
    else:
        wav_file = "/tmp/test_recording_gained.wav"

    if not Path(wav_file).exists():
        print(f"错误: 文件不存在 {wav_file}")
        print("请先运行 record_and_play.py 录制音频")
        print("\n用法:")
        print("  python test_official_silero.py                          # 测试增益后的录音")
        print("  python test_official_silero.py /tmp/test_recording.wav  # 测试原始录音")
    else:
        result = test_with_official_silero(wav_file)

        # 如果官方库也检测不到，打印建议
        if result is not None and len(result) == 0:
            print("\n建议:")
            print("  1. 尝试降低阈值 (在代码中改threshold=0.1)")
            print("  2. 检查麦克风是否正常工作")
            print("  3. 对比原始录音和增益后录音的检测结果")
            print("  4. 可能需要更换VAD算法（如WebRTC VAD）")

