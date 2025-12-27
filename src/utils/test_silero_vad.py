"""
测试Silero VAD对增益后音频的检测效果
"""

import wave
import numpy as np
import onnxruntime
from pathlib import Path


def test_vad_on_wav(wav_file, model_path="silero_vad.onnx"):
    """
    测试Silero VAD对WAV文件的检测效果

    Args:
        wav_file: WAV文件路径
        model_path: Silero VAD模型路径
    """
    print(f"\n测试文件: {wav_file}")
    print("="*70)

    # 读取WAV文件
    with wave.open(wav_file, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    print(f"采样率: {sample_rate} Hz")
    print(f"总帧数: {n_frames}")
    print(f"时长: {n_frames / sample_rate:.2f} 秒\n")

    # 转换为numpy数组
    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # 计算音频统计
    amplitude_mean = np.abs(audio_int16).mean()
    amplitude_max = np.abs(audio_int16).max()
    amplitude_mean_pct = (amplitude_mean / 32768.0) * 100
    amplitude_max_pct = (amplitude_max / 32768.0) * 100
    rms = np.sqrt(np.mean(audio_float32 ** 2))

    print("音频统计:")
    print(f"  平均电平: {amplitude_mean_pct:.2f}%")
    print(f"  峰值电平: {amplitude_max_pct:.2f}%")
    print(f"  RMS: {rms:.4f}\n")

    # 初始化Silero VAD
    print("加载Silero VAD模型...")
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    print("模型加载完成\n")

    # 重采样到16kHz（如果需要）
    if sample_rate != 16000:
        print(f"重采样: {sample_rate} Hz -> 16000 Hz")
        import resampy
        audio_float32 = resampy.resample(
            audio_float32, sample_rate, 16000, filter='kaiser_fast'
        )
        sample_rate = 16000

    # 初始化state
    state = np.zeros((2, 1, 128), dtype=np.float32)

    # 分块处理
    chunk_size = 512
    num_chunks = len(audio_float32) // chunk_size
    vad_probs = []

    print(f"开始VAD检测 (共 {num_chunks} 个chunk)...\n")

    for i in range(num_chunks):
        chunk = audio_float32[i * chunk_size : (i + 1) * chunk_size]

        # 准备输入
        input_data = {
            'input': chunk.reshape(1, -1),
            'state': state,
            'sr': np.array(16000, dtype=np.int64),
        }

        # 推理
        outputs = session.run(None, input_data)
        speech_prob = outputs[0][0][0]
        state = outputs[1]

        vad_probs.append(speech_prob)

        # 每隔50个chunk显示一次
        if i % 50 == 0:
            print(f"  Chunk {i:4d}/{num_chunks}: VAD概率 = {speech_prob:.4f}")

    # 统计
    vad_probs = np.array(vad_probs)
    print("\n" + "="*70)
    print("VAD检测结果:")
    print(f"  平均概率: {vad_probs.mean():.4f}")
    print(f"  最大概率: {vad_probs.max():.4f}")
    print(f"  最小概率: {vad_probs.min():.4f}")
    print(f"  中位数: {np.median(vad_probs):.4f}")

    # 检测到语音的chunk数量（阈值0.2）
    speech_chunks = (vad_probs > 0.2).sum()
    speech_ratio = speech_chunks / len(vad_probs) * 100
    print(f"\n  检测到语音的chunk: {speech_chunks}/{len(vad_probs)} ({speech_ratio:.1f}%)")

    if speech_chunks > 0:
        print(f"  ✓ 成功检测到语音!")
    else:
        print(f"  ✗ 未检测到语音 (所有chunk的VAD概率都<0.2)")

    print("="*70 + "\n")


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
        print("  python test_silero_vad.py                          # 测试增益后的录音")
        print("  python test_silero_vad.py /tmp/test_recording.wav  # 测试原始录音")
    else:
        test_vad_on_wav(wav_file)
