"""
对比官方silero-vad和我们的ONNX实现在同一音频上的输出
"""

import wave
import numpy as np
import torch
import onnxruntime
from pathlib import Path


def read_audio(wav_file):
    """读取WAV文件"""
    with wave.open(wav_file, 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    if channels > 1:
        audio_int16 = audio_int16[::channels]

    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return audio_float32, sample_rate


def test_official_model(audio_float32, model):
    """测试官方PyTorch模型"""
    chunk_size = 512
    num_chunks = len(audio_float32) // chunk_size

    audio_tensor = torch.from_numpy(audio_float32)

    probs = []
    for i in range(min(num_chunks, 50)):  # 只测试前50个chunk
        chunk = audio_tensor[i * chunk_size : (i + 1) * chunk_size]

        # 官方模型推理
        with torch.no_grad():
            prob = model(chunk, 16000).item()

        probs.append(prob)

        if i < 20 or prob > 0.1:  # 显示前20个chunk或高概率chunk
            print(f"  官方模型 Chunk {i:3d}: prob={prob:.6f}")

    return probs


def test_onnx_model(audio_float32, session):
    """测试ONNX模型"""
    chunk_size = 512
    num_chunks = len(audio_float32) // chunk_size

    # 初始化state
    state = np.zeros((2, 1, 128), dtype=np.float32)

    probs = []
    for i in range(min(num_chunks, 50)):
        chunk = audio_float32[i * chunk_size : (i + 1) * chunk_size]

        # 准备输入
        input_data = {
            'input': chunk.reshape(1, -1).astype(np.float32),
            'state': state,
            'sr': np.array(16000, dtype=np.int64),
        }

        # ONNX推理
        outputs = session.run(None, input_data)
        prob = outputs[0][0][0]
        state = outputs[1]

        probs.append(prob)

        if i < 20 or prob > 0.1:
            print(f"  ONNX模型  Chunk {i:3d}: prob={prob:.6f}")

    return probs


def main():
    wav_file = "/tmp/test_recording_gained.wav"

    print(f"\n测试文件: {wav_file}")
    print("=" * 70)

    # 读取音频
    audio_float32, sample_rate = read_audio(wav_file)
    print(f"采样率: {sample_rate} Hz")
    print(f"音频时长: {len(audio_float32) / sample_rate:.2f} 秒")
    print(f"音频形状: {audio_float32.shape}\n")

    # 加载官方模型
    print("加载官方PyTorch模型...")
    from silero_vad import load_silero_vad
    official_model = load_silero_vad()
    print("✓ 官方模型加载完成\n")

    # 加载ONNX模型
    print("加载ONNX模型...")
    model_path = Path(__file__).parent.parent.parent / "silero_vad.onnx"
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    onnx_session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )
    print("✓ ONNX模型加载完成\n")

    # 测试官方模型
    print("=" * 70)
    print("测试官方PyTorch模型（前50个chunk）:")
    print("=" * 70)
    official_probs = test_official_model(audio_float32, official_model)

    print("\n" + "=" * 70)
    print("测试ONNX模型（前50个chunk）:")
    print("=" * 70)
    onnx_probs = test_onnx_model(audio_float32, onnx_session)

    # 统计对比
    print("\n" + "=" * 70)
    print("统计对比:")
    print("=" * 70)
    print(f"官方模型:")
    print(f"  平均概率: {np.mean(official_probs):.6f}")
    print(f"  最大概率: {np.max(official_probs):.6f}")
    print(f"  >0.2的chunk: {np.sum(np.array(official_probs) > 0.2)}/{len(official_probs)}")

    print(f"\nONNX模型:")
    print(f"  平均概率: {np.mean(onnx_probs):.6f}")
    print(f"  最大概率: {np.max(onnx_probs):.6f}")
    print(f"  >0.2的chunk: {np.sum(np.array(onnx_probs) > 0.2)}/{len(onnx_probs)}")

    # 差异分析
    print(f"\n差异分析:")
    diff = np.abs(np.array(official_probs) - np.array(onnx_probs))
    print(f"  平均绝对差异: {np.mean(diff):.6f}")
    print(f"  最大绝对差异: {np.max(diff):.6f}")

    if np.max(official_probs) > 0.2 and np.max(onnx_probs) < 0.1:
        print(f"\n⚠️  官方模型能检测到语音，但ONNX模型不能！")
        print(f"    这说明我们的ONNX实现有问题。")
    elif np.mean(diff) < 0.01:
        print(f"\n✓ 两个模型输出基本一致")
    else:
        print(f"\n⚠️  两个模型输出存在明显差异")


if __name__ == "__main__":
    main()
