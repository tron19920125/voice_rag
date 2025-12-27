"""
录音并播放脚本
用于测试麦克风录音质量
"""

import wave
import numpy as np
import pyaudio


def record_audio(duration=5, device_index=None):
    """
    录制音频

    Args:
        duration: 录音时长（秒）
        device_index: 输入设备索引（None=默认）

    Returns:
        录制的音频数据（bytes）
    """
    print(f"\n开始录音 {duration} 秒...")
    print("请对着麦克风说话...\n")

    # 音频参数
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )

    frames = []
    num_chunks = int(RATE / CHUNK * duration)

    for i in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        # 显示实时进度
        if i % 10 == 0:
            progress = (i / num_chunks) * 100
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            amplitude = np.abs(audio_int16).mean()
            amplitude_pct = (amplitude / 32768.0) * 100

            bar_length = int(progress / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"\r进度: [{bar}] {progress:.0f}%  电平: {amplitude_pct:.1f}%  ", end="", flush=True)

    print("\n\n录音完成!\n")

    # 停止和关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 返回音频数据
    return b"".join(frames), RATE, CHANNELS


def save_wav(filename, audio_data, sample_rate, channels):
    """保存为WAV文件"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    print(f"已保存到: {filename}")


def apply_agc_gain(audio_data, target_rms=0.15, max_gain=50.0):
    """
    应用自动增益控制（AGC）

    Args:
        audio_data: 音频数据（bytes, int16格式）
        target_rms: 目标RMS（默认0.15，即15%电平）
        max_gain: 最大增益倍数（默认50）

    Returns:
        增益后的音频数据（bytes）
    """
    # 转换为numpy数组
    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # 计算原始RMS
    original_rms = np.sqrt(np.mean(audio_float32 ** 2))

    # 计算需要的增益
    if original_rms > 0.0001:
        gain = target_rms / original_rms
        gain = min(gain, max_gain)  # 限制最大增益

        # 应用增益
        audio_float32 = audio_float32 * gain

        # 软限幅（防止削波）
        audio_float32 = np.tanh(audio_float32)

        # 记录增益后的RMS
        gained_rms = np.sqrt(np.mean(audio_float32 ** 2))

        print(f"AGC增益: 原始RMS={original_rms:.4f} 增益={gain:.1f}x 输出RMS={gained_rms:.4f}")

    # 转回int16
    audio_int16 = (audio_float32 * 32768.0).astype(np.int16)
    return audio_int16.tobytes()


def play_audio(audio_data, sample_rate, channels, apply_gain=False):
    """播放音频"""
    # 应用增益（如果需要）
    if apply_gain:
        print("\n应用AGC增益...")
        audio_data = apply_agc_gain(audio_data)

    print("\n开始播放录音...\n")

    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        output=True,
    )

    # 分块播放
    CHUNK = 1024
    offset = 0
    while offset < len(audio_data):
        chunk = audio_data[offset:offset + CHUNK]
        stream.write(chunk)
        offset += CHUNK

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("播放完成!\n")


def analyze_audio(audio_data):
    """分析音频信号"""
    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

    amplitude_mean = np.abs(audio_int16).mean()
    amplitude_max = np.abs(audio_int16).max()
    amplitude_mean_pct = (amplitude_mean / 32768.0) * 100
    amplitude_max_pct = (amplitude_max / 32768.0) * 100

    print("音频分析:")
    print(f"  平均电平: {amplitude_mean_pct:.2f}%")
    print(f"  峰值电平: {amplitude_max_pct:.2f}%")

    if amplitude_max_pct < 1.0:
        print("  ⚠️  信号极弱，可能没有正确录到声音")
    elif amplitude_max_pct < 10.0:
        print("  ⚠️  信号较弱，建议调高麦克风音量")
    else:
        print("  ✓ 信号正常")

    return amplitude_mean_pct, amplitude_max_pct


def main():
    """主函数"""
    print("\n" + "="*70)
    print("录音测试工具")
    print("="*70)

    # 列出设备
    p = pyaudio.PyAudio()
    print("\n可用输入设备:")
    input_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info['name']))
            print(f"  [{i}] {info['name']}")

    try:
        default_info = p.get_default_input_device_info()
        default_idx = default_info['index']
        print(f"\n默认输入设备: [{default_idx}] {default_info['name']}")
    except:
        default_idx = None

    p.terminate()

    # 选择设备
    print("\n输入设备编号 (直接回车使用默认设备):")
    choice = input("> ").strip()

    if choice:
        device_index = int(choice)
    else:
        device_index = default_idx

    # 录音
    audio_data, sample_rate, channels = record_audio(duration=5, device_index=device_index)

    # 分析
    analyze_audio(audio_data)

    # 保存原始录音
    filename_original = "/tmp/test_recording.wav"
    save_wav(filename_original, audio_data, sample_rate, channels)

    # 保存增益后的录音
    print("\n生成增益后的录音...")
    audio_data_gained = apply_agc_gain(audio_data)
    filename_gained = "/tmp/test_recording_gained.wav"
    save_wav(filename_gained, audio_data_gained, sample_rate, channels)

    # 播放选项
    print("\n播放选项:")
    print("  1. 播放原始录音")
    print("  2. 播放增益后的录音（AGC）")
    print("  3. 对比播放（先原始，再增益）")
    print("  4. 不播放")
    choice = input("> ").strip()

    if choice == '1':
        play_audio(audio_data, sample_rate, channels, apply_gain=False)
    elif choice == '2':
        play_audio(audio_data, sample_rate, channels, apply_gain=True)
    elif choice == '3':
        print("\n【原始录音】")
        play_audio(audio_data, sample_rate, channels, apply_gain=False)
        print("\n【增益后录音】")
        play_audio(audio_data, sample_rate, channels, apply_gain=True)

    print("\n测试完成!\n")


if __name__ == "__main__":
    main()
