"""
音频设备诊断工具
用于检查PyAudio输入设备配置和音频采集
"""

import sys
import time
import numpy as np
import pyaudio


def list_audio_devices():
    """列出所有音频设备"""
    p = pyaudio.PyAudio()

    print("\n" + "="*70)
    print("所有音频设备列表:")
    print("="*70)

    device_count = p.get_device_count()
    input_devices = []

    for i in range(device_count):
        info = p.get_device_info_by_index(i)

        # 只显示输入设备
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))
            print(f"\n[设备 {i}]")
            print(f"  名称: {info['name']}")
            print(f"  输入通道数: {info['maxInputChannels']}")
            print(f"  默认采样率: {int(info['defaultSampleRate'])} Hz")

    print("\n" + "="*70)

    # 显示默认输入设备
    try:
        default_info = p.get_default_input_device_info()
        print(f"\n当前默认输入设备: [{default_info['index']}] {default_info['name']}")
    except Exception as e:
        print(f"\n⚠️  获取默认输入设备失败: {e}")

    print("="*70 + "\n")

    p.terminate()
    return input_devices


def test_device(device_index, duration=5):
    """测试指定设备的音频采集"""
    print(f"\n开始测试设备 {device_index} (录制 {duration} 秒)...")
    print("请对着麦克风说话...\n")

    p = pyaudio.PyAudio()

    try:
        # 打开音频流
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index if device_index >= 0 else None,
            frames_per_buffer=512,
        )

        # 录制音频
        frames = []
        num_chunks = int(16000 / 512 * duration)

        for i in range(num_chunks):
            try:
                data = stream.read(512, exception_on_overflow=False)
                frames.append(data)

                # 显示实时音频电平
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                amplitude = np.abs(audio_int16).mean()
                amplitude_pct = (amplitude / 32768.0) * 100

                # 每隔10个chunk显示一次
                if i % 10 == 0:
                    bar_length = int(amplitude_pct / 2)  # 最大50个字符
                    bar = "█" * bar_length
                    print(f"\r音频电平: {amplitude_pct:5.1f}% [{bar:<50}]", end="", flush=True)
            except Exception as e:
                print(f"\n读取音频数据出错: {e}")
                break

        print("\n")

        # 关闭流
        stream.stop_stream()
        stream.close()

        # 分析录制的音频
        all_audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        amplitude_mean = np.abs(all_audio).mean()
        amplitude_max = np.abs(all_audio).max()
        amplitude_mean_pct = (amplitude_mean / 32768.0) * 100
        amplitude_max_pct = (amplitude_max / 32768.0) * 100

        print(f"录制完成! 音频统计:")
        print(f"  平均电平: {amplitude_mean_pct:.2f}%")
        print(f"  峰值电平: {amplitude_max_pct:.2f}%")

        if amplitude_max_pct < 1.0:
            print("  ⚠️  警告: 音频信号极弱,可能没有正确录制到声音")
        elif amplitude_max_pct < 10.0:
            print("  ⚠️  警告: 音频信号较弱,建议检查麦克风设置或选择其他设备")
        else:
            print("  ✓ 音频信号正常")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
    finally:
        p.terminate()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("PyAudio 音频设备诊断工具")
    print("="*70)

    # 列出所有输入设备
    input_devices = list_audio_devices()

    if not input_devices:
        print("❌ 未找到任何输入设备!")
        return

    # 交互式选择设备测试
    while True:
        print("\n请选择要测试的设备 (输入设备编号, 或输入 'q' 退出):")
        choice = input("> ").strip()

        if choice.lower() == 'q':
            break

        try:
            device_index = int(choice)

            # 检查设备是否存在
            valid = False
            for idx, info in input_devices:
                if idx == device_index:
                    valid = True
                    break

            if not valid:
                print(f"❌ 设备 {device_index} 不是有效的输入设备!")
                continue

            # 测试设备
            test_device(device_index, duration=5)

        except ValueError:
            print("❌ 请输入有效的设备编号!")
        except KeyboardInterrupt:
            print("\n\n测试已取消")
            break

    print("\n诊断结束。\n")


if __name__ == "__main__":
    main()
