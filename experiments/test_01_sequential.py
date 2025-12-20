"""
实验1序列测试：依次测试不同模型
为了避免模型切换问题，分别运行不同模型的测试
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

def kill_vllm():
    """停止 vLLM 服务"""
    print("\n停止现有 vLLM 服务...")
    subprocess.run("pkill -f 'vllm.entrypoints.openai.api_server'", shell=True)
    time.sleep(5)

def start_vllm(model_name: str, tensor_parallel_size: int = 1):
    """启动 vLLM 服务"""
    print(f"\n启动 vLLM 服务: {model_name} (TP={tensor_parallel_size})...")

    cmd = f"""
    export HF_ENDPOINT=https://hf-mirror.com
    cd ~/tts
    nohup python -m vllm.entrypoints.openai.api_server \
        --model {model_name} \
        --served-model-name {model_name} \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --reasoning-parser qwen3 \
        --tensor-parallel-size {tensor_parallel_size} \
        > logs/vllm_{model_name.split('/')[-1]}.log 2>&1 &
    """

    subprocess.run(cmd, shell=True, executable='/bin/bash')

    # 等待服务启动
    print("等待服务启动（60秒）...")
    time.sleep(60)

    # 检查服务状态
    result = subprocess.run(
        "tail -20 ~/tts/logs/vllm_*.log | grep 'Application startup complete'",
        shell=True,
        capture_output=True,
        text=True
    )

    if "Application startup complete" in result.stdout:
        print("✓ vLLM 服务启动成功")
        return True
    else:
        print("✗ vLLM 服务启动失败，查看日志...")
        subprocess.run("tail -50 ~/tts/logs/vllm_*.log", shell=True)
        return False

def run_test_for_model(model_name: str):
    """为单个模型运行测试"""
    print(f"\n{'='*70}")
    print(f"开始测试模型: {model_name}")
    print(f"{'='*70}")

    # 修改实验脚本只测试当前模型
    model_short = model_name.split('/')[-1]

    cmd = f"""
    cd ~/tts
    source ~/miniconda3/bin/activate
    python3 -c "
import sys
sys.path.insert(0, '/home/azureuser/tts')
from experiments.test_01_model_comparison import Experiment1Runner, MODELS, MODEL_NAMES

# 临时修改为只测试当前模型
import experiments.test_01_model_comparison as exp1
exp1.MODELS = ['{model_name}']
exp1.MODEL_NAMES = {{'{model_name}': '{model_short}-local'}}

# 运行测试
runner = Experiment1Runner()
runner.run_all_tests(max_workers=3)
runner.generate_report()
print('\\n测试完成！')
"
    """

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)

    return result.returncode == 0

def main():
    """主流程"""
    models_to_test = [
        ("Qwen/Qwen3-8B", 1),   # 8B, 单卡
        ("Qwen/Qwen3-14B", 1),  # 14B, 单卡
        ("Qwen/Qwen3-32B", 2),  # 32B, 双卡
    ]

    results = {}

    for model_name, tp_size in models_to_test:
        print(f"\n\n{'#'*70}")
        print(f"# 测试配置: {model_name} (Tensor Parallel Size: {tp_size})")
        print(f"{'#'*70}\n")

        # 1. 停止现有服务
        kill_vllm()

        # 2. 启动新模型
        if not start_vllm(model_name, tp_size):
            print(f"✗ {model_name} 启动失败，跳过")
            results[model_name] = "启动失败"
            continue

        # 3. 运行测试
        success = run_test_for_model(model_name)
        results[model_name] = "成功" if success else "测试失败"

        print(f"\n{model_name} 测试完成: {results[model_name]}")

    # 汇总结果
    print(f"\n\n{'='*70}")
    print("所有测试完成！")
    print(f"{'='*70}\n")
    for model, status in results.items():
        print(f"  {model}: {status}")

if __name__ == "__main__":
    main()
