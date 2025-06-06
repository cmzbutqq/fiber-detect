import torch
import sys
import subprocess

print("=== 系统信息 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("CUDA不可用！")
    print(f"PyTorch编译的CUDA版本: {torch.version.cuda}")

# 检查NVIDIA驱动
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n=== NVIDIA驱动信息 ===")
        print(result.stdout)
    else:
        print("nvidia-smi命令失败，可能没有安装NVIDIA驱动")
except FileNotFoundError:
    print("未找到nvidia-smi命令")
