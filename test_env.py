import torch

# 检查 CUDA 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")

# 检查 PyTorch CUDA 版本
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version (PyTorch): {torch.version.cuda}")

# 检查 GPU 信息
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")