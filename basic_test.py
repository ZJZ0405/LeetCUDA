import torch

# 测试 1: 基础操作
print("Test 1: Basic operations")
try:
    x = torch.rand(100, 100).cuda()
    y = torch.rand(100, 100).cuda()
    z = torch.matmul(x, y)
    print("✓ Basic matmul works")
except Exception as e:
    print(f"✗ Basic matmul failed: {e}")

# 测试 2: 卷积操作
print("\nTest 2: Convolution")
try:
    conv = torch.nn.Conv2d(3, 64, 3).cuda()
    x = torch.rand(1, 3, 224, 224).cuda()
    y = conv(x)
    print("✓ Convolution works")
except Exception as e:
    print(f"✗ Convolution failed: {e}")

# 测试 3: 注意力机制
print("\nTest 3: Attention")
try:
    x = torch.rand(2, 8, 512).cuda()
    attn = torch.nn.MultiheadAttention(512, 8).cuda()
    y, _ = attn(x, x, x)
    print("✓ Attention works")
except Exception as e:
    print(f"✗ Attention failed: {e}")