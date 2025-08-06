import torch
import platform

print("="*50)
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python版本: {platform.python_version()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    print(f"💻 设备名称: {torch.cuda.get_device_name(0)}")
    print(f"💾 显存容量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    
    if "4070" in torch.cuda.get_device_name(0):
        print("🎉 RTX 4070 Super 优化就绪!")
else:
    print("❌ 未检测到CUDA设备")
    print("模拟结果: NVIDIA GeForce RTX 4070 SUPER (12.0GB)")
    
print("="*50)
