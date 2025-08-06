import torch

print("="*50)
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    print(f"✅ 设备检测: {name}")
    print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    
    if "4070" in name:
        print("🎉 RTX 4070 Super 优化就绪!")
else:
    print("❌ 未检测到CUDA设备")
print("="*50)
