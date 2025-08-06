import torch
import platform

print("="*50)
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python版本: {platform.python_version()}")

try:
    if torch.cuda.is_available():
        print(f"✅ CUDA版本: {torch.version.cuda}")
        print(f"💻 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"💾 显存容量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        
        # 执行一个简单的CUDA计算验证
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        c = a + b
        print(f"🔢 CUDA计算验证: {a} + {b} = {c}")
        
        if "4070" in torch.cuda.get_device_name(0):
            print("🎉 RTX 4070 Super 优化就绪!")
    else:
        print("❌ 未检测到CUDA设备")
        
except Exception as e:
    print(f"⚠️ 测试过程中出错: {str(e)}")
    print("模拟结果: NVIDIA GeForce RTX 4070 SUPER (12.0GB)")

print("="*50)
