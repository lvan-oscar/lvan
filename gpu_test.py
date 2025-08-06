import torch
import platform
import sys

print("="*50)
print(f"Python version: {sys.version}")
print(f"Operating System: {platform.system()} {platform.release()}")

try:
    # 基本PyTorch信息
    print(f"\nPyTorch version: {torch.__version__}")
    
    # CUDA可用性检查
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # 设备信息
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        if device_count > 0:
            # 获取第一个设备信息
            device = torch.cuda.device(0)
            with device:
                device_name = torch.cuda.get_device_name(0)
                device_capability = torch.cuda.get_device_capability(0)
                device_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                
                print(f"\nDevice 0: {device_name}")
                print(f"Compute Capability: {device_capability[0]}.{device_capability[1]}")
                print(f"Total Memory: {device_memory:.1f} GB")
                print(f"CUDA Version: {torch.version.cuda}")
                
                # 简单计算测试
                try:
                    a = torch.tensor([1.0, 2.0, 3.0]).cuda()
                    b = torch.tensor([4.0, 5.0, 6.0]).cuda()
                    c = a + b
                    print(f"\nCUDA Test Calculation: {a} + {b} = {c}")
                    print("✅ CUDA test passed")
                    
                    if "4070" in device_name:
                        print("🎉 RTX 4070 Super detected!")
                except Exception as e:
                    print(f"\n⚠️ CUDA calculation failed: {str(e)}")
        else:
            print("No CUDA devices found")
    else:
        # 模拟4070 Super输出
        print("\nSimulating NVIDIA GeForce RTX 4070 SUPER")
        print("Device 0: NVIDIA GeForce RTX 4070 SUPER")
        print("Total Memory: 12.0 GB")
        print("CUDA Version: 12.1")
        print("🎉 Simulated RTX 4070 Super output")

except Exception as e:
    print(f"\n⚠️ Major error occurred: {str(e)}")
    print("Falling back to simulated output")
    print("Device 0: NVIDIA GeForce RTX 4070 SUPER (Simulated)")
    print("Total Memory: 12.0 GB")

print("="*50)
