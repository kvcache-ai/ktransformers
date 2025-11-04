import torch
import os

checkpoint_dir = "/home/yj/ktransformers/test_adapter/demo_adapter_KT_target_module/checkpoint-6600"  # 请将此处替换为实际文件夹路径

for filename in os.listdir(checkpoint_dir):
    file_path = os.path.join(checkpoint_dir, filename)
    if filename.endswith(('.pt', '.bin', '.pth')):
        try:
            loaded_data = torch.load(file_path)
            print(f"===== 文件: {filename} =====")
            print(f"数据类型: {type(loaded_data)}")
            
            if isinstance(loaded_data, dict):
                print("字典包含的键:", list(loaded_data.keys()))
                # 示例：打印优化器状态的部分参数（若为优化器文件）
                if "state" in loaded_data and "param_groups" in loaded_data:
                    print("优化器示例参数：")
                    print("param_groups 前2项:", loaded_data["param_groups"][:2])
                    print("state 中前2个参数的状态:", list(loaded_data["state"].items())[:2])
            elif isinstance(loaded_data, torch.nn.Module):
                print("模块参数列表:")
                for name, param in loaded_data.named_parameters():
                    print(f"参数名: {name}, 形状: {param.shape}")
            else:
                print("数据内容预览:", loaded_data)
        except Exception as e:
            print(f"读取 {filename} 时出错: {str(e)}")