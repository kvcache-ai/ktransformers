import torch

# 定义一个包含线性层的浮点模型
class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

# 创建浮点模型实例
in_features = 64
out_features = 128
model_fp32 = LinearModel(in_features, out_features)

# 创建量化模型实例
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,          # 原始浮点模型
    {torch.nn.Linear},   # 要量化的层类型集合
    dtype=torch.qint8    # 量化的目标数据类型
)

# 测试模型
batch_size = 32
input_fp32 = torch.randn(1, batch_size, in_features)  # 生成随机输入数据
output_int8 = model_int8(input_fp32)               # 通过量化模型运行数据

# 打印输出形状验证
print(f"输入形状: {input_fp32.shape}")
print(f"输出形状: {output_int8.shape}")

# 比较原始模型和量化模型的输出
with torch.no_grad():
    output_fp32 = model_fp32(input_fp32)
    
print(f"FP32输出的前几个值: {output_fp32[0, :5]}")
print(f"INT8输出的前几个值: {output_int8[0, :5]}")

# 计算平均误差
error = torch.abs(output_fp32 - output_int8).mean().item()
print(f"平均绝对误差: {error}")

# 打印模型类型信息
print(f"量化前模型类型: {type(model_fp32.linear)}")
print(f"量化后模型类型: {type(model_int8.linear)}")