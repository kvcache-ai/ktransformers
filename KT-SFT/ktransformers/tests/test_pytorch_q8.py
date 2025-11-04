import torch

class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

in_features = 64
out_features = 128
model_fp32 = LinearModel(in_features, out_features)

model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

batch_size = 32
input_fp32 = torch.randn(1, batch_size, in_features)
output_int8 = model_int8(input_fp32)

print(f"输入形状: {input_fp32.shape}")
print(f"输出形状: {output_int8.shape}")

with torch.no_grad():
    output_fp32 = model_fp32(input_fp32)
    
print(f"FP32输出的前几个值: {output_fp32[0, :5]}")
print(f"INT8输出的前几个值: {output_int8[0, :5]}")

error = torch.abs(output_fp32 - output_int8).mean().item()
print(f"平均绝对误差: {error}")

print(f"量化前模型类型: {type(model_fp32.linear)}")
print(f"量化后模型类型: {type(model_int8.linear)}")