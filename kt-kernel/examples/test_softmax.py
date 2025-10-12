
import torch
from torch import nn


def load_fp16_tensor(file_path, shape):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float16)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

a = load_fp16_tensor("csrc/ktransformers_ext/build/before_softmax", (64,1024))
check = load_fp16_tensor("csrc/ktransformers_ext/build/after_softmax", (64,1024))


a = nn.functional.softmax(a, dim=-1, dtype=torch.float16)
diff = torch.abs(a - check).max()

print(a)
print(check)
print(diff)


