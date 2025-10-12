import torch
from torch_attention import apply_rotary_pos_emb, DeepseekV3YarnRotaryEmbedding, DeepseekV3RotaryEmbedding

batch_size  = 1
num_heads   = 1
seq_len     = 1024
rope_size   = 64
theta       = 10000

max_position_embeddings =  163840

scaling_cfg = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
}

rotary_emb = DeepseekV3YarnRotaryEmbedding(
    rope_size,
    max_position_embeddings=max_position_embeddings,
    scaling_factor=scaling_cfg["factor"],
    base=theta,
    beta_fast=scaling_cfg["beta_fast"],
    beta_slow=scaling_cfg["beta_slow"],
    mscale=scaling_cfg["mscale"],
    mscale_all_dim=scaling_cfg["mscale_all_dim"],
    original_max_position_embeddings=scaling_cfg["original_max_position_embeddings"],
)


def load_fp16_tensor(file_path, shape):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float16)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

def load_fp32_tensor(file_path, shape):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float32)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

#q_pe = torch.randn(batch_size, num_heads, seq_len, rope_size, dtype=torch.float32)
#k_pe = torch.randn_like(q_pe)

q_pe = load_fp16_tensor("csrc/ktransformers_ext/build/before_rope",(batch_size, num_heads, seq_len, rope_size)) 
# k_pe = torch.ones_like(q_pe) 
k_pe = load_fp16_tensor("csrc/ktransformers_ext/build/before_rope",(batch_size, num_heads, seq_len, rope_size)) 
print(q_pe)

check = load_fp16_tensor("csrc/ktransformers_ext/build/after_rope",(batch_size, num_heads, seq_len, rope_size))




def torch_rope(q, k):
    cos, sin = rotary_emb(q, seq_len=seq_len)

    cos_to_check = load_fp32_tensor("csrc/ktransformers_ext/build/cos",(seq_len, rope_size//2))
    sin_to_check = load_fp32_tensor("csrc/ktransformers_ext/build/sin",(seq_len, rope_size//2))


    

    sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(0)
    q2, k2 = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    return q2, k2

q2, k2 = torch_rope(q_pe, k_pe)
print(q2,k2)
print(check)

diff = torch.abs(q2 - check).max()


print(diff)

# print(q2,k2)

# print_tensor(q2, 'q_py.out')
# print_tensor(k2, 'k_py.out')

