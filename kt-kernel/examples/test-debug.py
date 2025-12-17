import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
import torch
import ctypes
from kt_kernel import kt_kernel_ext
from kt_kernel_ext.moe import MOEConfig, MOE, AMXBF16_MOE, AMXInt8_MOE, AMXInt4_MOE, AMXInt4_1_MOE

intermediate_size_full = 2048
moe_intermediate_size = 3072
hidden_size = 7168
experts_num = 256
num_experts_per_tok = 8
cpu_infer = kt_kernel_ext.CPUInfer(97)

up = torch.empty(experts_num, intermediate_size_full, hidden_size, dtype=torch.bfloat16, device="cpu")

gate = torch.empty(experts_num, intermediate_size_full, hidden_size, dtype=torch.bfloat16, device="cpu")

down = torch.empty(experts_num, hidden_size, intermediate_size_full, dtype=torch.bfloat16, device="cpu")

gate_ptr = ctypes.addressof(ctypes.cast(gate.data_ptr(), ctypes.POINTER(ctypes.c_uint64)).contents)
up_ptr = ctypes.addressof(ctypes.cast(up.data_ptr(), ctypes.POINTER(ctypes.c_uint64)).contents)
down_ptr = ctypes.addressof(ctypes.cast(down.data_ptr(), ctypes.POINTER(ctypes.c_uint64)).contents)
moe_config = MOEConfig(
    experts_num,
    num_experts_per_tok,
    hidden_size,
    moe_intermediate_size,
)
moe_config.layer_idx = 45
moe_config.pool = cpu_infer.backend_
moe_config.max_len = 1024  # TODO(zbx): multi cuda graph
moe_config.gate_proj = gate_ptr
moe_config.up_proj = up_ptr
moe_config.down_proj = down_ptr
moe_config.path = ""
moe = AMXInt4_MOE(moe_config)
