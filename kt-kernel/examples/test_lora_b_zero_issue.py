#!/usr/bin/env python
# coding=utf-8
"""
测试当 LoRA B = 0 时是否会产生 NaN

假设：pt 文件中 LoRA B 全为 0（标准初始化），这可能导致 C++ 代码中的某些问题。
测试：将 LoRA B 设置为非零值后，问题是否消失。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch

DATA_PATH = "/mnt/data/lpl/kt_nan_debug_data.pt"

try:
    from kt_kernel import kt_kernel_ext

    HAS_KT_KERNEL = True
except ImportError:
    HAS_KT_KERNEL = False
    kt_kernel_ext = None


def silu(x):
    return x * torch.sigmoid(x)


def moe_sft_torch_forward(
    input_data,
    expert_ids,
    weights,
    gate_proj,
    up_proj,
    down_proj,
    gate_lora_a,
    gate_lora_b,
    up_lora_a,
    up_lora_b,
    down_lora_a,
    down_lora_b,
    lora_scaling,
):
    """PyTorch reference implementation."""
    qlen = input_data.shape[0]
    hidden_size = input_data.shape[1]
    num_experts_per_tok = expert_ids.shape[1]
    expert_num = gate_proj.shape[0]

    output = torch.zeros((qlen, hidden_size), dtype=input_data.dtype)

    for i in range(qlen):
        for j in range(num_experts_per_tok):
            expert_id = expert_ids[i, j].item()
            weight = weights[i, j].item()

            x = input_data[i : i + 1].float()

            # Gate
            gate_base = x @ gate_proj[expert_id].float().T
            gate_lora = (x @ gate_lora_a[expert_id].float().T) @ gate_lora_b[expert_id].float().T
            gate_out = gate_base + gate_lora * lora_scaling

            # Up
            up_base = x @ up_proj[expert_id].float().T
            up_lora = (x @ up_lora_a[expert_id].float().T) @ up_lora_b[expert_id].float().T
            up_out = up_base + up_lora * lora_scaling

            # Activation
            intermediate = silu(gate_out) * up_out

            # Down
            down_base = intermediate @ down_proj[expert_id].float().T
            down_lora = (intermediate @ down_lora_a[expert_id].float().T) @ down_lora_b[expert_id].float().T
            expert_output = down_base + down_lora * lora_scaling

            output[i] += weight * expert_output.squeeze(0).to(output.dtype)

    return output


def test_with_modified_lora_b():
    """测试将 LoRA B 设置为非零值后是否还有 NaN"""
    print("=" * 70)
    print("测试：将 LoRA B 设置为非零值")
    print("=" * 70)

    data = torch.load(DATA_PATH)

    # 配置
    real_expert_num = data["expert_num"]
    real_hidden_size = data["hidden_size"]
    real_intermediate_size = data["intermediate_size"]
    real_num_experts_per_tok = data["num_experts_per_tok"]
    real_qlen = data["input_data"].shape[0]
    real_lora_rank = data["gate_lora_a"].shape[1]
    real_lora_alpha = 16.0
    real_lora_scaling = real_lora_alpha / real_lora_rank

    print(f"\n配置:")
    print(f"  expert_num: {real_expert_num}")
    print(f"  hidden_size: {real_hidden_size}")
    print(f"  intermediate_size: {real_intermediate_size}")
    print(f"  qlen: {real_qlen}")
    print(f"  lora_rank: {real_lora_rank}")

    # 提取数据
    input_data = data["input_data"].contiguous()
    expert_ids = data["expert_ids"].contiguous()
    weights = data["weights"].contiguous()
    gate_proj = data["gate_proj"].contiguous()
    up_proj = data["up_proj"].contiguous()
    down_proj = data["down_proj"].contiguous()

    # 原始 LoRA 权重（B 全为 0）
    gate_lora_a = data["gate_lora_a"].contiguous()
    gate_lora_b = data["gate_lora_b"].contiguous()
    up_lora_a = data["up_lora_a"].contiguous()
    up_lora_b = data["up_lora_b"].contiguous()
    down_lora_a = data["down_lora_a"].contiguous()
    down_lora_b = data["down_lora_b"].contiguous()

    print(f"\n原始 LoRA B 权重检查:")
    print(f"  gate_lora_b: min={gate_lora_b.min().item():.6f}, max={gate_lora_b.max().item():.6f}")
    print(f"  up_lora_b: min={up_lora_b.min().item():.6f}, max={up_lora_b.max().item():.6f}")
    print(f"  down_lora_b: min={down_lora_b.min().item():.6f}, max={down_lora_b.max().item():.6f}")

    # 修改 LoRA B 为非零值（与 accuracy 测试相同）
    print("\n将 LoRA B 设置为非零随机值...")
    torch.manual_seed(42)
    gate_lora_b_nonzero = torch.randn_like(gate_lora_b) / 100
    up_lora_b_nonzero = torch.randn_like(up_lora_b) / 100
    down_lora_b_nonzero = torch.randn_like(down_lora_b) / 100

    print(f"\n修改后 LoRA B 权重:")
    print(f"  gate_lora_b: min={gate_lora_b_nonzero.min().item():.6f}, max={gate_lora_b_nonzero.max().item():.6f}")
    print(f"  up_lora_b: min={up_lora_b_nonzero.min().item():.6f}, max={up_lora_b_nonzero.max().item():.6f}")
    print(f"  down_lora_b: min={down_lora_b_nonzero.min().item():.6f}, max={down_lora_b_nonzero.max().item():.6f}")

    if not HAS_KT_KERNEL:
        print("\n[SKIP] kt_kernel_ext 不可用")
        return

    # 测试 1: 原始 LoRA B (全零)
    print("\n" + "=" * 70)
    print("测试 1: 原始 LoRA B (全零)")
    print("=" * 70)

    num_threads = 60
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [num_threads]
    CPUInfer = kt_kernel_ext.CPUInfer(pool_config)

    config = kt_kernel_ext.moe.MOESFTConfig()
    config.expert_num = real_expert_num
    config.num_experts_per_tok = real_num_experts_per_tok
    config.hidden_size = real_hidden_size
    config.intermediate_size = real_intermediate_size
    config.lora_rank = real_lora_rank
    config.lora_alpha = real_lora_alpha
    config.max_cache_depth = 1
    config.max_len = max(real_qlen * 2, 4096)
    config.layer_idx = data["layer_idx"]

    config.gate_proj = gate_proj.data_ptr()
    config.up_proj = up_proj.data_ptr()
    config.down_proj = down_proj.data_ptr()
    config.gate_lora_a = gate_lora_a.data_ptr()
    config.gate_lora_b = gate_lora_b.data_ptr()  # 原始全零
    config.up_lora_a = up_lora_a.data_ptr()
    config.up_lora_b = up_lora_b.data_ptr()  # 原始全零
    config.down_lora_a = down_lora_a.data_ptr()
    config.down_lora_b = down_lora_b.data_ptr()  # 原始全零
    config.pool = CPUInfer.backend_

    moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()
    CPUInfer.submit(moe.warm_up_task())
    CPUInfer.sync()

    bsz_tensor = torch.tensor([real_qlen], device="cpu")
    amx_output_zero = torch.zeros((real_qlen, real_hidden_size), dtype=torch.bfloat16).contiguous()

    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            real_num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            amx_output_zero.data_ptr(),
            False,
        )
    )
    CPUInfer.sync()

    nan_count_zero = torch.isnan(amx_output_zero).sum().item()
    print(f"\n结果 (LoRA B = 0):")
    print(f"  NaN 数量: {nan_count_zero}")

    # 测试 2: 修改后 LoRA B (非零)
    print("\n" + "=" * 70)
    print("测试 2: 修改后 LoRA B (非零)")
    print("=" * 70)

    # 重新创建 MOE 实例
    config2 = kt_kernel_ext.moe.MOESFTConfig()
    config2.expert_num = real_expert_num
    config2.num_experts_per_tok = real_num_experts_per_tok
    config2.hidden_size = real_hidden_size
    config2.intermediate_size = real_intermediate_size
    config2.lora_rank = real_lora_rank
    config2.lora_alpha = real_lora_alpha
    config2.max_cache_depth = 1
    config2.max_len = max(real_qlen * 2, 4096)
    config2.layer_idx = data["layer_idx"]

    config2.gate_proj = gate_proj.data_ptr()
    config2.up_proj = up_proj.data_ptr()
    config2.down_proj = down_proj.data_ptr()
    config2.gate_lora_a = gate_lora_a.data_ptr()
    config2.gate_lora_b = gate_lora_b_nonzero.data_ptr()  # 非零
    config2.up_lora_a = up_lora_a.data_ptr()
    config2.up_lora_b = up_lora_b_nonzero.data_ptr()  # 非零
    config2.down_lora_a = down_lora_a.data_ptr()
    config2.down_lora_b = down_lora_b_nonzero.data_ptr()  # 非零
    config2.pool = CPUInfer.backend_

    moe2 = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config2)
    CPUInfer.submit(moe2.load_weights_task())
    CPUInfer.sync()
    CPUInfer.submit(moe2.warm_up_task())
    CPUInfer.sync()

    amx_output_nonzero = torch.zeros((real_qlen, real_hidden_size), dtype=torch.bfloat16).contiguous()

    CPUInfer.submit(
        moe2.forward_sft_task(
            bsz_tensor.data_ptr(),
            real_num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            amx_output_nonzero.data_ptr(),
            False,
        )
    )
    CPUInfer.sync()

    nan_count_nonzero = torch.isnan(amx_output_nonzero).sum().item()
    print(f"\n结果 (LoRA B = 非零):")
    print(f"  NaN 数量: {nan_count_nonzero}")

    # PyTorch 参考
    print("\n" + "=" * 70)
    print("PyTorch 参考")
    print("=" * 70)

    torch_output_zero = moe_sft_torch_forward(
        input_data,
        expert_ids,
        weights,
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b,  # 原始全零
        up_lora_a,
        up_lora_b,
        down_lora_a,
        down_lora_b,
        real_lora_scaling,
    )
    torch_nan_zero = torch.isnan(torch_output_zero).sum().item()

    torch_output_nonzero = moe_sft_torch_forward(
        input_data,
        expert_ids,
        weights,
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b_nonzero,  # 非零
        up_lora_a,
        up_lora_b_nonzero,
        down_lora_a,
        down_lora_b_nonzero,
        real_lora_scaling,
    )
    torch_nan_nonzero = torch.isnan(torch_output_nonzero).sum().item()

    print(f"  PyTorch (LoRA B = 0) NaN: {torch_nan_zero}")
    print(f"  PyTorch (LoRA B = 非零) NaN: {torch_nan_nonzero}")

    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"  AMX (LoRA B = 0): {nan_count_zero} NaN")
    print(f"  AMX (LoRA B = 非零): {nan_count_nonzero} NaN")
    print(f"  PyTorch (LoRA B = 0): {torch_nan_zero} NaN")
    print(f"  PyTorch (LoRA B = 非零): {torch_nan_nonzero} NaN")

    if nan_count_zero > 0 and nan_count_nonzero == 0:
        print("\n*** 问题与 LoRA B = 0 相关！***")
        print("当 LoRA B 为全零时，C++ 代码产生 NaN")
        print("当 LoRA B 为非零时，C++ 代码正常")
    elif nan_count_zero > 0 and nan_count_nonzero > 0:
        print("\n*** 问题与 LoRA B 值无关 ***")
        print("无论 LoRA B 是否为零，都有 NaN")
    else:
        print("\n*** 两种情况都没有 NaN ***")


if __name__ == "__main__":
    test_with_modified_lora_b()
