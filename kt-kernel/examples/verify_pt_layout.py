#!/usr/bin/env python
# coding=utf-8
"""
验证 PT 文件中 LoRA 权重的内存布局

用于调试 Bug-A: Expert 17-24 的 BufferB 出现垃圾数据问题
关键假设: 代码期望 gate_lora_b 的布局是 [expert_num, intermediate_size, lora_rank]
如果实际布局不同，会导致读取错误的内存位置
"""
import torch
import sys
import numpy as np


def verify_lora_layout(data_path: str):
    """验证 LoRA 权重的内存布局"""
    print("=" * 70)
    print("LoRA B 权重布局验证")
    print("=" * 70)
    print(f"数据文件: {data_path}")

    data = torch.load(data_path)

    # 打印配置信息
    print(f"\n[配置信息]")
    print(f"  expert_num: {data.get('expert_num', 'N/A')}")
    print(f"  hidden_size: {data.get('hidden_size', 'N/A')}")
    print(f"  intermediate_size: {data.get('intermediate_size', 'N/A')}")
    print(f"  num_experts_per_tok: {data.get('num_experts_per_tok', 'N/A')}")

    # 检查 LoRA B 权重
    lora_b_tensors = ["gate_lora_b", "up_lora_b", "down_lora_b"]
    lora_a_tensors = ["gate_lora_a", "up_lora_a", "down_lora_a"]

    print(f"\n[LoRA A 权重布局]")
    for name in lora_a_tensors:
        if name not in data:
            print(f"  {name}: NOT FOUND")
            continue

        tensor = data[name]
        print(f"\n  {name}:")
        print(f"    shape: {tensor.shape}")
        print(f"    stride: {tensor.stride()}")
        print(f"    is_contiguous: {tensor.is_contiguous()}")
        print(f"    dtype: {tensor.dtype}")

        # 对于 [expert_num, lora_rank, hidden_size] 布局
        if len(tensor.shape) == 3:
            e, r, h = tensor.shape
            expected_stride = (r * h, h, 1)
            matches = tensor.stride() == expected_stride
            print(f"    expected stride (for [E,R,H]): {expected_stride}")
            print(f"    {'✓ CORRECT' if matches else '✗ WRONG - 可能是转置布局!'}")

    print(f"\n[LoRA B 权重布局] ← 关键检查")
    for name in lora_b_tensors:
        if name not in data:
            print(f"  {name}: NOT FOUND")
            continue

        tensor = data[name]
        print(f"\n  {name}:")
        print(f"    shape: {tensor.shape}")
        print(f"    stride: {tensor.stride()}")
        print(f"    is_contiguous: {tensor.is_contiguous()}")
        print(f"    dtype: {tensor.dtype}")

        # 验证 stride 是否符合 [expert, n, k] 布局
        if len(tensor.shape) == 3:
            e, n, k = tensor.shape
            expected_stride = (n * k, k, 1)
            matches = tensor.stride() == expected_stride
            print(f"    expected stride (for [E,N,K]): {expected_stride}")
            if matches:
                print(f"    ✓ CORRECT - 代码期望的布局 [expert, intermediate/hidden, lora_rank]")
            else:
                print(f"    ✗ WRONG - 可能是转置布局!")
                # 检查是否是转置后的布局
                transposed_stride = (n * k, 1, n)
                if tensor.stride() == transposed_stride:
                    print(f"    ⚠️ 看起来像是 [expert, lora_rank, intermediate/hidden] 的转置视图!")

        # 检查具体 expert 的数据
        print(f"\n    Expert 数据对比 (关注 17-24 vs 25):")
        for exp_id in [16, 17, 18, 19, 24, 25, 26]:
            if exp_id >= tensor.shape[0]:
                continue
            exp_data = tensor[exp_id]
            nan_count = torch.isnan(exp_data).sum().item()
            zero_count = (exp_data == 0).sum().item()
            total = exp_data.numel()
            non_zero = total - zero_count

            # 计算非零值的统计
            non_zero_mask = exp_data != 0
            if non_zero_mask.any():
                non_zero_vals = exp_data[non_zero_mask].float()
                min_val = non_zero_vals.min().item()
                max_val = non_zero_vals.max().item()
                mean_val = non_zero_vals.mean().item()
            else:
                min_val = max_val = mean_val = 0.0

            status = "⚠️ 问题区域" if 17 <= exp_id <= 24 else ("✓ 正常" if exp_id == 25 else "")
            print(
                f"      Expert {exp_id:2d}: nan={nan_count:3d}, zero={zero_count:5d}/{total}, "
                f"non_zero={non_zero:5d}, range=[{min_val:+.4f}, {max_val:+.4f}] {status}"
            )

    # 验证 C++ 代码期望的内存访问模式
    print(f"\n[内存访问模式验证]")

    if "gate_lora_b" in data:
        tensor = data["gate_lora_b"]
        e, n, k = tensor.shape

        print(f"\n  gate_lora_b 内存布局分析:")
        print(f"    shape = [{e}, {n}, {k}]")
        print(f"    stride = {tensor.stride()}")

        # C++ 代码期望:
        # expert_src = src + expert_idx * n * k
        # element = expert_src[r * k + c]  for r in [0,n), c in [0,k)
        print(f"\n  C++ 代码期望的访问模式:")
        print(f"    expert_src = src + expert_idx * {n} * {k}")
        print(f"    element[r,c] = expert_src[r * {k} + c]")

        # 验证实际布局
        flat = tensor.view(-1)
        print(f"\n  验证 Expert 17 的第一行数据:")
        exp_17 = tensor[17]
        print(f"    exp_17[0, :8] = {exp_17[0, :min(8, k)].tolist()}")

        # 使用 C++ 的访问方式读取
        offset_17 = 17 * n * k
        print(f"    flat[{offset_17}:{offset_17+8}] = {flat[offset_17:offset_17+8].tolist()}")

        # 检查是否一致
        cpp_view = flat[offset_17 : offset_17 + n * k].view(n, k)
        matches = torch.allclose(cpp_view, exp_17)
        print(f"    C++ 访问与 Python 索引一致: {'✓ YES' if matches else '✗ NO - 布局问题!'}")

        if not matches:
            print(f"\n  ⚠️ 发现布局不一致!")
            print(f"    Python tensor[17] 的数据与 flat[17*n*k:(17+1)*n*k] 不同")
            print(f"    这可能是因为 tensor 不是 contiguous 或有 transpose 操作")

    # 创建简单的可视化比较
    print(f"\n[Expert 17 vs Expert 25 的原始数据 (前 32 个元素)]")
    if "gate_lora_b" in data:
        tensor = data["gate_lora_b"]
        e, n, k = tensor.shape

        for exp_id in [17, 25]:
            if exp_id >= e:
                continue
            exp_data = tensor[exp_id].flatten()[:32]
            print(f"  Expert {exp_id}: {[f'{x:.4f}' for x in exp_data.float().tolist()]}")


def main():
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "/mnt/data/lpl/kt_nan_debug_data.pt"

    import os

    if not os.path.exists(data_path):
        print(f"错误: 文件不存在 {data_path}")
        return 1

    verify_lora_layout(data_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
