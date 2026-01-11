#!/usr/bin/env python
# coding=utf-8
"""
验证 TP 分区数据是否正确复制

测试假设：TP_MOE_SFT::update_lora_weights 中的分区逻辑有 bug，
导致 Expert 17-24 的数据被错误复制。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch
import numpy as np

DATA_PATH = "/mnt/data/lpl/kt_nan_debug_data.pt"


def simulate_partition_copy():
    """模拟 C++ 中的分区复制逻辑"""
    print("=" * 70)
    print("模拟 TP 分区复制逻辑")
    print("=" * 70)

    data = torch.load(DATA_PATH)

    expert_num = data["expert_num"]  # 64
    lora_rank = data["gate_lora_a"].shape[1]  # 8
    full_intermediate_size = data["intermediate_size"]  # 1408

    gate_lora_b = data["gate_lora_b"]  # [64, 1408, 8]

    print(f"\n原始数据:")
    print(f"  expert_num: {expert_num}")
    print(f"  intermediate_size: {full_intermediate_size}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  gate_lora_b shape: {gate_lora_b.shape}")

    # 模拟 tp_count = 1 的情况
    tp_count = 1
    tp_intermediate = full_intermediate_size // tp_count  # 1408

    print(f"\nTP 分区参数:")
    print(f"  tp_count: {tp_count}")
    print(f"  tp_intermediate: {tp_intermediate}")

    # 模拟 C++ 中的分区复制
    lora_b_slice_size = tp_intermediate * lora_rank  # 1408 * 8 = 11264
    print(f"  lora_b_slice_size: {lora_b_slice_size}")

    # 将 gate_lora_b 转为 flat 格式（与 C++ 中的内存布局相同）
    gate_lora_b_flat = gate_lora_b.view(-1).float().numpy()  # [64 * 1408 * 8]
    print(f"  gate_lora_b_flat size: {len(gate_lora_b_flat)}")

    # 分配分区数据空间
    partitioned_size = expert_num * lora_b_slice_size
    partitioned_gate_lora_b = np.zeros(partitioned_size, dtype=np.float32)
    print(f"  partitioned_gate_lora_b size: {len(partitioned_gate_lora_b)}")

    # 模拟 memcpy 循环
    for i in range(tp_count):  # i = 0
        for expert_id in range(expert_num):
            # 目标偏移
            dst_offset = expert_id * lora_b_slice_size

            # 源偏移 (C++ 代码中的公式)
            src_offset = expert_id * full_intermediate_size * lora_rank + i * lora_b_slice_size
            # = expert_id * 1408 * 8 + 0 * 11264
            # = expert_id * 11264

            # 复制数据
            partitioned_gate_lora_b[dst_offset : dst_offset + lora_b_slice_size] = gate_lora_b_flat[
                src_offset : src_offset + lora_b_slice_size
            ]

    # 验证分区数据与原始数据是否一致
    print("\n" + "=" * 70)
    print("验证分区数据")
    print("=" * 70)

    all_correct = True
    for expert_id in range(expert_num):
        # 原始数据
        original = gate_lora_b[expert_id].view(-1).float().numpy()

        # 分区数据
        partitioned = partitioned_gate_lora_b[expert_id * lora_b_slice_size : (expert_id + 1) * lora_b_slice_size]

        # 比较
        if not np.allclose(original, partitioned, rtol=1e-5, atol=1e-5):
            print(f"  Expert {expert_id}: *** MISMATCH ***")
            diff = np.abs(original - partitioned)
            print(f"    max diff: {diff.max()}")
            all_correct = False
        elif expert_id in range(17, 25):  # 重点关注 Expert 17-24
            print(f"  Expert {expert_id}: OK (suspect range)")
        elif expert_id in [0, 8, 16, 32, 48, 63]:  # 采样其他 expert
            print(f"  Expert {expert_id}: OK")

    if all_correct:
        print("\n*** 所有 Expert 的分区数据与原始数据一致 ***")
    else:
        print("\n*** 发现数据不一致！***")

    # 检查 Expert 17-24 的原始数据的内存偏移
    print("\n" + "=" * 70)
    print("Expert 17-24 的内存偏移分析")
    print("=" * 70)

    for expert_id in range(17, 25):
        offset = expert_id * full_intermediate_size * lora_rank
        end_offset = (expert_id + 1) * full_intermediate_size * lora_rank
        print(f"  Expert {expert_id}: offset = {offset} to {end_offset} (size = {end_offset - offset})")

    # 检查是否有任何边界问题
    total_size = expert_num * full_intermediate_size * lora_rank
    print(f"\n  总数据大小: {total_size}")
    print(f"  Expert 24 结束位置: {25 * full_intermediate_size * lora_rank}")
    print(f"  是否越界: {25 * full_intermediate_size * lora_rank > total_size}")

    return all_correct


def check_expert_17_24_data():
    """检查 Expert 17-24 的数据特征"""
    print("\n" + "=" * 70)
    print("Expert 17-24 数据特征分析")
    print("=" * 70)

    data = torch.load(DATA_PATH)
    gate_lora_b = data["gate_lora_b"]

    print("\n原始 gate_lora_b (numpy) 检查:")
    gate_lora_b_np = gate_lora_b.view(-1).float().numpy()

    # 检查整体数据
    print(f"  总元素数: {len(gate_lora_b_np)}")
    print(f"  非零元素数: {np.count_nonzero(gate_lora_b_np)}")
    print(f"  所有值为零: {np.all(gate_lora_b_np == 0)}")

    # 检查特定 expert 的数据
    lora_rank = data["gate_lora_a"].shape[1]
    intermediate_size = data["intermediate_size"]
    slice_size = intermediate_size * lora_rank

    print("\nExpert 16-25 的数据统计:")
    for expert_id in range(16, 26):
        offset = expert_id * slice_size
        expert_data = gate_lora_b_np[offset : offset + slice_size]
        print(
            f"  Expert {expert_id}: min={expert_data.min():.6f}, max={expert_data.max():.6f}, "
            f"mean={expert_data.mean():.6f}, non-zero={np.count_nonzero(expert_data)}"
        )


if __name__ == "__main__":
    simulate_partition_copy()
    check_expert_17_24_data()
