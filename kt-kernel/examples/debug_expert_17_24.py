#!/usr/bin/env python
# coding=utf-8
"""
深入分析 Expert 17-24 产生 NaN 的原因

根据之前的调试日志，只有 Expert 17-24 这 8 个连续的 expert 产生 NaN。
本脚本尝试：
1. 分析哪些 token 激活了 Expert 17-24
2. 检查这些 token 的输入数据特征
3. 验证 Expert 17-24 的权重数据是否有异常
4. 手动执行 LoRA 计算，逐步定位 NaN 产生位置
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch
import torch.nn.functional as F
import numpy as np

# 数据路径
DATA_PATH = "/mnt/data/lpl/kt_nan_debug_data.pt"


def silu(x):
    """SiLU activation function."""
    return x * torch.sigmoid(x)


def load_and_analyze_data():
    """加载数据并进行详细分析"""
    print(f"\n{'='*70}")
    print("加载和分析 PT 文件数据")
    print(f"{'='*70}")

    data = torch.load(DATA_PATH)

    # 配置
    expert_num = data["expert_num"]
    hidden_size = data["hidden_size"]
    intermediate_size = data["intermediate_size"]
    num_experts_per_tok = data["num_experts_per_tok"]
    qlen = data["input_data"].shape[0]
    lora_rank = data["gate_lora_a"].shape[1]
    lora_alpha = 16.0
    lora_scaling = lora_alpha / lora_rank

    print(f"\n配置参数:")
    print(f"  expert_num: {expert_num}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  qlen: {qlen}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  lora_scaling: {lora_scaling}")

    # 提取数据
    input_data = data["input_data"].contiguous()
    expert_ids = data["expert_ids"].contiguous()
    weights = data["weights"].contiguous()
    gate_proj = data["gate_proj"].contiguous()
    up_proj = data["up_proj"].contiguous()
    down_proj = data["down_proj"].contiguous()
    gate_lora_a = data["gate_lora_a"].contiguous()
    gate_lora_b = data["gate_lora_b"].contiguous()
    up_lora_a = data["up_lora_a"].contiguous()
    up_lora_b = data["up_lora_b"].contiguous()
    down_lora_a = data["down_lora_a"].contiguous()
    down_lora_b = data["down_lora_b"].contiguous()

    return {
        "input_data": input_data,
        "expert_ids": expert_ids,
        "weights": weights,
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
        "gate_lora_a": gate_lora_a,
        "gate_lora_b": gate_lora_b,
        "up_lora_a": up_lora_a,
        "up_lora_b": up_lora_b,
        "down_lora_a": down_lora_a,
        "down_lora_b": down_lora_b,
        "config": {
            "expert_num": expert_num,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts_per_tok": num_experts_per_tok,
            "qlen": qlen,
            "lora_rank": lora_rank,
            "lora_scaling": lora_scaling,
        },
    }


def analyze_expert_routing(data):
    """分析 Expert 路由情况，特别关注 Expert 17-24"""
    print(f"\n{'='*70}")
    print("Expert 路由分析")
    print(f"{'='*70}")

    expert_ids = data["expert_ids"]
    config = data["config"]
    qlen = config["qlen"]
    num_experts_per_tok = config["num_experts_per_tok"]

    # 统计每个 expert 被激活的次数
    expert_counts = {}
    expert_token_map = {}  # expert_id -> list of (token_idx, position_in_k)

    for tok_idx in range(qlen):
        for k_idx in range(num_experts_per_tok):
            expert_id = expert_ids[tok_idx, k_idx].item()
            if expert_id not in expert_counts:
                expert_counts[expert_id] = 0
                expert_token_map[expert_id] = []
            expert_counts[expert_id] += 1
            expert_token_map[expert_id].append((tok_idx, k_idx))

    print(f"\n所有激活的 Expert (共 {len(expert_counts)} 个):")
    for expert_id in sorted(expert_counts.keys()):
        count = expert_counts[expert_id]
        marker = " *** SUSPECT ***" if 17 <= expert_id <= 24 else ""
        print(f"  Expert {expert_id}: {count} tokens{marker}")

    # 详细分析 Expert 17-24
    print(f"\n{'='*70}")
    print("Expert 17-24 详细分析")
    print(f"{'='*70}")

    problem_experts = list(range(17, 25))
    for expert_id in problem_experts:
        if expert_id in expert_token_map:
            tokens = expert_token_map[expert_id]
            print(f"\nExpert {expert_id}: 被 {len(tokens)} 个 token 激活")
            print(f"  激活的 token (token_idx, k_position):")
            for tok_idx, k_idx in tokens[:10]:  # 只显示前 10 个
                print(f"    Token {tok_idx}, k={k_idx}")
            if len(tokens) > 10:
                print(f"    ... 还有 {len(tokens) - 10} 个")
        else:
            print(f"\nExpert {expert_id}: 未被激活")

    return expert_token_map


def check_data_for_expert(data, expert_id):
    """检查特定 Expert 的输入数据和权重是否有异常"""
    print(f"\n{'='*70}")
    print(f"Expert {expert_id} 数据检查")
    print(f"{'='*70}")

    config = data["config"]

    # 检查基础权重
    gate_proj = data["gate_proj"][expert_id]
    up_proj = data["up_proj"][expert_id]
    down_proj = data["down_proj"][expert_id]

    print(f"\n基础权重检查:")
    for name, w in [("gate_proj", gate_proj), ("up_proj", up_proj), ("down_proj", down_proj)]:
        has_nan = torch.isnan(w).any().item()
        has_inf = torch.isinf(w).any().item()
        w_min = w.min().item()
        w_max = w.max().item()
        w_mean = w.float().mean().item()
        w_std = w.float().std().item()
        print(
            f"  {name}: NaN={has_nan}, Inf={has_inf}, range=[{w_min:.6f}, {w_max:.6f}], mean={w_mean:.6f}, std={w_std:.6f}"
        )

    # 检查 LoRA 权重
    gate_lora_a = data["gate_lora_a"][expert_id]
    gate_lora_b = data["gate_lora_b"][expert_id]
    up_lora_a = data["up_lora_a"][expert_id]
    up_lora_b = data["up_lora_b"][expert_id]
    down_lora_a = data["down_lora_a"][expert_id]
    down_lora_b = data["down_lora_b"][expert_id]

    print(f"\nLoRA 权重检查:")
    for name, w in [
        ("gate_lora_a", gate_lora_a),
        ("gate_lora_b", gate_lora_b),
        ("up_lora_a", up_lora_a),
        ("up_lora_b", up_lora_b),
        ("down_lora_a", down_lora_a),
        ("down_lora_b", down_lora_b),
    ]:
        has_nan = torch.isnan(w).any().item()
        has_inf = torch.isinf(w).any().item()
        w_min = w.min().item()
        w_max = w.max().item()
        w_mean = w.float().mean().item()
        w_std = w.float().std().item()
        print(
            f"  {name}: NaN={has_nan}, Inf={has_inf}, range=[{w_min:.6f}, {w_max:.6f}], mean={w_mean:.6f}, std={w_std:.6f}"
        )

    return not any(
        [
            torch.isnan(gate_proj).any(),
            torch.isnan(up_proj).any(),
            torch.isnan(down_proj).any(),
            torch.isnan(gate_lora_a).any(),
            torch.isnan(gate_lora_b).any(),
            torch.isnan(up_lora_a).any(),
            torch.isnan(up_lora_b).any(),
            torch.isnan(down_lora_a).any(),
            torch.isnan(down_lora_b).any(),
        ]
    )


def manual_forward_for_expert(data, expert_token_map, expert_id):
    """对单个 Expert 手动执行 forward 计算，逐步定位 NaN"""
    print(f"\n{'='*70}")
    print(f"Expert {expert_id} 手动 Forward 计算")
    print(f"{'='*70}")

    if expert_id not in expert_token_map:
        print(f"Expert {expert_id} 未被激活，跳过")
        return

    config = data["config"]
    tokens = expert_token_map[expert_id]
    num_tokens = len(tokens)

    print(f"Expert {expert_id} 处理 {num_tokens} 个 token")

    # 收集该 expert 的输入
    input_data = data["input_data"]
    local_input = torch.stack([input_data[tok_idx] for tok_idx, _ in tokens])

    # 获取权重
    gate_proj = data["gate_proj"][expert_id]  # [intermediate_size, hidden_size]
    up_proj = data["up_proj"][expert_id]
    gate_lora_a = data["gate_lora_a"][expert_id]  # [lora_rank, hidden_size]
    gate_lora_b = data["gate_lora_b"][expert_id]  # [intermediate_size, lora_rank]
    up_lora_a = data["up_lora_a"][expert_id]
    up_lora_b = data["up_lora_b"][expert_id]

    lora_scaling = config["lora_scaling"]

    # Step 1: Base Gate GEMM
    # [num_tokens, hidden_size] @ [hidden_size, intermediate_size] -> [num_tokens, intermediate_size]
    gate_base = local_input.float() @ gate_proj.float().T
    print(f"\nStep 1 - Gate Base GEMM:")
    print(f"  local_input: shape={local_input.shape}, NaN={torch.isnan(local_input).sum().item()}")
    print(f"  gate_proj: shape={gate_proj.shape}, NaN={torch.isnan(gate_proj).sum().item()}")
    print(f"  gate_base: shape={gate_base.shape}, NaN={torch.isnan(gate_base).sum().item()}")
    print(f"  gate_base range: [{gate_base.min().item():.4f}, {gate_base.max().item():.4f}]")

    # Step 2: Gate LoRA
    # intermediate = input @ lora_A^T: [num_tokens, hidden_size] @ [hidden_size, lora_rank] -> [num_tokens, lora_rank]
    gate_lora_inter = local_input.float() @ gate_lora_a.float().T
    print(f"\nStep 2a - Gate LoRA intermediate (input @ lora_A^T):")
    print(f"  gate_lora_a: shape={gate_lora_a.shape}, NaN={torch.isnan(gate_lora_a).sum().item()}")
    print(f"  gate_lora_inter: shape={gate_lora_inter.shape}, NaN={torch.isnan(gate_lora_inter).sum().item()}")
    print(f"  gate_lora_inter range: [{gate_lora_inter.min().item():.6f}, {gate_lora_inter.max().item():.6f}]")

    # lora_out = intermediate @ lora_B^T: [num_tokens, lora_rank] @ [lora_rank, intermediate_size] -> [num_tokens, intermediate_size]
    gate_lora_out = gate_lora_inter @ gate_lora_b.float().T
    print(f"\nStep 2b - Gate LoRA output (inter @ lora_B^T):")
    print(f"  gate_lora_b: shape={gate_lora_b.shape}, NaN={torch.isnan(gate_lora_b).sum().item()}")
    print(f"  gate_lora_out: shape={gate_lora_out.shape}, NaN={torch.isnan(gate_lora_out).sum().item()}")
    print(f"  gate_lora_out range: [{gate_lora_out.min().item():.6f}, {gate_lora_out.max().item():.6f}]")

    # Step 3: Add LoRA to base with scaling
    gate_output = gate_base + gate_lora_out * lora_scaling
    print(f"\nStep 3 - Gate output (base + lora * scaling):")
    print(f"  gate_output: shape={gate_output.shape}, NaN={torch.isnan(gate_output).sum().item()}")
    print(f"  gate_output range: [{gate_output.min().item():.4f}, {gate_output.max().item():.4f}]")

    # 同样计算 Up
    up_base = local_input.float() @ up_proj.float().T
    up_lora_inter = local_input.float() @ up_lora_a.float().T
    up_lora_out = up_lora_inter @ up_lora_b.float().T
    up_output = up_base + up_lora_out * lora_scaling

    print(f"\nUp projection 汇总:")
    print(f"  up_base NaN: {torch.isnan(up_base).sum().item()}")
    print(f"  up_lora_inter NaN: {torch.isnan(up_lora_inter).sum().item()}")
    print(f"  up_lora_out NaN: {torch.isnan(up_lora_out).sum().item()}")
    print(f"  up_output NaN: {torch.isnan(up_output).sum().item()}")

    # Step 4: Activation
    intermediate = silu(gate_output) * up_output
    print(f"\nStep 4 - Activation (silu(gate) * up):")
    print(f"  intermediate: shape={intermediate.shape}, NaN={torch.isnan(intermediate).sum().item()}")

    if torch.isnan(gate_output).sum().item() > 0:
        # 详细分析 NaN 位置
        nan_mask = torch.isnan(gate_output)
        nan_indices = torch.nonzero(nan_mask)
        print(f"\n*** 发现 NaN! ***")
        print(f"  NaN 数量: {nan_mask.sum().item()}")
        print(f"  前 10 个 NaN 位置:")
        for i in range(min(10, len(nan_indices))):
            idx = nan_indices[i]
            print(f"    位置 [{idx[0].item()}, {idx[1].item()}]")

    return gate_output, up_output


def compare_with_other_experts(data, expert_token_map):
    """对比 Expert 17-24 与其他 Expert 的计算结果"""
    print(f"\n{'='*70}")
    print("对比不同 Expert 的计算结果")
    print(f"{'='*70}")

    problem_experts = list(range(17, 25))
    other_experts = [e for e in expert_token_map.keys() if e not in problem_experts]

    print(f"\n问题 Expert (17-24): {problem_experts}")
    print(f"正常 Expert (采样): {other_experts[:5]}")

    # 对比
    print("\n对比各 Expert 的 Forward 计算:")
    for expert_id in problem_experts[:2] + other_experts[:2]:  # 各取 2 个
        if expert_id in expert_token_map:
            gate_out, up_out = manual_forward_for_expert(data, expert_token_map, expert_id)


def main():
    print("=" * 70)
    print("Expert 17-24 NaN 问题深度分析")
    print("=" * 70)

    # 1. 加载数据
    data = load_and_analyze_data()

    # 2. 分析 Expert 路由
    expert_token_map = analyze_expert_routing(data)

    # 3. 检查 Expert 17-24 的数据
    print("\n" + "=" * 70)
    print("检查 Expert 17-24 的原始数据")
    print("=" * 70)
    all_clean = True
    for expert_id in range(17, 25):
        is_clean = check_data_for_expert(data, expert_id)
        all_clean = all_clean and is_clean

    if all_clean:
        print("\n*** Expert 17-24 的原始数据没有 NaN/Inf，问题可能在计算过程中 ***")

    # 4. 手动计算并对比
    print("\n" + "=" * 70)
    print("手动执行 Forward 计算，逐步追踪 NaN")
    print("=" * 70)

    # 检查一个正常的 Expert
    normal_expert = None
    for e in expert_token_map.keys():
        if e not in range(17, 25):
            normal_expert = e
            break

    if normal_expert:
        print(f"\n--- 正常 Expert {normal_expert} ---")
        manual_forward_for_expert(data, expert_token_map, normal_expert)

    # 检查一个问题 Expert
    for e in [17, 20, 24]:
        if e in expert_token_map:
            print(f"\n--- 问题 Expert {e} ---")
            manual_forward_for_expert(data, expert_token_map, e)
            break


if __name__ == "__main__":
    main()
