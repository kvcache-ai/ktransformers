#!/usr/bin/env python
# coding=utf-8
"""
Test NaN Issue with Real Data from LlamaFactory Training

This file loads real training data saved from LlamaFactory's KT MoE integration
and attempts to reproduce the NaN issue in the AMX forward pass.

The data is saved by kt_moe.py when NaN is detected during training.
Data path: /tmp/kt_nan_debug_data.pt

Usage:
    1. First run LlamaFactory training to trigger NaN and save debug data
    2. Then run this test: python test_nan_with_real_data.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
print("sys.path:", sys.path)

import torch
import torch.nn.functional as F

# Try to import kt_kernel_ext
try:
    from kt_kernel import kt_kernel_ext

    HAS_KT_KERNEL = True
except ImportError:
    HAS_KT_KERNEL = False
    kt_kernel_ext = None

# Configuration
DEBUG_DATA_PATH = "/tmp/kt_nan_debug_data.pt"
NUM_THREADS = 60


def load_real_data(data_path: str) -> dict:
    """Load real training data saved from LlamaFactory."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Debug data file not found: {data_path}\n"
            "Please run LlamaFactory training first to generate the debug data.\n"
            "The data is automatically saved when NaN is detected in MOE output."
        )

    data = torch.load(data_path)
    print(f"[INFO] Loaded debug data from {data_path}")
    print(f"[INFO] Data keys: {list(data.keys())}")

    # Print data shapes
    print(f"\n[INFO] Data shapes:")
    print(f"  layer_idx: {data['layer_idx']}")
    print(f"  input_data: {data['input_data'].shape}")
    print(f"  expert_ids: {data['expert_ids'].shape}")
    print(f"  weights: {data['weights'].shape}")
    print(f"  output: {data['output'].shape}")
    print(f"  hidden_size: {data['hidden_size']}")
    print(f"  num_experts_per_tok: {data['num_experts_per_tok']}")
    print(f"  expert_num: {data['expert_num']}")
    print(f"  intermediate_size: {data['intermediate_size']}")

    # Print LoRA shapes
    print(f"\n[INFO] LoRA parameter shapes:")
    print(f"  gate_lora_a: {data['gate_lora_a'].shape}")
    print(f"  gate_lora_b: {data['gate_lora_b'].shape}")
    print(f"  up_lora_a: {data['up_lora_a'].shape}")
    print(f"  up_lora_b: {data['up_lora_b'].shape}")
    print(f"  down_lora_a: {data['down_lora_a'].shape}")
    print(f"  down_lora_b: {data['down_lora_b'].shape}")

    # Check if base weights are present
    if "gate_proj" in data:
        print(f"\n[INFO] Base weights shapes:")
        print(f"  gate_proj: {data['gate_proj'].shape}")
        print(f"  up_proj: {data['up_proj'].shape}")
        print(f"  down_proj: {data['down_proj'].shape}")
    else:
        print("\n[WARNING] Base weights not present in debug data!")
        print("          Need to load from model checkpoint manually.")

    return data


def analyze_nan_in_output(output: torch.Tensor, expert_ids: torch.Tensor):
    """Analyze NaN distribution in output."""
    nan_mask = torch.isnan(output)
    if not nan_mask.any():
        print("\n[INFO] No NaN in output - cannot reproduce the issue!")
        return False

    nan_count = nan_mask.sum().item()
    total_elements = output.numel()
    qlen = output.shape[0]
    hidden_size = output.shape[1]

    print(f"\n[NaN ANALYSIS]")
    print(f"  Total NaN count: {nan_count} / {total_elements} ({100*nan_count/total_elements:.2f}%)")

    # Find affected tokens
    nan_per_token = nan_mask.sum(dim=1)
    affected_tokens = torch.nonzero(nan_per_token > 0).squeeze(-1)
    print(f"  Affected tokens: {len(affected_tokens)} / {qlen}")
    print(f"  Affected token indices: {affected_tokens.tolist()[:20]}{'...' if len(affected_tokens) > 20 else ''}")

    # Analyze which experts are common among affected tokens
    if len(affected_tokens) > 0:
        print(f"\n[Expert Analysis for affected tokens]")
        expert_frequency = {}
        for tok_idx in affected_tokens[:10]:  # Check first 10 affected tokens
            experts = expert_ids[tok_idx].tolist()
            print(f"  Token {tok_idx}: experts = {experts}")
            for e in experts:
                expert_frequency[e] = expert_frequency.get(e, 0) + 1

        # Sort by frequency
        sorted_experts = sorted(expert_frequency.items(), key=lambda x: -x[1])
        print(f"\n  Most common experts among affected tokens:")
        for expert_id, count in sorted_experts[:10]:
            print(f"    Expert {expert_id}: appears in {count} affected tokens")

    return True


def moe_sft_torch_forward(
    input_data: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    lora_scaling: float,
    check_nan_per_expert: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    PyTorch reference implementation of MoE SFT forward.

    With per-expert NaN checking to identify problematic experts.
    """
    qlen, hidden_size = input_data.shape
    num_experts_per_tok = expert_ids.shape[1]

    # Convert to float32 for reference computation
    x = input_data.float()
    output = torch.zeros_like(x)

    nan_experts = set()
    nan_info = {}

    for i in range(qlen):
        for k in range(num_experts_per_tok):
            expert_id = expert_ids[i, k].item()
            w = weights[i, k].item()

            # Get base weights for this expert
            gate_w = gate_proj[expert_id].float()  # [intermediate_size, hidden_size]
            up_w = up_proj[expert_id].float()
            down_w = down_proj[expert_id].float()  # [hidden_size, intermediate_size]

            # Get LoRA weights for this expert
            gate_la = gate_lora_a[expert_id].float()  # [rank, hidden_size]
            gate_lb = gate_lora_b[expert_id].float()  # [intermediate_size, rank]
            up_la = up_lora_a[expert_id].float()
            up_lb = up_lora_b[expert_id].float()
            down_la = down_lora_a[expert_id].float()  # [rank, intermediate_size]
            down_lb = down_lora_b[expert_id].float()  # [hidden_size, rank]

            # Token input
            token_x = x[i]  # [hidden_size]

            # Gate computation with LoRA: gate_output = (W + s*B@A) @ x
            gate_base = gate_w @ token_x
            gate_lora = lora_scaling * (gate_lb @ (gate_la @ token_x))
            gate_output = gate_base + gate_lora

            # Up computation with LoRA
            up_base = up_w @ token_x
            up_lora = lora_scaling * (up_lb @ (up_la @ token_x))
            up_output = up_base + up_lora

            # SiLU activation and element-wise multiply
            hidden = F.silu(gate_output) * up_output

            # Down computation with LoRA
            down_base = down_w @ hidden
            down_lora = lora_scaling * (down_lb @ (down_la @ hidden))
            expert_output = down_base + down_lora

            # Check for NaN in expert output
            if check_nan_per_expert and torch.isnan(expert_output).any():
                if expert_id not in nan_experts:
                    nan_experts.add(expert_id)
                    nan_info[expert_id] = {
                        "token_idx": i,
                        "gate_base_nan": torch.isnan(gate_base).any().item(),
                        "gate_base_range": (
                            (gate_base.min().item(), gate_base.max().item())
                            if not torch.isnan(gate_base).any()
                            else ("NaN", "NaN")
                        ),
                        "gate_lora_nan": torch.isnan(gate_lora).any().item(),
                        "up_base_nan": torch.isnan(up_base).any().item(),
                        "up_lora_nan": torch.isnan(up_lora).any().item(),
                        "hidden_nan": torch.isnan(hidden).any().item(),
                        "down_base_nan": torch.isnan(down_base).any().item(),
                        "down_lora_nan": torch.isnan(down_lora).any().item(),
                    }

            # Weighted accumulation
            output[i] += w * expert_output

    if nan_experts:
        print(f"\n[PyTorch Reference] Found NaN in {len(nan_experts)} experts: {sorted(nan_experts)[:20]}")
        for expert_id in sorted(nan_experts)[:5]:
            info = nan_info[expert_id]
            print(f"  Expert {expert_id} (first seen at token {info['token_idx']}):")
            print(f"    gate_base NaN: {info['gate_base_nan']}, gate_lora NaN: {info['gate_lora_nan']}")
            print(f"    up_base NaN: {info['up_base_nan']}, up_lora NaN: {info['up_lora_nan']}")
            print(f"    hidden NaN: {info['hidden_nan']}")
            print(f"    down_base NaN: {info['down_base_nan']}, down_lora NaN: {info['down_lora_nan']}")

    return output.to(input_data.dtype), {"nan_experts": nan_experts, "nan_info": nan_info}


def test_with_real_data():
    """Test AMX forward with real data from LlamaFactory training."""
    print("=" * 70)
    print("Testing AMX MoE Forward with Real Training Data")
    print("=" * 70)

    # Load real data
    try:
        data = load_real_data(DEBUG_DATA_PATH)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return False

    # Extract data
    input_data = data["input_data"].contiguous()
    expert_ids = data["expert_ids"].contiguous()
    weights = data["weights"].contiguous()
    output_original = data["output"]  # Original output with NaN

    hidden_size = data["hidden_size"]
    num_experts_per_tok = data["num_experts_per_tok"]
    expert_num = data["expert_num"]
    intermediate_size = data["intermediate_size"]

    # LoRA params
    gate_lora_a = data["gate_lora_a"].contiguous()
    gate_lora_b = data["gate_lora_b"].contiguous()
    up_lora_a = data["up_lora_a"].contiguous()
    up_lora_b = data["up_lora_b"].contiguous()
    down_lora_a = data["down_lora_a"].contiguous()
    down_lora_b = data["down_lora_b"].contiguous()

    # Check if base weights exist
    if "gate_proj" not in data:
        print("\n[ERROR] Base weights not present in debug data!")
        print("Cannot proceed with test.")
        return False

    gate_proj = data["gate_proj"].contiguous()
    up_proj = data["up_proj"].contiguous()
    down_proj = data["down_proj"].contiguous()

    qlen = input_data.shape[0]
    lora_rank = gate_lora_a.shape[1]
    lora_alpha = 32.0  # Default from LlamaFactory
    lora_scaling = lora_alpha / lora_rank

    print(f"\n[INFO] Test configuration:")
    print(f"  qlen: {qlen}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  expert_num: {expert_num}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  lora_alpha: {lora_alpha}")

    # Analyze NaN in original output
    print("\n" + "=" * 70)
    print("Original Output NaN Analysis (from LlamaFactory)")
    print("=" * 70)
    analyze_nan_in_output(output_original, expert_ids)

    # Check input data for NaN
    print(f"\n[Input Check]")
    print(f"  input_data NaN: {torch.isnan(input_data).any().item()}")
    print(f"  input_data range: [{input_data.min().item():.4f}, {input_data.max().item():.4f}]")
    print(f"  weights NaN: {torch.isnan(weights).any().item()}")
    print(f"  weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")

    # Check base weights for NaN/Inf
    print(f"\n[Base Weights Check]")
    for name, w in [("gate_proj", gate_proj), ("up_proj", up_proj), ("down_proj", down_proj)]:
        has_nan = torch.isnan(w).any().item()
        has_inf = torch.isinf(w).any().item()
        if has_nan or has_inf:
            print(f"  {name}: NaN={has_nan}, Inf={has_inf} <- PROBLEM!")
        else:
            print(f"  {name}: range=[{w.min().item():.4f}, {w.max().item():.4f}]")

    # Check LoRA weights for NaN/Inf
    print(f"\n[LoRA Weights Check]")
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
        if has_nan or has_inf:
            print(f"  {name}: NaN={has_nan}, Inf={has_inf} <- PROBLEM!")
        else:
            print(f"  {name}: range=[{w.min().item():.4f}, {w.max().item():.4f}]")

    # Run PyTorch reference forward
    print("\n" + "=" * 70)
    print("PyTorch Reference Forward")
    print("=" * 70)
    torch_output, torch_nan_info = moe_sft_torch_forward(
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
        check_nan_per_expert=True,
    )

    print(f"\n[PyTorch Output]")
    print(f"  NaN count: {torch.isnan(torch_output).sum().item()}")
    if torch.isnan(torch_output).any():
        analyze_nan_in_output(torch_output, expert_ids)

    # Run AMX forward
    if not HAS_KT_KERNEL:
        print("\n[WARNING] kt_kernel_ext not available, skipping AMX test")
        return False

    print("\n" + "=" * 70)
    print("AMX Forward")
    print("=" * 70)

    # Initialize CPUInfer with single NUMA node
    print("\n[INFO] Creating CPUInfer with single NUMA node...")
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [NUM_THREADS]
    CPUInfer = kt_kernel_ext.CPUInfer(pool_config)

    # Create MOE SFT config
    config = kt_kernel_ext.moe.MOESFTConfig()
    config.expert_num = expert_num
    config.num_experts_per_tok = num_experts_per_tok
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.max_cache_depth = 1
    config.max_len = max(qlen * 2, 4096)
    config.layer_idx = data["layer_idx"]

    # Set weight pointers
    config.gate_proj = gate_proj.data_ptr()
    config.up_proj = up_proj.data_ptr()
    config.down_proj = down_proj.data_ptr()

    config.gate_lora_a = gate_lora_a.data_ptr()
    config.gate_lora_b = gate_lora_b.data_ptr()
    config.up_lora_a = up_lora_a.data_ptr()
    config.up_lora_b = up_lora_b.data_ptr()
    config.down_lora_a = down_lora_a.data_ptr()
    config.down_lora_b = down_lora_b.data_ptr()
    config.pool = CPUInfer.backend_

    # Create MOE instance
    moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)
    print(f"[INFO] Created AMXBF16_SFT_MOE instance")

    # Load weights
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()

    # Warm up
    CPUInfer.submit(moe.warm_up_task())
    CPUInfer.sync()

    # Run forward
    bsz_tensor = torch.tensor([qlen], device="cpu")
    amx_output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            amx_output.data_ptr(),
            False,  # save_for_backward
        )
    )
    CPUInfer.sync()

    print(f"\n[AMX Output]")
    print(f"  NaN count: {torch.isnan(amx_output).sum().item()}")
    if torch.isnan(amx_output).any():
        print("\n*** AMX also produces NaN - issue reproduced! ***")
        analyze_nan_in_output(amx_output, expert_ids)
    else:
        print("\n*** AMX output is clean - NaN issue NOT reproduced ***")
        print("This suggests the NaN may come from:")
        print("  1. Different LoRA pointer state during training")
        print("  2. Some other factor in the training pipeline")

    # Compare outputs
    print("\n" + "=" * 70)
    print("Output Comparison")
    print("=" * 70)

    # Compare with original (contains NaN)
    valid_mask_orig = ~torch.isnan(output_original)
    if valid_mask_orig.any():
        diff_orig = (amx_output[valid_mask_orig].float() - output_original[valid_mask_orig].float()).abs()
        print(f"\n[AMX vs Original (valid values only)]")
        print(f"  Max diff: {diff_orig.max().item():.6f}")
        print(f"  Mean diff: {diff_orig.mean().item():.6f}")

    # Compare with PyTorch reference
    valid_mask_both = ~(torch.isnan(amx_output) | torch.isnan(torch_output))
    if valid_mask_both.any():
        diff_torch = (amx_output[valid_mask_both].float() - torch_output[valid_mask_both].float()).abs()
        print(f"\n[AMX vs PyTorch Reference (valid values only)]")
        print(f"  Max diff: {diff_torch.max().item():.6f}")
        print(f"  Mean diff: {diff_torch.mean().item():.6f}")

    return True


def check_specific_expert(expert_id: int):
    """
    Detailed analysis of a specific expert's weights.

    Useful when we identify a problematic expert from the NaN analysis.
    """
    print(f"\n{'='*70}")
    print(f"Detailed Analysis of Expert {expert_id}")
    print(f"{'='*70}")

    try:
        data = load_real_data(DEBUG_DATA_PATH)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    if "gate_proj" not in data:
        print("[ERROR] Base weights not in debug data")
        return

    # Get weights for this expert
    gate_w = data["gate_proj"][expert_id]
    up_w = data["up_proj"][expert_id]
    down_w = data["down_proj"][expert_id]

    gate_la = data["gate_lora_a"][expert_id]
    gate_lb = data["gate_lora_b"][expert_id]
    up_la = data["up_lora_a"][expert_id]
    up_lb = data["up_lora_b"][expert_id]
    down_la = data["down_lora_a"][expert_id]
    down_lb = data["down_lora_b"][expert_id]

    print(f"\n[Base Weights]")
    for name, w in [("gate_proj", gate_w), ("up_proj", up_w), ("down_proj", down_w)]:
        has_nan = torch.isnan(w).any().item()
        has_inf = torch.isinf(w).any().item()
        w_abs = w.abs()
        print(f"  {name}: shape={w.shape}")
        print(f"    NaN: {has_nan}, Inf: {has_inf}")
        print(f"    range: [{w.min().item():.4f}, {w.max().item():.4f}]")
        print(f"    abs max: {w_abs.max().item():.4f}, abs mean: {w_abs.mean().item():.4f}")

    print(f"\n[LoRA Weights]")
    for name, w in [
        ("gate_lora_a", gate_la),
        ("gate_lora_b", gate_lb),
        ("up_lora_a", up_la),
        ("up_lora_b", up_lb),
        ("down_lora_a", down_la),
        ("down_lora_b", down_lb),
    ]:
        has_nan = torch.isnan(w).any().item()
        has_inf = torch.isinf(w).any().item()
        w_abs = w.abs()
        print(f"  {name}: shape={w.shape}")
        print(f"    NaN: {has_nan}, Inf: {has_inf}")
        print(f"    range: [{w.min().item():.4f}, {w.max().item():.4f}]")
        print(f"    abs max: {w_abs.max().item():.4f}, abs mean: {w_abs.mean().item():.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test NaN issue with real data")
    parser.add_argument("--expert", type=int, help="Analyze a specific expert")
    parser.add_argument("--data-path", type=str, default=DEBUG_DATA_PATH, help="Path to debug data file")
    args = parser.parse_args()

    if args.data_path != DEBUG_DATA_PATH:
        DEBUG_DATA_PATH = args.data_path

    if args.expert is not None:
        check_specific_expert(args.expert)
    else:
        test_with_real_data()
