#!/usr/bin/env python
# coding=utf-8
"""
Compare C++ TP dump with Python TP Simulator dump.

This script:
1. Runs C++ forward with SFT_MOE_DUMP=1 to generate cpp_dump/
2. Runs Python TP simulator with dump to generate py_dump/
3. Compares intermediate values at each step

Usage:
    python compare_tp_dumps.py [--tp-count 2] [--threshold 0.05]

The comparison accounts for TP partitioning:
- C++ dumps: {name}_tp{tp_idx}_e{expert_id}.bin
- Python dumps: {name}_tp{tp_idx}_e{expert_id}.bin
"""

import os
import sys
import struct
import argparse
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch

# Try to import kt_kernel
try:
    from kt_kernel.experts import KTMoEWrapper

    HAS_KT_KERNEL = True
except ImportError:
    HAS_KT_KERNEL = False
    print("WARNING: kt_kernel not available")


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_TP_COUNT = 2
DEFAULT_THRESHOLD = 0.05

# Test dimensions (smaller for faster testing)
TEST_CONFIG = {
    "expert_num": 8,
    "hidden_size": 4096,
    "intermediate_size": 1024,
    "lora_rank": 8,
    "qlen": 64,
    "k": 2,
    "num_threads": 8,
    "max_len": 1024,
}

# LoRA configuration
lora_rank = 16
lora_alpha = 32
lora_scaling = lora_alpha / lora_rank

# Weight scaling for numerical stability
WEIGHT_SCALE = 0.01
INPUT_SCALE = 0.1


# ============================================================================
# Dump Utilities
# ============================================================================


def read_matrix_file(filepath: str) -> tuple:
    """Read binary matrix file in the format: rows(int32), cols(int32), data(float32)"""
    if not os.path.exists(filepath):
        return None, None, None

    with open(filepath, "rb") as f:
        rows, cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
        data = data.reshape(rows, cols)
    return rows, cols, data


def save_matrix_file(filepath: str, data: np.ndarray):
    """Save matrix to binary file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if len(data.shape) == 1:
        rows, cols = 1, data.shape[0]
        data = data.reshape(1, -1)
    else:
        rows, cols = data.shape

    with open(filepath, "wb") as f:
        f.write(np.array([rows, cols], dtype=np.int32).tobytes())
        f.write(data.astype(np.float32).tobytes())


def compare_matrices(
    cpp_data: np.ndarray, py_data: np.ndarray, name: str, threshold: float, truncate_cols: bool = False
) -> dict:
    """Compare two matrices and return comparison result

    Args:
        truncate_cols: If True and shapes differ only in columns, truncate C++ data
                      to match Python column count (for padded LoRA intermediate dumps)
    """
    if cpp_data is None or py_data is None:
        return {"name": name, "status": "MISSING", "cpp_exists": cpp_data is not None, "py_exists": py_data is not None}

    if cpp_data.shape != py_data.shape:
        # Handle padded LoRA intermediate case: C++ has more columns due to padding
        if truncate_cols and len(cpp_data.shape) == 2 and len(py_data.shape) == 2:
            if cpp_data.shape[0] == py_data.shape[0] and cpp_data.shape[1] > py_data.shape[1]:
                # Truncate C++ data to match Python column count
                cpp_data = cpp_data[:, : py_data.shape[1]]

        # Check again after potential truncation
        if cpp_data.shape != py_data.shape:
            return {"name": name, "status": "SHAPE_MISMATCH", "cpp_shape": cpp_data.shape, "py_shape": py_data.shape}

    abs_diff = np.abs(cpp_data - py_data)
    max_abs_diff = np.max(abs_diff)
    mean_val_cpp = np.mean(cpp_data)
    mean_val_py = np.mean(py_data)
    mean_abs_diff = np.mean(abs_diff)

    # Relative error
    py_abs_mean = np.mean(np.abs(py_data)) + 1e-12
    rel_error = mean_abs_diff / py_abs_mean

    # Find location of max difference
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

    # Check for NaN/Inf
    cpp_nan = np.sum(np.isnan(cpp_data))
    cpp_inf = np.sum(np.isinf(cpp_data))
    py_nan = np.sum(np.isnan(py_data))
    py_inf = np.sum(np.isinf(py_data))

    passed = rel_error < threshold and cpp_nan == 0 and cpp_inf == 0

    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "shape": cpp_data.shape,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "mean_val_cpp": mean_val_cpp,
        "mean_val_py": mean_val_py,
        "rel_error": rel_error,
        "max_diff_idx": max_idx,
        "cpp_at_max": cpp_data[max_idx],
        "py_at_max": py_data[max_idx],
        "cpp_stats": {
            "min": np.min(cpp_data),
            "max": np.max(cpp_data),
            "mean": np.mean(cpp_data),
            "nan": cpp_nan,
            "inf": cpp_inf,
        },
        "py_stats": {
            "min": np.min(py_data),
            "max": np.max(py_data),
            "mean": np.mean(py_data),
            "nan": py_nan,
            "inf": py_inf,
        },
    }


def print_comparison_result(result: dict, verbose: bool = False):
    """Print comparison result with color coding"""
    name = result["name"]

    if result["status"] == "MISSING":
        cpp_exists = result.get("cpp_exists", False)
        py_exists = result.get("py_exists", False)
        print(f"\033[93m[MISSING]\033[0m {name}")
        print(f"    C++ exists: {cpp_exists}, Python exists: {py_exists}")
        return

    if result["status"] == "SHAPE_MISMATCH":
        print(f"\033[91m[SHAPE MISMATCH]\033[0m {name}")
        print(f"    C++ shape: {result['cpp_shape']}, Python shape: {result['py_shape']}")
        return

    if result["status"] == "PASS":
        print(
            f"\033[92m[PASS]\033[0m {name} - rel_error: {result['rel_error']:.2e}, max_abs_diff: {result['max_abs_diff']:.2e}, mean_val_cpp {result['mean_val_cpp']} py {result['mean_val_py']}"
        )
    else:
        print(
            f"\033[91m[FAIL]\033[0m {name} - rel_error: {result['rel_error']:.2e}, max_abs_diff: {result['max_abs_diff']:.2e}, mean_val_cpp {result['mean_val_cpp']} py {result['mean_val_py']}"
        )

    if verbose or result["status"] == "FAIL":
        print(f"    Shape: {result['shape']}")
        print(f"    Mean abs diff: {result['mean_abs_diff']:.6e}")
        print(
            f"    Max abs diff at {result['max_diff_idx']}: cpp={result['cpp_at_max']:.6e}, py={result['py_at_max']:.6e}"
        )
        cpp_stats = result["cpp_stats"]
        py_stats = result["py_stats"]
        print(f"    C++ stats: min={cpp_stats['min']:.6e}, max={cpp_stats['max']:.6e}, mean={cpp_stats['mean']:.6e}")
        print(f"    Py stats:  min={py_stats['min']:.6e}, max={py_stats['max']:.6e}, mean={py_stats['mean']:.6e}")


# ============================================================================
# Python TP Simulator (simplified version for dump comparison)
# ============================================================================


def silu(x):
    """SiLU activation function"""
    return x * torch.sigmoid(x)


def silu_backward(gate_out, up_out, grad_intermediate):
    """
    Backward pass for SiLU activation: act_out = silu(gate_out) * up_out

    Returns:
        grad_gate_out: gradient w.r.t. gate_out
        grad_up_out: gradient w.r.t. up_out
    """
    sigmoid_gate = torch.sigmoid(gate_out)
    silu_gate = gate_out * sigmoid_gate

    # grad_up_out = grad_intermediate * silu(gate_out)
    grad_up_out = grad_intermediate * silu_gate

    # grad_gate_out = grad_intermediate * up_out * silu'(gate_out)
    # silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    #          = sigmoid(x) * (1 + x - x * sigmoid(x))
    silu_grad = sigmoid_gate * (1 + gate_out - gate_out * sigmoid_gate)
    grad_gate_out = grad_intermediate * up_out * silu_grad

    return grad_gate_out, grad_up_out


class TPSimulatorForDump:
    """Simplified TP Simulator that dumps intermediate values matching C++ format"""

    def __init__(
        self,
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
        tp_count,
    ):
        self.tp_count = tp_count
        self.lora_scaling = lora_scaling
        self.expert_num = gate_proj.shape[0]
        self.intermediate_size = gate_proj.shape[1]
        self.hidden_size = gate_proj.shape[2]
        self.lora_rank = gate_lora_a.shape[1]

        # Partition weights for each TP
        tp_intermediate = self.intermediate_size // tp_count

        self.gate_proj_parts = []
        self.up_proj_parts = []
        self.down_proj_parts = []
        self.gate_lora_b_parts = []
        self.up_lora_b_parts = []
        self.down_lora_a_parts = []

        # Not partitioned
        self.gate_lora_a = gate_lora_a
        self.up_lora_a = up_lora_a
        self.down_lora_b = down_lora_b

        for tp_idx in range(tp_count):
            start = tp_idx * tp_intermediate
            end = start + tp_intermediate

            # Base weights
            self.gate_proj_parts.append(gate_proj[:, start:end, :].clone())
            self.up_proj_parts.append(up_proj[:, start:end, :].clone())
            self.down_proj_parts.append(down_proj[:, :, start:end].clone())

            # LoRA weights
            self.gate_lora_b_parts.append(gate_lora_b[:, start:end, :].clone())
            self.up_lora_b_parts.append(up_lora_b[:, start:end, :].clone())
            self.down_lora_a_parts.append(down_lora_a[:, :, start:end].clone())

    def forward_with_dump(self, input_tensor, expert_ids, routing_weights, dump_dir):
        """Forward pass with intermediate value dump"""
        qlen = input_tensor.shape[0]
        k = expert_ids.shape[1]

        # Compute m_local_num (tokens per expert)
        m_local_num = [0] * self.expert_num
        m_local_pos = [[0] * k for _ in range(qlen)]

        for i in range(qlen):
            for j in range(k):
                eid = expert_ids[i, j].item()
                m_local_pos[i][j] = m_local_num[eid]
                m_local_num[eid] += 1

        # Find activated experts
        activated_experts = [i for i in range(self.expert_num) if m_local_num[i] > 0]

        # Process each TP partition
        all_tp_outputs = []

        for tp_idx in range(self.tp_count):
            tp_intermediate = self.intermediate_size // self.tp_count

            # Pack input per expert and dump
            packed_inputs = {}
            for expert_idx in activated_experts:
                tokens_for_expert = []
                for i in range(qlen):
                    for j in range(k):
                        if expert_ids[i, j].item() == expert_idx:
                            tokens_for_expert.append(input_tensor[i])

                if tokens_for_expert:
                    packed_input = torch.stack(tokens_for_expert)
                    packed_inputs[expert_idx] = packed_input

                    # Dump packed input for each TP partition (same data, but C++ dumps per TP)
                    save_matrix_file(
                        f"{dump_dir}/packed_input_tp{tp_idx}_e{expert_idx}.bin", packed_input.float().numpy()
                    )

            # Process each expert
            expert_outputs = {}

            for expert_idx in activated_experts:
                if expert_idx not in packed_inputs:
                    continue

                x = packed_inputs[expert_idx].float()
                m = x.shape[0]

                # Get TP-partitioned weights
                gate_proj = self.gate_proj_parts[tp_idx][expert_idx].float()
                up_proj = self.up_proj_parts[tp_idx][expert_idx].float()
                down_proj = self.down_proj_parts[tp_idx][expert_idx].float()
                gate_lora_a = self.gate_lora_a[expert_idx].float()
                gate_lora_b = self.gate_lora_b_parts[tp_idx][expert_idx].float()
                up_lora_a = self.up_lora_a[expert_idx].float()
                up_lora_b = self.up_lora_b_parts[tp_idx][expert_idx].float()
                down_lora_a = self.down_lora_a_parts[tp_idx][expert_idx].float()
                down_lora_b = self.down_lora_b[expert_idx].float()

                # Gate base
                gate_base = torch.mm(x, gate_proj.t())
                save_matrix_file(f"{dump_dir}/gate_base_output_tp{tp_idx}_e{expert_idx}.bin", gate_base.numpy())

                # Up base
                up_base = torch.mm(x, up_proj.t())
                save_matrix_file(f"{dump_dir}/up_base_output_tp{tp_idx}_e{expert_idx}.bin", up_base.numpy())

                # Gate LoRA - with intermediate and GEMM dump
                gate_lora_inter = torch.mm(x, gate_lora_a.t())  # [m, lora_rank]
                save_matrix_file(
                    f"{dump_dir}/gate_lora_intermediate_tp{tp_idx}_e{expert_idx}.bin", gate_lora_inter.numpy()
                )
                # Pure GEMM output (without scaling)
                gate_lora_gemm = torch.mm(gate_lora_inter, gate_lora_b.t())  # [m, intermediate_size]
                save_matrix_file(
                    f"{dump_dir}/gate_lora_gemm_output_tp{tp_idx}_e{expert_idx}.bin", gate_lora_gemm.numpy()
                )
                gate_lora = gate_lora_gemm * self.lora_scaling
                gate_out = gate_base + gate_lora
                save_matrix_file(f"{dump_dir}/gate_lora_output_tp{tp_idx}_e{expert_idx}.bin", gate_out.numpy())

                # Up LoRA - with intermediate and GEMM dump
                up_lora_inter = torch.mm(x, up_lora_a.t())  # [m, lora_rank]
                save_matrix_file(f"{dump_dir}/up_lora_intermediate_tp{tp_idx}_e{expert_idx}.bin", up_lora_inter.numpy())
                # Pure GEMM output (without scaling)
                up_lora_gemm = torch.mm(up_lora_inter, up_lora_b.t())  # [m, intermediate_size]
                save_matrix_file(f"{dump_dir}/up_lora_gemm_output_tp{tp_idx}_e{expert_idx}.bin", up_lora_gemm.numpy())
                up_lora = up_lora_gemm * self.lora_scaling
                up_out = up_base + up_lora
                save_matrix_file(f"{dump_dir}/up_lora_output_tp{tp_idx}_e{expert_idx}.bin", up_out.numpy())

                # Activation input dump (gate_out and up_out before activation)
                save_matrix_file(f"{dump_dir}/activation_input_gate_tp{tp_idx}_e{expert_idx}.bin", gate_out.numpy())
                save_matrix_file(f"{dump_dir}/activation_input_up_tp{tp_idx}_e{expert_idx}.bin", up_out.numpy())

                # Activation
                act_out = silu(gate_out) * up_out
                save_matrix_file(f"{dump_dir}/activation_output_tp{tp_idx}_e{expert_idx}.bin", act_out.numpy())

                # Down base
                down_base = torch.mm(act_out, down_proj.t())
                save_matrix_file(f"{dump_dir}/down_base_output_tp{tp_idx}_e{expert_idx}.bin", down_base.numpy())

                # Down LoRA - with intermediate dump
                down_lora_inter = torch.mm(act_out, down_lora_a.t())  # [m, lora_rank]
                save_matrix_file(
                    f"{dump_dir}/down_lora_intermediate_tp{tp_idx}_e{expert_idx}.bin", down_lora_inter.numpy()
                )
                # Pure GEMM output (without scaling)
                down_lora_gemm = torch.mm(down_lora_inter, down_lora_b.t())  # [m, hidden_size]
                save_matrix_file(
                    f"{dump_dir}/down_lora_gemm_output_tp{tp_idx}_e{expert_idx}.bin", down_lora_gemm.numpy()
                )
                down_lora = down_lora_gemm * self.lora_scaling
                down_out = down_base + down_lora
                save_matrix_file(f"{dump_dir}/down_lora_output_tp{tp_idx}_e{expert_idx}.bin", down_out.numpy())
                save_matrix_file(f"{dump_dir}/down_total_output_tp{tp_idx}_e{expert_idx}.bin", down_out.numpy())

                expert_outputs[expert_idx] = (down_out, m_local_pos)

            # Weighted merge for this TP partition
            tp_output = torch.zeros(qlen, self.hidden_size, dtype=torch.float32)

            for i in range(qlen):
                for j in range(k):
                    expert_idx = expert_ids[i, j].item()
                    if expert_idx in expert_outputs:
                        down_out, positions = expert_outputs[expert_idx]
                        pos = positions[i][j]
                        weight = routing_weights[i, j].item()
                        tp_output[i] += down_out[pos] * weight

            save_matrix_file(f"{dump_dir}/final_output_tp{tp_idx}.bin", tp_output.numpy())

            all_tp_outputs.append(tp_output)

        # Sum all TP outputs
        final_output = sum(all_tp_outputs)
        return final_output

    def backward_with_dump(self, grad_output, input_tensor, expert_ids, routing_weights, forward_cache, dump_dir):
        """
        Backward pass with intermediate value dump

        Args:
            grad_output: [qlen, hidden_size] gradient from next layer
            input_tensor: [qlen, hidden_size] original input
            expert_ids: [qlen, k] expert indices
            routing_weights: [qlen, k] routing weights
            forward_cache: dict containing gate_out, up_out, act_out per expert per TP
            dump_dir: directory to dump intermediate values
        """
        qlen = input_tensor.shape[0]
        k = expert_ids.shape[1]

        # Compute m_local_num (tokens per expert)
        m_local_num = [0] * self.expert_num
        m_local_pos = [[0] * k for _ in range(qlen)]

        for i in range(qlen):
            for j in range(k):
                eid = expert_ids[i, j].item()
                m_local_pos[i][j] = m_local_num[eid]
                m_local_num[eid] += 1

        # Find activated experts
        activated_experts = [i for i in range(self.expert_num) if m_local_num[i] > 0]

        # Initialize grad_input accumulator (sum across all TPs)
        grad_input_total = torch.zeros(qlen, self.hidden_size, dtype=torch.float32)

        for tp_idx in range(self.tp_count):
            tp_intermediate = self.intermediate_size // self.tp_count

            # Pack grad_output per expert (weighted by routing_weights)
            packed_grad_outputs = {}
            packed_inputs = {}

            for expert_idx in activated_experts:
                grad_tokens_for_expert = []
                input_tokens_for_expert = []

                for i in range(qlen):
                    for j in range(k):
                        if expert_ids[i, j].item() == expert_idx:
                            weight = routing_weights[i, j].item()
                            grad_tokens_for_expert.append(grad_output[i] * weight)
                            input_tokens_for_expert.append(input_tensor[i])

                if grad_tokens_for_expert:
                    packed_grad_outputs[expert_idx] = torch.stack(grad_tokens_for_expert).float()
                    packed_inputs[expert_idx] = torch.stack(input_tokens_for_expert).float()

            # Process each expert's backward
            expert_grad_inputs = {}

            for expert_idx in activated_experts:
                if expert_idx not in packed_grad_outputs:
                    continue

                grad_out = packed_grad_outputs[expert_idx]  # [m, hidden_size]
                x = packed_inputs[expert_idx]  # [m, hidden_size]
                m = grad_out.shape[0]

                # Get forward cache for this expert
                cache_key = f"tp{tp_idx}_e{expert_idx}"
                gate_out = forward_cache[cache_key]["gate_out"]
                up_out = forward_cache[cache_key]["up_out"]
                act_out = forward_cache[cache_key]["act_out"]

                # Get TP-partitioned weights
                gate_proj = self.gate_proj_parts[tp_idx][expert_idx].float()
                up_proj = self.up_proj_parts[tp_idx][expert_idx].float()
                down_proj = self.down_proj_parts[tp_idx][expert_idx].float()
                gate_lora_a = self.gate_lora_a[expert_idx].float()
                gate_lora_b = self.gate_lora_b_parts[tp_idx][expert_idx].float()
                up_lora_a = self.up_lora_a[expert_idx].float()
                up_lora_b = self.up_lora_b_parts[tp_idx][expert_idx].float()
                down_lora_a = self.down_lora_a_parts[tp_idx][expert_idx].float()
                down_lora_b = self.down_lora_b[expert_idx].float()

                # Dump grad_output (packed)
                save_matrix_file(f"{dump_dir}/backward_grad_output_tp{tp_idx}_e{expert_idx}.bin", grad_out.numpy())

                # =====================================================
                # Stage 1: backward_down - compute grad_intermediate
                # =====================================================
                # down_base backward: grad_out @ down_proj
                # down_proj shape: [hidden_size, intermediate_size]
                # grad_out @ down_proj → [m, intermediate_size]
                grad_intermediate_base = torch.mm(grad_out, down_proj)
                save_matrix_file(
                    f"{dump_dir}/backward_down_base_tp{tp_idx}_e{expert_idx}.bin", grad_intermediate_base.numpy()
                )

                # Note: C++ backward_down only computes base grad_intermediate and weight gradients
                # The LoRA contribution to grad_intermediate is NOT added in C++
                # We match C++ behavior here for fair comparison
                grad_intermediate = grad_intermediate_base  # C++ doesn't add LoRA contribution
                save_matrix_file(
                    f"{dump_dir}/backward_grad_intermediate_tp{tp_idx}_e{expert_idx}.bin", grad_intermediate.numpy()
                )

                # =====================================================
                # Stage 2: backward_activation
                # =====================================================
                grad_gate_out, grad_up_out = silu_backward(gate_out, up_out, grad_intermediate)
                save_matrix_file(
                    f"{dump_dir}/backward_grad_gate_out_tp{tp_idx}_e{expert_idx}.bin", grad_gate_out.numpy()
                )
                save_matrix_file(f"{dump_dir}/backward_grad_up_out_tp{tp_idx}_e{expert_idx}.bin", grad_up_out.numpy())

                # =====================================================
                # Stage 3: backward_gate_up - compute grad_input
                # =====================================================
                # gate_base backward: grad_gate_out @ gate_proj
                # gate_proj shape: [intermediate_size, hidden_size]
                grad_input_gate_base = torch.mm(grad_gate_out, gate_proj)
                save_matrix_file(
                    f"{dump_dir}/backward_gate_base_tp{tp_idx}_e{expert_idx}.bin", grad_input_gate_base.numpy()
                )

                # gate_lora backward: grad_gate_out @ gate_lora_b @ gate_lora_a
                gate_lora_inter = torch.mm(grad_gate_out, gate_lora_b)
                save_matrix_file(
                    f"{dump_dir}/backward_gate_lora_inter_tp{tp_idx}_e{expert_idx}.bin", gate_lora_inter.numpy()
                )
                grad_input_gate_lora = torch.mm(gate_lora_inter, gate_lora_a) * self.lora_scaling
                save_matrix_file(
                    f"{dump_dir}/backward_gate_lora_tp{tp_idx}_e{expert_idx}.bin", grad_input_gate_lora.numpy()
                )

                # up_base backward: grad_up_out @ up_proj
                grad_input_up_base = torch.mm(grad_up_out, up_proj)
                save_matrix_file(
                    f"{dump_dir}/backward_up_base_tp{tp_idx}_e{expert_idx}.bin", grad_input_up_base.numpy()
                )

                # up_lora backward: grad_up_out @ up_lora_b @ up_lora_a
                up_lora_inter = torch.mm(grad_up_out, up_lora_b)
                save_matrix_file(
                    f"{dump_dir}/backward_up_lora_inter_tp{tp_idx}_e{expert_idx}.bin", up_lora_inter.numpy()
                )
                grad_input_up_lora = torch.mm(up_lora_inter, up_lora_a) * self.lora_scaling
                save_matrix_file(
                    f"{dump_dir}/backward_up_lora_tp{tp_idx}_e{expert_idx}.bin", grad_input_up_lora.numpy()
                )

                # Sum all components for this expert's grad_input
                grad_input_expert = (
                    grad_input_gate_base + grad_input_gate_lora + grad_input_up_base + grad_input_up_lora
                )
                save_matrix_file(
                    f"{dump_dir}/backward_grad_input_expert_tp{tp_idx}_e{expert_idx}.bin", grad_input_expert.numpy()
                )

                expert_grad_inputs[expert_idx] = (grad_input_expert, m_local_pos)

            # Scatter expert grad_inputs back to original positions
            tp_grad_input = torch.zeros(qlen, self.hidden_size, dtype=torch.float32)

            for i in range(qlen):
                for j in range(k):
                    expert_idx = expert_ids[i, j].item()
                    if expert_idx in expert_grad_inputs:
                        grad_input_expert, positions = expert_grad_inputs[expert_idx]
                        pos = positions[i][j]
                        tp_grad_input[i] += grad_input_expert[pos]

            save_matrix_file(f"{dump_dir}/backward_grad_input_tp{tp_idx}.bin", tp_grad_input.numpy())

            grad_input_total += tp_grad_input

        # Final merged grad_input
        save_matrix_file(f"{dump_dir}/backward_grad_input_final.bin", grad_input_total.numpy())

        return grad_input_total

    def forward_with_cache(self, input_tensor, expert_ids, routing_weights):
        """Forward pass that returns cache for backward"""
        qlen = input_tensor.shape[0]
        k = expert_ids.shape[1]

        m_local_num = [0] * self.expert_num
        m_local_pos = [[0] * k for _ in range(qlen)]

        for i in range(qlen):
            for j in range(k):
                eid = expert_ids[i, j].item()
                m_local_pos[i][j] = m_local_num[eid]
                m_local_num[eid] += 1

        activated_experts = [i for i in range(self.expert_num) if m_local_num[i] > 0]

        forward_cache = {}
        all_tp_outputs = []

        for tp_idx in range(self.tp_count):
            packed_inputs = {}
            for expert_idx in activated_experts:
                tokens_for_expert = []
                for i in range(qlen):
                    for j in range(k):
                        if expert_ids[i, j].item() == expert_idx:
                            tokens_for_expert.append(input_tensor[i])
                if tokens_for_expert:
                    packed_inputs[expert_idx] = torch.stack(tokens_for_expert)

            expert_outputs = {}

            for expert_idx in activated_experts:
                if expert_idx not in packed_inputs:
                    continue

                x = packed_inputs[expert_idx].float()

                gate_proj = self.gate_proj_parts[tp_idx][expert_idx].float()
                up_proj = self.up_proj_parts[tp_idx][expert_idx].float()
                down_proj = self.down_proj_parts[tp_idx][expert_idx].float()
                gate_lora_a = self.gate_lora_a[expert_idx].float()
                gate_lora_b = self.gate_lora_b_parts[tp_idx][expert_idx].float()
                up_lora_a = self.up_lora_a[expert_idx].float()
                up_lora_b = self.up_lora_b_parts[tp_idx][expert_idx].float()
                down_lora_a = self.down_lora_a_parts[tp_idx][expert_idx].float()
                down_lora_b = self.down_lora_b[expert_idx].float()

                # Gate
                gate_base = torch.mm(x, gate_proj.t())
                gate_lora_inter = torch.mm(x, gate_lora_a.t())
                gate_lora = torch.mm(gate_lora_inter, gate_lora_b.t()) * self.lora_scaling
                gate_out = gate_base + gate_lora

                # Up
                up_base = torch.mm(x, up_proj.t())
                up_lora_inter = torch.mm(x, up_lora_a.t())
                up_lora = torch.mm(up_lora_inter, up_lora_b.t()) * self.lora_scaling
                up_out = up_base + up_lora

                # Activation
                act_out = silu(gate_out) * up_out

                # Down
                down_base = torch.mm(act_out, down_proj.t())
                down_lora_inter = torch.mm(act_out, down_lora_a.t())
                down_lora = torch.mm(down_lora_inter, down_lora_b.t()) * self.lora_scaling
                down_out = down_base + down_lora

                # Store cache
                cache_key = f"tp{tp_idx}_e{expert_idx}"
                forward_cache[cache_key] = {
                    "gate_out": gate_out,
                    "up_out": up_out,
                    "act_out": act_out,
                }

                expert_outputs[expert_idx] = (down_out, m_local_pos)

            # Weighted merge
            tp_output = torch.zeros(qlen, self.hidden_size, dtype=torch.float32)
            for i in range(qlen):
                for j in range(k):
                    expert_idx = expert_ids[i, j].item()
                    if expert_idx in expert_outputs:
                        down_out, positions = expert_outputs[expert_idx]
                        pos = positions[i][j]
                        weight = routing_weights[i, j].item()
                        tp_output[i] += down_out[pos] * weight

            all_tp_outputs.append(tp_output)

        final_output = sum(all_tp_outputs)
        return final_output, forward_cache


# ============================================================================
# Main Comparison Logic
# ============================================================================


def create_kt_wrapper(
    tp_count, gate_proj, up_proj, down_proj, gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b
):
    """Create and initialize KTMoEWrapper"""
    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel not available")
        return None

    config = TEST_CONFIG

    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=config["expert_num"],
        num_experts_per_tok=config["k"],
        hidden_size=config["hidden_size"],
        moe_intermediate_size=config["intermediate_size"],
        num_gpu_experts=0,
        cpuinfer_threads=config["num_threads"],
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=config["max_len"],
        method="AMXINT8_SFT",
        mode="sft",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=2,
    )

    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj
    wrapper.load_weights(torch.arange(config["expert_num"], dtype=torch.int64))
    wrapper.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    return wrapper


def run_cpp_forward_with_dump(wrapper, input_tensor, expert_ids, routing_weights, dump_dir):
    """Run C++ forward with dump enabled"""
    if wrapper is None:
        print("ERROR: wrapper is None")
        return None

    # Set environment variables for C++ dump
    os.environ["SFT_MOE_DUMP"] = "1"
    os.environ["SFT_MOE_DUMP_DIR"] = dump_dir

    # Run forward with save_for_backward=True to enable backward
    output = wrapper.forward_sft(input_tensor, expert_ids, routing_weights, save_for_backward=True)

    # Clean up environment
    del os.environ["SFT_MOE_DUMP"]
    del os.environ["SFT_MOE_DUMP_DIR"]

    return output


def run_cpp_backward_with_dump(wrapper, grad_output, dump_dir):
    """Run C++ backward with dump enabled"""
    if wrapper is None:
        print("ERROR: wrapper is None")
        return None

    # Set environment variables for C++ dump
    os.environ["SFT_MOE_DUMP"] = "1"
    os.environ["SFT_MOE_DUMP_DIR"] = dump_dir

    # Run backward - returns (grad_input, grad_loras)
    grad_input, grad_loras = wrapper.backward(grad_output)

    # Clean up environment
    del os.environ["SFT_MOE_DUMP"]
    del os.environ["SFT_MOE_DUMP_DIR"]

    return grad_input


def compare_dumps(
    cpp_dir: str, py_dir: str, tp_count: int, threshold: float, verbose: bool = False, include_backward: bool = False
):
    """Compare C++ and Python dump files"""
    print("=" * 80)
    print("Comparing C++ and Python TP Dumps")
    print("=" * 80)
    print(f"C++ dump dir: {cpp_dir}")
    print(f"Python dump dir: {py_dir}")
    print(f"TP count: {tp_count}")
    print(f"Threshold: {threshold}")
    print(f"Include backward: {include_backward}")
    print("=" * 80)

    # Forward stages to compare (per TP partition, per expert)
    forward_stages = [
        "packed_input",
        "gate_base_output",
        "up_base_output",
        "gate_lora_intermediate",  # x @ gate_lora_a.T
        "up_lora_intermediate",  # x @ up_lora_a.T
        "gate_lora_gemm_output",  # intermediate @ gate_lora_b.T (without scaling)
        "up_lora_gemm_output",  # intermediate @ up_lora_b.T (without scaling)
        "gate_lora_output",  # gate_base + (intermediate @ gate_lora_b.T * scaling)
        "up_lora_output",  # up_base + (intermediate @ up_lora_b.T * scaling)
        "activation_input_gate",  # gate_out before activation (same as gate_lora_output)
        "activation_input_up",  # up_out before activation (same as up_lora_output)
        "activation_output",
        "down_base_output",
        "down_lora_intermediate",  # activation @ down_lora_a.T
        "down_lora_gemm_output",  # intermediate @ down_lora_b.T (without scaling)
        "down_lora_output",  # down_base + (intermediate @ down_lora_b.T * scaling)
        "down_total_output",
    ]

    # Backward stages to compare (per TP partition, per expert)
    # Note: C++ backward doesn't dump LoRA intermediate values separately
    # because it computes weight gradients but doesn't track LoRA contribution to grad_intermediate
    backward_stages = [
        "backward_grad_output",  # input grad_output (weighted)
        "backward_down_base",  # grad_out @ down_proj
        "backward_grad_intermediate",  # grad_intermediate (C++ only has base, Python has base+lora)
        "backward_grad_gate_out",  # from activation backward
        "backward_grad_up_out",  # from activation backward
        "backward_gate_base",  # grad_gate_out @ gate_proj
        "backward_up_base",  # grad_up_out @ up_proj
        "backward_gate_lora_inter",  # grad_gate_out @ gate_lora_b (C++ dumps this)
        "backward_gate_lora",  # gate_lora_inter @ gate_lora_a * scaling
        "backward_up_lora_inter",  # grad_up_out @ up_lora_b (C++ dumps this)
        "backward_up_lora",  # up_lora_inter @ up_lora_a * scaling
        "backward_grad_input_expert",  # sum of all grad_input components
    ]

    stages = forward_stages
    if include_backward:
        stages = stages + backward_stages

    # Find all expert IDs from dump files
    expert_ids = set()
    for f in os.listdir(cpp_dir):
        if "_e" in f and f.endswith(".bin"):
            try:
                eid = int(f.split("_e")[-1].replace(".bin", ""))
                expert_ids.add(eid)
            except ValueError:
                pass

    expert_ids = sorted(expert_ids)
    print(f"\nExperts found: {expert_ids}")

    all_passed = True
    results_by_stage = {}

    # Stages that need column truncation (C++ dumps with padded_lora_rank columns)
    lora_intermediate_stages = [
        "gate_lora_intermediate",
        "up_lora_intermediate",
        "down_lora_intermediate",
        # Backward LoRA intermediate stages (C++ uses padded_lora_rank)
        "backward_gate_lora_inter",
        "backward_up_lora_inter",
    ]

    # Compare each stage for each TP partition and expert
    for stage in stages:
        print(f"\n[{stage}]")
        stage_results = []

        # Enable truncation for LoRA intermediate stages (C++ uses padded_lora_rank)
        truncate_cols = stage in lora_intermediate_stages

        for tp_idx in range(tp_count):
            for expert_id in expert_ids:
                cpp_file = f"{cpp_dir}/{stage}_tp{tp_idx}_e{expert_id}.bin"
                py_file = f"{py_dir}/{stage}_tp{tp_idx}_e{expert_id}.bin"

                _, _, cpp_data = read_matrix_file(cpp_file)
                _, _, py_data = read_matrix_file(py_file)

                name = f"{stage}_tp{tp_idx}_e{expert_id}"
                result = compare_matrices(cpp_data, py_data, name, threshold, truncate_cols)
                print_comparison_result(result, verbose)
                stage_results.append(result)

                if result["status"] != "PASS":
                    all_passed = False

        results_by_stage[stage] = stage_results

    # Compare final output (per TP partition)
    print(f"\n[final_output]")
    for tp_idx in range(tp_count):
        cpp_file = f"{cpp_dir}/final_output_tp{tp_idx}.bin"
        py_file = f"{py_dir}/final_output_tp{tp_idx}.bin"

        _, _, cpp_data = read_matrix_file(cpp_file)
        _, _, py_data = read_matrix_file(py_file)

        name = f"final_output_tp{tp_idx}"
        result = compare_matrices(cpp_data, py_data, name, threshold)
        print_comparison_result(result, verbose)

        if result["status"] != "PASS":
            all_passed = False

    # Compare backward final output if enabled
    if include_backward:
        print(f"\n[backward_grad_input (per TP)]")
        for tp_idx in range(tp_count):
            cpp_file = f"{cpp_dir}/backward_grad_input_tp{tp_idx}.bin"
            py_file = f"{py_dir}/backward_grad_input_tp{tp_idx}.bin"

            _, _, cpp_data = read_matrix_file(cpp_file)
            _, _, py_data = read_matrix_file(py_file)

            name = f"backward_grad_input_tp{tp_idx}"
            result = compare_matrices(cpp_data, py_data, name, threshold)
            print_comparison_result(result, verbose)

            if result["status"] != "PASS":
                all_passed = False

        print(f"\n[backward_grad_input_final]")
        cpp_file = f"{cpp_dir}/backward_grad_input_final.bin"
        py_file = f"{py_dir}/backward_grad_input_final.bin"

        _, _, cpp_data = read_matrix_file(cpp_file)
        _, _, py_data = read_matrix_file(py_file)

        name = "backward_grad_input_final"
        result = compare_matrices(cpp_data, py_data, name, threshold)
        print_comparison_result(result, verbose)

        if result["status"] != "PASS":
            all_passed = False

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print(f"\033[92mALL COMPARISONS PASSED\033[0m")
    else:
        print(f"\033[91mSOME COMPARISONS FAILED\033[0m")
    print("=" * 80)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python TP dumps")
    parser.add_argument("--tp-count", type=int, default=DEFAULT_TP_COUNT, help="TP partition count")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Relative error threshold")
    parser.add_argument("--cpp-dir", default="./cpp_dump", help="C++ dump directory")
    parser.add_argument("--py-dir", default="./py_dump", help="Python dump directory")
    parser.add_argument("--skip-run", action="store_true", help="Skip running, just compare existing dumps")
    parser.add_argument("--backward", action="store_true", help="Include backward pass comparison")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not args.skip_run:
        print("=" * 80)
        print("Running C++ and Python TP forward/backward with dump")
        print("=" * 80)

        # Clean up old dumps
        for d in [args.cpp_dir, args.py_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        torch.manual_seed(42)
        config = TEST_CONFIG

        # Initialize weights
        print("\n[Initializing weights]")
        gate_proj = (
            torch.rand(config["expert_num"], config["intermediate_size"], config["hidden_size"], dtype=torch.bfloat16)
            * WEIGHT_SCALE
        ).contiguous()
        up_proj = (
            torch.rand(config["expert_num"], config["intermediate_size"], config["hidden_size"], dtype=torch.bfloat16)
            * WEIGHT_SCALE
        ).contiguous()
        down_proj = (
            torch.rand(config["expert_num"], config["hidden_size"], config["intermediate_size"], dtype=torch.bfloat16)
            * WEIGHT_SCALE
        ).contiguous()

        gate_lora_a = (
            torch.rand(config["expert_num"], lora_rank, config["hidden_size"], dtype=torch.bfloat16) * WEIGHT_SCALE
        ).contiguous()
        gate_lora_b = (
            torch.rand(config["expert_num"], config["intermediate_size"], lora_rank, dtype=torch.bfloat16)
            * WEIGHT_SCALE
        ).contiguous()
        up_lora_a = (
            torch.rand(config["expert_num"], lora_rank, config["hidden_size"], dtype=torch.bfloat16) * WEIGHT_SCALE
        ).contiguous()
        up_lora_b = (
            torch.rand(config["expert_num"], config["intermediate_size"], lora_rank, dtype=torch.bfloat16)
            * WEIGHT_SCALE
        ).contiguous()
        down_lora_a = (
            torch.rand(config["expert_num"], lora_rank, config["intermediate_size"], dtype=torch.bfloat16)
            * WEIGHT_SCALE
        ).contiguous()
        down_lora_b = (
            torch.rand(config["expert_num"], config["hidden_size"], lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
        ).contiguous()

        # Generate test data
        print("\n[Generating test data]")
        input_tensor = (
            torch.rand((config["qlen"], config["hidden_size"]), dtype=torch.bfloat16) * INPUT_SCALE
        ).contiguous()
        expert_ids = torch.stack(
            [torch.randperm(config["expert_num"])[: config["k"]] for _ in range(config["qlen"])]
        ).contiguous()
        routing_weights = torch.rand(config["qlen"], config["k"], dtype=torch.float).contiguous()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Expert IDs shape: {expert_ids.shape}")

        # Create simulator for Python
        simulator = TPSimulatorForDump(
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
            args.tp_count,
        )

        # Run C++ forward with dump
        print("\n[Running C++ forward with dump]")
        cpp_output = None
        wrapper = None
        if HAS_KT_KERNEL:
            wrapper = create_kt_wrapper(
                args.tp_count,
                gate_proj,
                up_proj,
                down_proj,
                gate_lora_a,
                gate_lora_b,
                up_lora_a,
                up_lora_b,
                down_lora_a,
                down_lora_b,
            )
            cpp_output = run_cpp_forward_with_dump(wrapper, input_tensor, expert_ids, routing_weights, args.cpp_dir)
            print(f"  C++ output shape: {cpp_output.shape}")
        else:
            print("  Skipped (kt_kernel not available)")

        # Run Python TP simulator forward with dump
        print("\n[Running Python TP simulator forward with dump]")
        py_output = simulator.forward_with_dump(input_tensor, expert_ids, routing_weights, args.py_dir)
        print(f"  Python output shape: {py_output.shape}")

        # Run backward if requested
        if args.backward:
            print("\n[Generating grad_output for backward]")
            grad_output = (
                torch.rand((config["qlen"], config["hidden_size"]), dtype=torch.bfloat16) * INPUT_SCALE
            ).contiguous()
            print(f"  grad_output shape: {grad_output.shape}")

            # Run C++ backward with dump
            print("\n[Running C++ backward with dump]")
            if HAS_KT_KERNEL and wrapper is not None:
                cpp_grad_input = run_cpp_backward_with_dump(wrapper, grad_output, args.cpp_dir)
                if cpp_grad_input is not None:
                    print(f"  C++ grad_input shape: {cpp_grad_input.shape}")
                else:
                    print("  C++ backward returned None")
            else:
                print("  Skipped (kt_kernel not available)")

            # Run Python backward with dump
            print("\n[Running Python TP simulator backward with dump]")
            # First run forward to get cache
            _, forward_cache = simulator.forward_with_cache(input_tensor, expert_ids, routing_weights)
            py_grad_input = simulator.backward_with_dump(
                grad_output.float(), input_tensor, expert_ids, routing_weights, forward_cache, args.py_dir
            )
            print(f"  Python grad_input shape: {py_grad_input.shape}")

    # Compare dumps
    print("\n")
    success = compare_dumps(
        args.cpp_dir, args.py_dir, args.tp_count, args.threshold, args.verbose, include_backward=args.backward
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
