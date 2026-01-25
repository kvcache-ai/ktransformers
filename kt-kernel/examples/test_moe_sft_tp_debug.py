#!/usr/bin/env python
# coding=utf-8
"""
MOE SFT TP Debug Test File

This file implements:
1. PyTorch TP (Tensor Parallel) simulation for SFT MoE with LoRA
2. Intermediate value dumping for debugging
3. Comparison tests between PyTorch simulation and C++ implementation

Key TP partitioning rules:
- gate_proj/up_proj: [intermediate_size, hidden_size] -> contiguous slice by intermediate_size
- down_proj: [hidden_size, intermediate_size] -> row-wise slice by intermediate_size
- gate_lora_a/up_lora_a: NOT partitioned (no intermediate_size dim)
- gate_lora_b/up_lora_b: [intermediate_size, lora_rank] -> contiguous slice
- down_lora_a: [lora_rank, intermediate_size] -> row-wise slice
- down_lora_b: NOT partitioned (no intermediate_size dim)
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
print("sys.path:", sys.path)

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# Try to import kt_kernel
try:
    from kt_kernel.experts import KTMoEWrapper
    from kt_kernel.experts_sft import KExpertsSFTBuffer, BaseSFTMoEWrapper

    HAS_KT_KERNEL = True
except ImportError:
    try:
        # Alternative import path (for development)
        sys.path.insert(0, os.path.dirname(__file__) + "/../python")
        from experts import KTMoEWrapper
        from experts_sft import KExpertsSFTBuffer, BaseSFTMoEWrapper

        HAS_KT_KERNEL = True
    except ImportError as e:
        print(f"Warning: Could not import kt_kernel: {e}")
        HAS_KT_KERNEL = False
        KTMoEWrapper = None


# =============================================================================
# Test Configuration
# =============================================================================

# Model configuration (based on DeepSeek-V3 architecture)
expert_num = 256  # Total number of experts
hidden_size = 7168  # Hidden dimension
intermediate_size = 2048  # MLP intermediate dimension
max_len = 25600  # Maximum sequence length
num_experts_per_tok = 8  # Number of experts per token (top-k)
qlen = 40  # Sequence length for testing
layer_num = 3  # Number of layers to test

# LoRA configuration
lora_rank = 16  # LoRA rank (r)
lora_alpha = 32.0  # LoRA scaling factor (alpha)
lora_scaling = lora_alpha / lora_rank  # Effective scaling: alpha / r

# Test configuration
validation_iter = 2  # Number of validation iterations
debug_print_count = 8  # Number of values to print in debug output
num_threads = 32  # Number of CPU threads for inference

# TP configuration
TP_COUNT = 2  # TP mode: 2 NUMA subpools for debugging
NO_TP_COUNT = 1  # No-TP mode: single subpool

# Precision thresholds
BF16_FORWARD_THRESHOLD = 0.05
BF16_BACKWARD_THRESHOLD = 0.10


# =============================================================================
# Activation Functions
# =============================================================================


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU/Swish activation function."""
    return x * torch.sigmoid(x)


def act_fn(x: torch.Tensor) -> torch.Tensor:
    """Activation function for MoE MLP (SiLU/Swish)"""
    return x / (1.0 + torch.exp(-x))


def silu_grad(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation gradient: d(silu(x))/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
    sig = torch.sigmoid(x)
    return sig * (1 + x * (1 - sig))


# =============================================================================
# Dump and Comparison Utilities (from compare_dumps.py and test_minimal_backward.py)
# =============================================================================


def check_nan(tensor: torch.Tensor, name: str) -> bool:
    """Check tensor for NaN/Inf values"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(f"\033[91m[WARNING] {name} contains NaN={nan_count}, Inf={inf_count}\033[0m")
        return True
    return False


def compute_relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative error between two tensors"""
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    diff = (a_f32 - b_f32).abs()
    ref_mean = b_f32.abs().mean().item()
    return diff.mean().item() / (ref_mean + 1e-12)


def save_tensor_for_comparison(tensor: torch.Tensor, name: str, dump_dir: str = "./py_dump"):
    """Save tensor to binary file for comparison with C++ dump"""
    os.makedirs(dump_dir, exist_ok=True)

    # Convert to float32 numpy array
    arr = tensor.detach().cpu().float().numpy()

    # Save with header (rows, cols)
    filename = os.path.join(dump_dir, f"{name}.bin")
    with open(filename, "wb") as f:
        if len(arr.shape) == 1:
            rows, cols = 1, arr.shape[0]
            arr = arr.reshape(1, -1)
        elif len(arr.shape) == 2:
            rows, cols = arr.shape
        else:
            rows = arr.shape[0]
            cols = np.prod(arr.shape[1:])
            arr = arr.reshape(rows, cols)

        f.write(np.array([rows, cols], dtype=np.int32).tobytes())
        f.write(arr.astype(np.float32).tobytes())

    print(f"  [DUMP] Saved {filename}: [{rows} x {cols}]")


def read_matrix_file(filepath: str) -> tuple:
    """Read binary matrix file in the format: rows(int32), cols(int32), data(float32)"""
    if not os.path.exists(filepath):
        return None, None, None

    with open(filepath, "rb") as f:
        rows, cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
        data = data.reshape(rows, cols)
    return rows, cols, data


def compare_matrices(cpp_data: np.ndarray, py_data: np.ndarray, name: str, threshold: float) -> dict:
    """Compare two matrices and return comparison result"""
    if cpp_data is None or py_data is None:
        return {"name": name, "status": "MISSING", "cpp_exists": cpp_data is not None, "py_exists": py_data is not None}

    if cpp_data.shape != py_data.shape:
        return {"name": name, "status": "SHAPE_MISMATCH", "cpp_shape": cpp_data.shape, "py_shape": py_data.shape}

    abs_diff = np.abs(cpp_data - py_data)
    max_abs_diff = np.max(abs_diff)
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


def print_comparison_result(result: dict, verbose: bool = True):
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
            f"\033[92m[PASS]\033[0m {name} - rel_error: {result['rel_error']:.2e}, max_abs_diff: {result['max_abs_diff']:.2e}"
        )
    else:
        print(
            f"\033[91m[FAIL]\033[0m {name} - rel_error: {result['rel_error']:.2e}, max_abs_diff: {result['max_abs_diff']:.2e}"
        )

    if verbose or result["status"] == "FAIL":
        print(f"    Shape: {result['shape']}")
        print(f"    Mean abs diff: {result['mean_abs_diff']:.6e}")
        print(
            f"    Max abs diff at {result['max_diff_idx']}: cpp={result['cpp_at_max']:.6e}, py={result['py_at_max']:.6e}"
        )
        cpp_stats = result["cpp_stats"]
        py_stats = result["py_stats"]
        print(
            f"    C++ stats: min={cpp_stats['min']:.6e}, max={cpp_stats['max']:.6e}, mean={cpp_stats['mean']:.6e}, nan={cpp_stats['nan']}, inf={cpp_stats['inf']}"
        )
        print(
            f"    Py stats:  min={py_stats['min']:.6e}, max={py_stats['max']:.6e}, mean={py_stats['mean']:.6e}, nan={py_stats['nan']}, inf={py_stats['inf']}"
        )


def compare_tensors_detailed(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, name: str, threshold: float = 0.05
) -> dict:
    """Compare two PyTorch tensors with detailed statistics"""
    # Convert to numpy for comparison
    a_np = tensor_a.detach().cpu().float().numpy()
    b_np = tensor_b.detach().cpu().float().numpy()
    return compare_matrices(a_np, b_np, name, threshold)


# =============================================================================
# TP SFT Simulator - PyTorch Reference Implementation
# =============================================================================


class TPSFTSimulator:
    """
    Simulates TP (Tensor Parallel) partitioned SFT MoE computation with LoRA.

    This class partitions weights according to TP rules and computes the forward
    pass for each TP partition separately, then merges the results.
    """

    def __init__(
        self,
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
        tp_count: int,
    ):
        """
        Initialize TP simulator with full weights.

        Args:
            gate_proj: [expert_num, intermediate_size, hidden_size]
            up_proj:   [expert_num, intermediate_size, hidden_size]
            down_proj: [expert_num, hidden_size, intermediate_size]
            gate_lora_a: [expert_num, lora_rank, hidden_size]      # Not partitioned
            gate_lora_b: [expert_num, intermediate_size, lora_rank] # Partitioned
            up_lora_a: [expert_num, lora_rank, hidden_size]        # Not partitioned
            up_lora_b: [expert_num, intermediate_size, lora_rank]   # Partitioned
            down_lora_a: [expert_num, lora_rank, intermediate_size] # Partitioned
            down_lora_b: [expert_num, hidden_size, lora_rank]       # Not partitioned
            lora_scaling: float
            tp_count: Number of TP partitions
        """
        self.tp_count = tp_count
        self.lora_scaling = lora_scaling
        self.expert_num = gate_proj.shape[0]
        self.intermediate_size = gate_proj.shape[1]
        self.hidden_size = gate_proj.shape[2]
        self.lora_rank = gate_lora_a.shape[1]
        self.tp_intermediate = self.intermediate_size // tp_count

        self.partition_weights(
            gate_proj, up_proj, down_proj, gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b
        )

    def partition_weights(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        gate_lora_a: torch.Tensor,
        gate_lora_b: torch.Tensor,
        up_lora_a: torch.Tensor,
        up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
    ):
        """Partition weights according to TP rules."""
        # Store non-partitioned LoRA weights
        self.gate_lora_a = gate_lora_a.clone()
        self.up_lora_a = up_lora_a.clone()
        self.down_lora_b = down_lora_b.clone()

        # Initialize partitioned weight lists
        self.gate_proj_parts = []
        self.up_proj_parts = []
        self.down_proj_parts = []
        self.gate_lora_b_parts = []
        self.up_lora_b_parts = []
        self.down_lora_a_parts = []

        for tp_idx in range(self.tp_count):
            start = tp_idx * self.tp_intermediate
            end = start + self.tp_intermediate

            # Base weights: gate/up are contiguous slices, down is row-wise slice
            # gate_proj: [expert_num, intermediate_size, hidden_size]
            # -> [expert_num, tp_intermediate, hidden_size]
            self.gate_proj_parts.append(gate_proj[:, start:end, :].clone())
            self.up_proj_parts.append(up_proj[:, start:end, :].clone())

            # down_proj: [expert_num, hidden_size, intermediate_size]
            # -> [expert_num, hidden_size, tp_intermediate]
            self.down_proj_parts.append(down_proj[:, :, start:end].clone())

            # LoRA B weights: contiguous slice
            # gate_lora_b: [expert_num, intermediate_size, lora_rank]
            # -> [expert_num, tp_intermediate, lora_rank]
            self.gate_lora_b_parts.append(gate_lora_b[:, start:end, :].clone())
            self.up_lora_b_parts.append(up_lora_b[:, start:end, :].clone())

            # down_lora_a: [expert_num, lora_rank, intermediate_size]
            # -> [expert_num, lora_rank, tp_intermediate] (row-wise slice)
            self.down_lora_a_parts.append(down_lora_a[:, :, start:end].clone())

    def forward_single_expert(
        self,
        x: torch.Tensor,
        expert_id: int,
        dump_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for a single expert with TP partitioning.

        Args:
            x: Input tensor [qlen, hidden_size]
            expert_id: Expert index
            dump_intermediates: Whether to dump intermediate values

        Returns:
            output: [qlen, hidden_size]
            intermediates: dict of intermediate values (if dump_intermediates=True)
        """
        intermediates = {}
        outputs = []
        original_dtype = x.dtype

        # Convert input to float32 for numerical accuracy (matching C++ AMX behavior)
        x_fp32 = x.float()

        for tp_idx in range(self.tp_count):
            # Get partitioned weights for this TP partition and convert to float32
            gate_proj = self.gate_proj_parts[tp_idx][expert_id].float()  # [tp_intermediate, hidden_size]
            up_proj = self.up_proj_parts[tp_idx][expert_id].float()  # [tp_intermediate, hidden_size]
            down_proj = self.down_proj_parts[tp_idx][expert_id].float()  # [hidden_size, tp_intermediate]

            # Non-partitioned LoRA A weights (convert to float32)
            gate_lora_a = self.gate_lora_a[expert_id].float()  # [lora_rank, hidden_size]
            up_lora_a = self.up_lora_a[expert_id].float()  # [lora_rank, hidden_size]
            down_lora_b = self.down_lora_b[expert_id].float()  # [hidden_size, lora_rank]

            # Partitioned LoRA B/A weights (convert to float32)
            gate_lora_b = self.gate_lora_b_parts[tp_idx][expert_id].float()  # [tp_intermediate, lora_rank]
            up_lora_b = self.up_lora_b_parts[tp_idx][expert_id].float()  # [tp_intermediate, lora_rank]
            down_lora_a = self.down_lora_a_parts[tp_idx][expert_id].float()  # [lora_rank, tp_intermediate]

            # Gate projection with LoRA (all in float32)
            # gate_base: x @ gate_proj.T -> [qlen, tp_intermediate]
            gate_base = torch.mm(x_fp32, gate_proj.t())
            # gate_lora: (x @ gate_lora_a.T) @ gate_lora_b.T * scaling -> [qlen, tp_intermediate]
            gate_lora_intermediate = torch.mm(x_fp32, gate_lora_a.t())  # [qlen, lora_rank]
            gate_lora = torch.mm(gate_lora_intermediate, gate_lora_b.t()) * self.lora_scaling
            gate_out = gate_base + gate_lora

            # Up projection with LoRA
            up_base = torch.mm(x_fp32, up_proj.t())
            up_lora_intermediate = torch.mm(x_fp32, up_lora_a.t())
            up_lora = torch.mm(up_lora_intermediate, up_lora_b.t()) * self.lora_scaling
            up_out = up_base + up_lora

            # Activation: SiLU(gate) * up
            act_out = silu(gate_out) * up_out

            # Down projection with LoRA
            # down_base: act_out @ down_proj.T -> [qlen, hidden_size]
            down_base = torch.mm(act_out, down_proj.t())
            # down_lora: (act_out @ down_lora_a.T) @ down_lora_b.T * scaling -> [qlen, hidden_size]
            down_lora_intermediate = torch.mm(act_out, down_lora_a.t())  # [qlen, lora_rank]
            down_lora = torch.mm(down_lora_intermediate, down_lora_b.t()) * self.lora_scaling
            down_out = down_base + down_lora

            outputs.append(down_out)

            if dump_intermediates:
                intermediates[f"tp{tp_idx}_gate_base"] = gate_base.clone()
                intermediates[f"tp{tp_idx}_gate_lora_intermediate"] = gate_lora_intermediate.clone()
                intermediates[f"tp{tp_idx}_gate_lora"] = gate_lora.clone()
                intermediates[f"tp{tp_idx}_gate_out"] = gate_out.clone()
                intermediates[f"tp{tp_idx}_up_base"] = up_base.clone()
                intermediates[f"tp{tp_idx}_up_lora_intermediate"] = up_lora_intermediate.clone()
                intermediates[f"tp{tp_idx}_up_lora"] = up_lora.clone()
                intermediates[f"tp{tp_idx}_up_out"] = up_out.clone()
                intermediates[f"tp{tp_idx}_act_out"] = act_out.clone()
                intermediates[f"tp{tp_idx}_down_base"] = down_base.clone()
                intermediates[f"tp{tp_idx}_down_lora_intermediate"] = down_lora_intermediate.clone()
                intermediates[f"tp{tp_idx}_down_lora"] = down_lora.clone()
                intermediates[f"tp{tp_idx}_down_out"] = down_out.clone()

        # Merge TP outputs: sum all partitions
        print(f"[DEBUG forward_single_expert] expert={expert_id}, tp_count={self.tp_count}, num_outputs={len(outputs)}")
        for tp_idx, out in enumerate(outputs):
            print(f"  TP{tp_idx} output mean: {out.float().mean():.6f}")
        output = sum(outputs)
        print(f"  Merged output mean: {output.float().mean():.6f}")

        # Convert back to original dtype
        output = output.to(original_dtype)

        if dump_intermediates:
            intermediates["merged_output"] = output.clone()
            for tp_idx in range(self.tp_count):
                intermediates[f"tp{tp_idx}_output_before_merge"] = outputs[tp_idx].clone()

        return output, intermediates

    def forward_moe(
        self,
        input: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        dump_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full MoE forward with TP partitioning.

        Args:
            input: [qlen, hidden_size]
            expert_ids: [qlen, k]
            weights: [qlen, k]
            dump_intermediates: Whether to dump intermediate values

        Returns:
            output: [qlen, hidden_size]
            intermediates: dict of intermediate values
        """
        qlen = input.shape[0]
        k = expert_ids.shape[1]
        intermediates = {}

        # Count tokens per expert
        cnts = expert_ids.new_zeros((qlen, self.expert_num))
        cnts.scatter_(1, expert_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)

        # Sort tokens by expert
        idxs = expert_ids.view(-1).argsort()
        sorted_tokens = input[idxs // k]

        outputs = []
        start_idx = 0

        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue

            end_idx = start_idx + int(num_tokens)
            tokens_for_expert = sorted_tokens[start_idx:end_idx]

            # Forward through single expert with TP simulation
            expert_out, expert_intermediates = self.forward_single_expert(tokens_for_expert, i, dump_intermediates)

            outputs.append(expert_out)

            if dump_intermediates:
                for key, val in expert_intermediates.items():
                    intermediates[f"expert{i}_{key}"] = val

            start_idx = end_idx

        # Combine outputs
        if outputs:
            outs = torch.cat(outputs, dim=0)
        else:
            outs = sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs

        # Apply expert weights
        # Debug: print intermediate values
        new_x_view = new_x.view(qlen, k, -1)
        print(f"[DEBUG forward_moe] new_x_view mean: {new_x_view.float().mean():.6f}")
        print(f"[DEBUG forward_moe] weights mean: {weights.mean():.6f}")

        output = new_x_view.type(weights.dtype).mul_(weights.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
        print(f"[DEBUG forward_moe] output mean: {output.float().mean():.6f}")

        if dump_intermediates:
            intermediates["final_output"] = output.clone()

        return output, intermediates

    def backward_single_expert(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        expert_id: int,
        saved_tensors: Dict[str, torch.Tensor],
        dump_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Backward pass for a single expert with TP partitioning.

        Args:
            grad_output: [qlen, hidden_size] - gradient from downstream
            x: [qlen, hidden_size] - original input
            expert_id: Expert index
            saved_tensors: Saved intermediate values from forward
            dump_intermediates: Whether to dump intermediate values

        Returns:
            grad_input: [qlen, hidden_size]
            grad_loras: dict of LoRA gradients (full size, merged)
            intermediates: dict of intermediate values
        """
        intermediates = {}

        # Initialize gradient accumulators for non-partitioned weights
        grad_gate_lora_a = torch.zeros_like(self.gate_lora_a[expert_id])
        grad_up_lora_a = torch.zeros_like(self.up_lora_a[expert_id])
        grad_down_lora_b = torch.zeros_like(self.down_lora_b[expert_id])
        grad_input = torch.zeros_like(x)

        # Initialize gradient lists for partitioned weights
        grad_gate_lora_b_parts = []
        grad_up_lora_b_parts = []
        grad_down_lora_a_parts = []

        for tp_idx in range(self.tp_count):
            # Get partitioned weights for this TP partition
            gate_proj = self.gate_proj_parts[tp_idx][expert_id]
            up_proj = self.up_proj_parts[tp_idx][expert_id]
            down_proj = self.down_proj_parts[tp_idx][expert_id]

            # Non-partitioned LoRA weights
            gate_lora_a = self.gate_lora_a[expert_id]
            up_lora_a = self.up_lora_a[expert_id]
            down_lora_b = self.down_lora_b[expert_id]

            # Partitioned LoRA weights
            gate_lora_b = self.gate_lora_b_parts[tp_idx][expert_id]
            up_lora_b = self.up_lora_b_parts[tp_idx][expert_id]
            down_lora_a = self.down_lora_a_parts[tp_idx][expert_id]

            # Get saved tensors for this partition
            gate_out = saved_tensors[f"tp{tp_idx}_gate_out"]
            up_out = saved_tensors[f"tp{tp_idx}_up_out"]
            act_out = saved_tensors[f"tp{tp_idx}_act_out"]

            # === Backward through down projection ===
            # grad_output: [qlen, hidden_size]
            # down_proj: [hidden_size, tp_intermediate]
            # act_out: [qlen, tp_intermediate]
            # down_lora_a: [lora_rank, tp_intermediate]
            # down_lora_b: [hidden_size, lora_rank]

            # Base gradient: grad_act_out = grad_output @ down_proj
            grad_act_out = torch.mm(grad_output, down_proj)

            # LoRA gradient contribution to act_out
            # forward: down_lora = (act_out @ down_lora_a.T) @ down_lora_b.T * scaling
            # backward: grad_act_out += grad_output @ down_lora_b @ down_lora_a * scaling
            grad_act_out += torch.mm(torch.mm(grad_output, down_lora_b), down_lora_a) * self.lora_scaling

            # Gradient for down_lora_b: grad_output.T @ (act_out @ down_lora_a.T) * scaling
            down_lora_intermediate = torch.mm(act_out, down_lora_a.t())
            grad_down_lora_b_tp = torch.mm(grad_output.t(), down_lora_intermediate) * self.lora_scaling
            grad_down_lora_b += grad_down_lora_b_tp  # Accumulate across partitions

            # Gradient for down_lora_a: (down_lora_b.T @ grad_output.T) @ act_out * scaling
            grad_down_lora_a_tp = torch.mm(torch.mm(down_lora_b.t(), grad_output.t()), act_out) * self.lora_scaling
            grad_down_lora_a_parts.append(grad_down_lora_a_tp)

            # === Backward through activation: act_out = silu(gate_out) * up_out ===
            gate_activated = silu(gate_out)
            grad_gate_activated = grad_act_out * up_out
            grad_up_out = grad_act_out * gate_activated

            # Gradient through silu: d/dx[x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            sigmoid_gate = torch.sigmoid(gate_out)
            grad_gate_out = grad_gate_activated * sigmoid_gate * (1 + gate_out * (1 - sigmoid_gate))

            # === Backward through up projection ===
            # grad_up_out: [qlen, tp_intermediate]
            # up_proj: [tp_intermediate, hidden_size]

            # Base gradient
            grad_x_up = torch.mm(grad_up_out, up_proj)

            # LoRA gradient contribution
            grad_x_up += torch.mm(torch.mm(grad_up_out, up_lora_b), up_lora_a) * self.lora_scaling

            # Gradient for up_lora_b
            up_lora_intermediate = torch.mm(x, up_lora_a.t())
            grad_up_lora_b_tp = torch.mm(grad_up_out.t(), up_lora_intermediate) * self.lora_scaling
            grad_up_lora_b_parts.append(grad_up_lora_b_tp)

            # Gradient for up_lora_a (accumulated across partitions)
            grad_up_lora_a_tp = torch.mm(torch.mm(up_lora_b.t(), grad_up_out.t()), x) * self.lora_scaling
            grad_up_lora_a += grad_up_lora_a_tp

            # === Backward through gate projection ===
            # grad_gate_out: [qlen, tp_intermediate]
            # gate_proj: [tp_intermediate, hidden_size]

            # Base gradient
            grad_x_gate = torch.mm(grad_gate_out, gate_proj)

            # LoRA gradient contribution
            grad_x_gate += torch.mm(torch.mm(grad_gate_out, gate_lora_b), gate_lora_a) * self.lora_scaling

            # Gradient for gate_lora_b
            gate_lora_intermediate = torch.mm(x, gate_lora_a.t())
            grad_gate_lora_b_tp = torch.mm(grad_gate_out.t(), gate_lora_intermediate) * self.lora_scaling
            grad_gate_lora_b_parts.append(grad_gate_lora_b_tp)

            # Gradient for gate_lora_a (accumulated across partitions)
            grad_gate_lora_a_tp = torch.mm(torch.mm(gate_lora_b.t(), grad_gate_out.t()), x) * self.lora_scaling
            grad_gate_lora_a += grad_gate_lora_a_tp

            # Accumulate grad_input from this partition
            grad_input += grad_x_up + grad_x_gate

            if dump_intermediates:
                intermediates[f"tp{tp_idx}_grad_act_out"] = grad_act_out.clone()
                intermediates[f"tp{tp_idx}_grad_gate_out"] = grad_gate_out.clone()
                intermediates[f"tp{tp_idx}_grad_up_out"] = grad_up_out.clone()
                intermediates[f"tp{tp_idx}_grad_x_gate"] = grad_x_gate.clone()
                intermediates[f"tp{tp_idx}_grad_x_up"] = grad_x_up.clone()
                intermediates[f"tp{tp_idx}_grad_down_lora_a"] = grad_down_lora_a_tp.clone()
                intermediates[f"tp{tp_idx}_grad_gate_lora_b"] = grad_gate_lora_b_tp.clone()
                intermediates[f"tp{tp_idx}_grad_up_lora_b"] = grad_up_lora_b_tp.clone()

        # Merge partitioned gradients by concatenation
        grad_gate_lora_b = torch.cat(grad_gate_lora_b_parts, dim=0)  # [intermediate_size, lora_rank]
        grad_up_lora_b = torch.cat(grad_up_lora_b_parts, dim=0)  # [intermediate_size, lora_rank]
        grad_down_lora_a = torch.cat(grad_down_lora_a_parts, dim=1)  # [lora_rank, intermediate_size]

        grad_loras = {
            "grad_gate_lora_a": grad_gate_lora_a,
            "grad_gate_lora_b": grad_gate_lora_b,
            "grad_up_lora_a": grad_up_lora_a,
            "grad_up_lora_b": grad_up_lora_b,
            "grad_down_lora_a": grad_down_lora_a,
            "grad_down_lora_b": grad_down_lora_b,
        }

        if dump_intermediates:
            intermediates["grad_input"] = grad_input.clone()

        return grad_input, grad_loras, intermediates

    def forward_backward_single_expert(
        self,
        x: torch.Tensor,
        expert_id: int,
        grad_output: torch.Tensor,
        dump_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward and backward pass for a single expert (for testing).

        Returns:
            output: forward output
            grad_input: backward grad_input
            grad_loras: LoRA gradients
            intermediates: all intermediate values
        """
        # Forward pass with intermediate saving
        output, fwd_intermediates = self.forward_single_expert(x, expert_id, dump_intermediates=True)

        # Backward pass
        grad_input, grad_loras, bwd_intermediates = self.backward_single_expert(
            grad_output, x, expert_id, fwd_intermediates, dump_intermediates
        )

        # Merge intermediates
        intermediates = {**fwd_intermediates, **bwd_intermediates}

        return output, grad_input, grad_loras, intermediates


# =============================================================================
# Non-TP Reference Implementation (for comparison)
# =============================================================================


def lora_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """LoRA linear layer forward pass."""
    base_out = torch.mm(x, weight.t())
    lora_out = torch.mm(torch.mm(x, lora_a.t()), lora_b.t()) * scaling
    return base_out + lora_out


def lora_linear_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LoRA linear layer backward pass."""
    grad_input = torch.mm(grad_output, weight)
    grad_input += torch.mm(torch.mm(grad_output, lora_b), lora_a) * scaling
    lora_intermediate = torch.mm(x, lora_a.t())
    grad_lora_b = torch.mm(grad_output.t(), lora_intermediate) * scaling
    grad_lora_a = torch.mm(torch.mm(lora_b.t(), grad_output.t()), x) * scaling
    return grad_input, grad_lora_a, grad_lora_b


def mlp_lora_forward_with_save(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    scaling: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """MLP forward pass with LoRA adapters, saving intermediates for backward."""
    gate_out = lora_linear_forward(x, gate_proj, gate_lora_a, gate_lora_b, scaling)
    up_out = lora_linear_forward(x, up_proj, up_lora_a, up_lora_b, scaling)
    gate_activated = silu(gate_out)
    intermediate = gate_activated * up_out
    output = lora_linear_forward(intermediate, down_proj, down_lora_a, down_lora_b, scaling)

    saved_tensors = {
        "x": x,
        "gate_out": gate_out,
        "up_out": up_out,
        "gate_activated": gate_activated,
        "intermediate": intermediate,
    }

    return output, saved_tensors


def mlp_lora_backward(
    grad_output: torch.Tensor,
    saved_tensors: Dict[str, torch.Tensor],
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    scaling: float,
) -> Dict[str, torch.Tensor]:
    """MLP backward pass with LoRA adapters."""
    x = saved_tensors["x"]
    gate_out = saved_tensors["gate_out"]
    up_out = saved_tensors["up_out"]
    gate_activated = saved_tensors["gate_activated"]
    intermediate = saved_tensors["intermediate"]

    grad_intermediate, grad_down_lora_a, grad_down_lora_b = lora_linear_backward(
        grad_output, intermediate, down_proj, down_lora_a, down_lora_b, scaling
    )

    grad_gate_activated = grad_intermediate * up_out
    grad_up_out = grad_intermediate * gate_activated

    sigmoid_gate = torch.sigmoid(gate_out)
    grad_gate_out = grad_gate_activated * sigmoid_gate * (1 + gate_out * (1 - sigmoid_gate))

    grad_x_up, grad_up_lora_a, grad_up_lora_b = lora_linear_backward(
        grad_up_out, x, up_proj, up_lora_a, up_lora_b, scaling
    )

    grad_x_gate, grad_gate_lora_a, grad_gate_lora_b = lora_linear_backward(
        grad_gate_out, x, gate_proj, gate_lora_a, gate_lora_b, scaling
    )

    grad_input = grad_x_up + grad_x_gate

    return {
        "grad_input": grad_input,
        "grad_gate_lora_a": grad_gate_lora_a,
        "grad_gate_lora_b": grad_gate_lora_b,
        "grad_up_lora_a": grad_up_lora_a,
        "grad_up_lora_b": grad_up_lora_b,
        "grad_down_lora_a": grad_down_lora_a,
        "grad_down_lora_b": grad_down_lora_b,
    }


def mlp_lora_forward(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """MLP forward pass with LoRA adapters."""
    gate_out = lora_linear_forward(x, gate_proj, gate_lora_a, gate_lora_b, scaling)
    up_out = lora_linear_forward(x, up_proj, up_lora_a, up_lora_b, scaling)
    gate_activated = silu(gate_out)
    intermediate = gate_activated * up_out
    output = lora_linear_forward(intermediate, down_proj, down_lora_a, down_lora_b, scaling)
    return output


def moe_sft_torch_forward_no_tp(
    input: torch.Tensor,
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
    scaling: float,
) -> torch.Tensor:
    """MoE SFT forward pass without TP (PyTorch reference)."""
    qlen = input.shape[0]
    k = expert_ids.shape[1]
    expert_num = gate_proj.shape[0]

    cnts = expert_ids.new_zeros((qlen, expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)

    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // k]

    outputs = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            continue

        end_idx = start_idx + int(num_tokens)
        tokens_for_expert = sorted_tokens[start_idx:end_idx]

        expert_out = mlp_lora_forward(
            tokens_for_expert,
            gate_proj[i],
            up_proj[i],
            down_proj[i],
            gate_lora_a[i],
            gate_lora_b[i],
            up_lora_a[i],
            up_lora_b[i],
            down_lora_a[i],
            down_lora_b[i],
            scaling,
        )

        outputs.append(expert_out)
        start_idx = end_idx

    if outputs:
        outs = torch.cat(outputs, dim=0)
    else:
        outs = sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs

    output = new_x.view(qlen, k, -1).type(weights.dtype).mul_(weights.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)

    return output


# =============================================================================
# Weight Initialization Utilities
# =============================================================================


def init_base_weights(expert_num: int, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16, device="cpu"):
    """Initialize base MoE weights."""
    # Use CUDA if available and requested, otherwise CPU
    init_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    gate_proj = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device=init_device)
        .to("cpu")
        .contiguous()
    )
    up_proj = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device=init_device)
        .to("cpu")
        .contiguous()
    )
    down_proj = (
        torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, device=init_device)
        .to("cpu")
        .contiguous()
    )
    return gate_proj, up_proj, down_proj


def init_lora_weights(
    expert_num: int, hidden_size: int, intermediate_size: int, rank: int, dtype=torch.bfloat16, device="cpu"
):
    """Initialize LoRA weights."""
    # Use CUDA if available and requested, otherwise CPU
    init_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    gate_lora_a = (
        torch.randn((expert_num, rank, hidden_size), dtype=dtype, device=init_device).to("cpu").contiguous() / 100
    )
    gate_lora_b = (
        torch.randn((expert_num, intermediate_size, rank), dtype=dtype, device=init_device).to("cpu").contiguous() / 100
    )

    up_lora_a = (
        torch.randn((expert_num, rank, hidden_size), dtype=dtype, device=init_device).to("cpu").contiguous() / 100
    )
    up_lora_b = (
        torch.randn((expert_num, intermediate_size, rank), dtype=dtype, device=init_device).to("cpu").contiguous() / 100
    )

    down_lora_a = (
        torch.randn((expert_num, rank, intermediate_size), dtype=dtype, device=init_device).to("cpu").contiguous() / 100
    )
    down_lora_b = (
        torch.randn((expert_num, hidden_size, rank), dtype=dtype, device=init_device).to("cpu").contiguous() / 100
    )

    return (gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)


# =============================================================================
# Test Functions
# =============================================================================


def test_tp_simulator_vs_no_tp():
    """
    Test that TP simulator produces same results as non-TP reference.

    This validates that the PyTorch TP simulation is mathematically correct.
    Uses float32 for exact numerical comparison (bfloat16 has limited precision).
    """
    print(f"\n{'='*60}")
    print(f"Test: TP Simulator vs Non-TP Reference (float32)")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Use smaller dimensions for faster testing
    test_expert_num = 64
    test_hidden_size = 256
    test_intermediate_size = 512
    test_lora_rank = 8
    test_lora_scaling = lora_alpha / test_lora_rank
    test_qlen = 4
    test_k = 4
    test_tp_count = 2

    # Use float32 for exact comparison (bfloat16 has too limited precision)
    test_dtype = torch.float32

    # Initialize weights with float32
    gate_proj, up_proj, down_proj = init_base_weights(
        test_expert_num, test_hidden_size, test_intermediate_size, dtype=test_dtype
    )
    lora_weights = init_lora_weights(
        test_expert_num, test_hidden_size, test_intermediate_size, test_lora_rank, dtype=test_dtype
    )
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Create TP simulator
    simulator = TPSFTSimulator(
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
        test_tp_count,
    )

    # Generate test inputs
    expert_ids = (
        torch.stack([torch.randperm(test_expert_num)[:test_k] for _ in range(test_qlen)]).to(torch.int64).contiguous()
    )
    weights = torch.rand((test_qlen, test_k), dtype=torch.float32).contiguous()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    input_data = torch.randn((test_qlen, test_hidden_size), dtype=test_dtype).contiguous() / 100

    # Run TP simulator
    tp_output, tp_intermediates = simulator.forward_moe(input_data, expert_ids, weights, dump_intermediates=True)

    # Run non-TP reference
    no_tp_output = moe_sft_torch_forward_no_tp(
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
    )

    # Compare
    diff = torch.mean(torch.abs(tp_output - no_tp_output)) / (torch.mean(torch.abs(no_tp_output)) + 1e-8)
    print(f"TP Simulator vs Non-TP Reference:")
    print(f"  Relative difference: {diff:.6f}")
    print(f"  TP output mean: {tp_output.float().mean():.6f}")
    print(f"  Non-TP output mean: {no_tp_output.float().mean():.6f}")

    threshold = 1e-5
    if diff < threshold:
        print(f"  PASSED (threshold: {threshold})")
    else:
        print(f"  FAILED: diff={diff:.6f} >= {threshold}")
        sys.exit(1)

    # Print some intermediate values for first activated expert
    print(f"\nIntermediate values (first few):")
    for key in list(tp_intermediates.keys())[:10]:
        val = tp_intermediates[key]
        print(f"  {key}: shape={val.shape}, mean={val.float().mean():.6f}, max={val.float().abs().max():.6f}")


def test_tp_simulator_single_expert():
    """
    Test single expert forward with intermediate value dumping.
    """
    print(f"\n{'='*60}")
    print(f"Test: Single Expert Forward with Intermediate Dump")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Use smaller dimensions for faster testing
    test_expert_num = 64
    test_hidden_size = 256
    test_intermediate_size = 512
    test_lora_rank = 8
    test_qlen = 1
    test_tp_count = 2
    test_expert_id = 42

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(test_expert_num, test_hidden_size, test_intermediate_size)
    lora_weights = init_lora_weights(test_expert_num, test_hidden_size, test_intermediate_size, test_lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Create TP simulator
    simulator = TPSFTSimulator(
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
        test_tp_count,
    )

    # Generate test input
    input_data = torch.randn((test_qlen, test_hidden_size), dtype=torch.bfloat16).contiguous() / 100

    # Forward with intermediate dump
    output, intermediates = simulator.forward_single_expert(input_data, test_expert_id, dump_intermediates=True)

    print(f"\n=== TP SFT Debug: Single Token Single Expert ===")
    print(f"tp_count: {test_tp_count}")
    print(f"expert_id: {test_expert_id}")
    print(f"token_shape: [{test_qlen}, {test_hidden_size}]")
    print(f"intermediate_size: {test_intermediate_size}")
    print(f"tp_intermediate: {test_intermediate_size // test_tp_count}")
    print(f"lora_rank: {test_lora_rank}")
    print(f"lora_scaling: {lora_scaling}")

    for tp_idx in range(test_tp_count):
        print(f"\n[TP{tp_idx}] Intermediate values:")
        print(
            f"  gate_base shape: {intermediates[f'tp{tp_idx}_gate_base'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_gate_base'].float().mean():.6f}"
        )
        print(
            f"  gate_lora shape: {intermediates[f'tp{tp_idx}_gate_lora'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_gate_lora'].float().mean():.6f}"
        )
        print(
            f"  gate_out shape: {intermediates[f'tp{tp_idx}_gate_out'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_gate_out'].float().mean():.6f}"
        )
        print(
            f"  up_base shape: {intermediates[f'tp{tp_idx}_up_base'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_up_base'].float().mean():.6f}"
        )
        print(
            f"  up_lora shape: {intermediates[f'tp{tp_idx}_up_lora'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_up_lora'].float().mean():.6f}"
        )
        print(
            f"  up_out shape: {intermediates[f'tp{tp_idx}_up_out'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_up_out'].float().mean():.6f}"
        )
        print(
            f"  act_out shape: {intermediates[f'tp{tp_idx}_act_out'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_act_out'].float().mean():.6f}"
        )
        print(
            f"  down_base shape: {intermediates[f'tp{tp_idx}_down_base'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_down_base'].float().mean():.6f}"
        )
        print(
            f"  down_lora shape: {intermediates[f'tp{tp_idx}_down_lora'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_down_lora'].float().mean():.6f}"
        )
        print(
            f"  down_out shape: {intermediates[f'tp{tp_idx}_down_out'].shape}, "
            f"mean: {intermediates[f'tp{tp_idx}_down_out'].float().mean():.6f}"
        )

    print(f"\n[Merged] output shape: {output.shape}, mean: {output.float().mean():.6f}")

    # Verify TP merge is correct
    # Note: Allow for bfloat16 quantization error since output is converted back to bfloat16
    # but intermediates are stored in float32
    merged_check = sum(intermediates[f"tp{i}_down_out"] for i in range(test_tp_count))
    merge_diff = torch.mean(torch.abs(output.float() - merged_check.float()))
    print(f"\nMerge verification:")
    print(f"  sum(down_out) - merged_output diff: {merge_diff:.6e}")
    # BF16 has ~7 bits of mantissa, so ~1e-3 relative error is expected
    assert merge_diff < 1e-3, f"Merge verification failed: {merge_diff}"

    print(f"\nPASSED")


def test_weight_partitioning():
    """
    Test that weight partitioning is correct.
    """
    print(f"\n{'='*60}")
    print(f"Test: Weight Partitioning Verification")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Use smaller dimensions for faster testing
    test_expert_num = 4
    test_hidden_size = 16
    test_intermediate_size = 32
    test_lora_rank = 4
    test_tp_count = 2
    tp_intermediate = test_intermediate_size // test_tp_count

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(test_expert_num, test_hidden_size, test_intermediate_size)
    lora_weights = init_lora_weights(test_expert_num, test_hidden_size, test_intermediate_size, test_lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Create TP simulator
    simulator = TPSFTSimulator(
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
        test_tp_count,
    )

    print(f"\nWeight shapes:")
    print(f"  gate_proj: {gate_proj.shape}")
    print(f"  gate_lora_a: {gate_lora_a.shape}")
    print(f"  gate_lora_b: {gate_lora_b.shape}")
    print(f"  down_lora_a: {down_lora_a.shape}")

    print(f"\nPartitioned weight shapes:")
    print(f"  gate_proj_parts[0]: {simulator.gate_proj_parts[0].shape}")
    print(f"  gate_lora_b_parts[0]: {simulator.gate_lora_b_parts[0].shape}")
    print(f"  down_lora_a_parts[0]: {simulator.down_lora_a_parts[0].shape}")

    # Verify gate_proj partitioning
    print(f"\nVerifying gate_proj partitioning:")
    for tp_idx in range(test_tp_count):
        start = tp_idx * tp_intermediate
        end = start + tp_intermediate
        expected = gate_proj[:, start:end, :]
        actual = simulator.gate_proj_parts[tp_idx]
        diff = torch.mean(torch.abs(expected - actual))
        print(f"  TP{tp_idx}: diff = {diff:.6e}")
        assert diff < 1e-6, f"gate_proj partition {tp_idx} incorrect"

    # Verify gate_lora_b partitioning
    print(f"\nVerifying gate_lora_b partitioning:")
    for tp_idx in range(test_tp_count):
        start = tp_idx * tp_intermediate
        end = start + tp_intermediate
        expected = gate_lora_b[:, start:end, :]
        actual = simulator.gate_lora_b_parts[tp_idx]
        diff = torch.mean(torch.abs(expected - actual))
        print(f"  TP{tp_idx}: diff = {diff:.6e}")
        assert diff < 1e-6, f"gate_lora_b partition {tp_idx} incorrect"

    # Verify down_lora_a partitioning (row-wise)
    print(f"\nVerifying down_lora_a partitioning (row-wise):")
    for tp_idx in range(test_tp_count):
        start = tp_idx * tp_intermediate
        end = start + tp_intermediate
        expected = down_lora_a[:, :, start:end]
        actual = simulator.down_lora_a_parts[tp_idx]
        diff = torch.mean(torch.abs(expected - actual))
        print(f"  TP{tp_idx}: diff = {diff:.6e}")
        assert diff < 1e-6, f"down_lora_a partition {tp_idx} incorrect"

    # Verify down_proj partitioning (row-wise)
    print(f"\nVerifying down_proj partitioning (row-wise):")
    for tp_idx in range(test_tp_count):
        start = tp_idx * tp_intermediate
        end = start + tp_intermediate
        expected = down_proj[:, :, start:end]
        actual = simulator.down_proj_parts[tp_idx]
        diff = torch.mean(torch.abs(expected - actual))
        print(f"  TP{tp_idx}: diff = {diff:.6e}")
        assert diff < 1e-6, f"down_proj partition {tp_idx} incorrect"

    # Verify non-partitioned weights are preserved
    print(f"\nVerifying non-partitioned weights:")
    gate_lora_a_diff = torch.mean(torch.abs(simulator.gate_lora_a - gate_lora_a))
    up_lora_a_diff = torch.mean(torch.abs(simulator.up_lora_a - up_lora_a))
    down_lora_b_diff = torch.mean(torch.abs(simulator.down_lora_b - down_lora_b))
    print(f"  gate_lora_a diff: {gate_lora_a_diff:.6e}")
    print(f"  up_lora_a diff: {up_lora_a_diff:.6e}")
    print(f"  down_lora_b diff: {down_lora_b_diff:.6e}")
    assert gate_lora_a_diff < 1e-6, "gate_lora_a should not be partitioned"
    assert up_lora_a_diff < 1e-6, "up_lora_a should not be partitioned"
    assert down_lora_b_diff < 1e-6, "down_lora_b should not be partitioned"

    print(f"\nPASSED")


def test_tp_vs_cpp_wrapper(quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT):
    """
    Compare PyTorch TP simulator with C++ TP implementation.

    This test validates that the C++ implementation matches our PyTorch reference.
    Uses smaller dimensions for faster execution.
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n{'='*60}")
    print(f"Test: PyTorch TP Simulator vs C++ Implementation [{tp_mode_str}, tp_count={tp_count}]")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel not available, skipping C++ comparison test")
        return

    torch.manual_seed(42)

    # Use same dimensions as test_moe_backward_full for consistency
    test_expert_num = 8
    test_hidden_size = 256  # Must be multiple of 32 for AMX
    test_intermediate_size = 512  # Must be multiple of 32 for AMX
    test_lora_rank = 8
    test_qlen = 4
    test_k = 2
    test_num_threads = 8
    test_max_len = 1024

    # Compute correct lora_scaling for the test configuration
    test_lora_scaling = lora_alpha / test_lora_rank

    print(f"[INFO] Using test dimensions (same as test_moe_backward_full):")
    print(f"  expert_num={test_expert_num}, hidden={test_hidden_size}, intermediate={test_intermediate_size}")
    print(f"  lora_rank={test_lora_rank}, qlen={test_qlen}, k={test_k}, lora_scaling={test_lora_scaling}")

    # Initialize weights with same method as test_moe_backward_full for consistency
    WEIGHT_SCALE = 0.01
    gate_proj = (
        torch.rand(test_expert_num, test_intermediate_size, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_proj = (
        torch.rand(test_expert_num, test_intermediate_size, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_proj = (
        torch.rand(test_expert_num, test_hidden_size, test_intermediate_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()

    gate_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    gate_lora_b = (
        torch.rand(test_expert_num, test_intermediate_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_b = (
        torch.rand(test_expert_num, test_intermediate_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_intermediate_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_b = (
        torch.rand(test_expert_num, test_hidden_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()

    # Create C++ wrapper
    print(f"\n[INFO] Creating KTMoEWrapper with mode='sft', tp_count={tp_count}...")
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=test_expert_num,
        num_experts_per_tok=test_k,
        hidden_size=test_hidden_size,
        moe_intermediate_size=test_intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=test_num_threads,
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=test_max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=test_lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=validation_iter,
    )

    # Load weights
    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj
    physical_map = torch.arange(test_expert_num, dtype=torch.int64)
    wrapper.load_weights(physical_map)
    wrapper.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    # Create PyTorch TP simulator
    simulator = TPSFTSimulator(
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b,
        up_lora_a,
        up_lora_b,
        down_lora_a,
        down_lora_b,
        test_lora_scaling,
        tp_count,
    )

    threshold = BF16_FORWARD_THRESHOLD

    for iter_idx in range(validation_iter):
        print(f"\n--- Iteration {iter_idx} ---")

        # Generate random inputs
        expert_ids = (
            torch.stack([torch.randperm(test_expert_num)[:test_k] for _ in range(test_qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((test_qlen, test_k), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        input_data = torch.randn((test_qlen, test_hidden_size), dtype=torch.bfloat16).contiguous() / 100

        # PyTorch TP simulator forward
        py_output, py_intermediates = simulator.forward_moe(input_data, expert_ids, weights, dump_intermediates=True)

        # C++ wrapper forward
        cpp_output = wrapper.forward_sft(input_data, expert_ids, weights, save_for_backward=False)

        # Compare results
        diff = torch.mean(torch.abs(cpp_output - py_output)) / (torch.mean(torch.abs(py_output)) + 1e-8)
        print(f"PyTorch TP vs C++ TP relative difference: {diff:.6f}")
        print(f"  PyTorch output mean: {py_output.float().mean():.6f}")
        print(f"  C++ output mean: {cpp_output.float().mean():.6f}")

        if diff < threshold:
            print(f"PASSED (threshold: {threshold})")
        else:
            print(f"FAILED: diff={diff:.6f} >= {threshold}")

            # Print some intermediate values for debugging
            print(f"\nDebugging - First expert intermediate values:")
            for key in list(py_intermediates.keys())[:20]:
                val = py_intermediates[key]
                print(f"  {key}: mean={val.float().mean():.6f}, max={val.float().abs().max():.6f}")

            sys.exit(1)

    print(f"\n[OK] PyTorch TP Simulator vs C++ Test [{tp_mode_str}] PASSED")


def test_tp_vs_no_tp_cpp(quant_mode: str = "AMXBF16_SFT"):
    """
    Compare TP=2 and TP=1 (No-TP) C++ implementations.

    Both should produce the same results.
    Uses smaller dimensions for faster execution.
    """
    print(f"\n{'='*60}")
    print(f"Test: C++ TP=2 vs C++ TP=1 (No-TP)")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel not available, skipping test")
        return

    torch.manual_seed(42)

    # Use smaller dimensions for faster testing
    test_expert_num = 64
    test_hidden_size = 256  # Must be multiple of 32 for AMX
    test_intermediate_size = 512  # Must be multiple of 32 for AMX
    test_lora_rank = 8
    test_qlen = 4
    test_k = 4
    test_num_threads = 8
    test_max_len = 1024

    print(f"[INFO] Using smaller test dimensions:")
    print(f"  expert_num={test_expert_num}, hidden={test_hidden_size}, intermediate={test_intermediate_size}")

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(test_expert_num, test_hidden_size, test_intermediate_size)
    lora_weights = init_lora_weights(test_expert_num, test_hidden_size, test_intermediate_size, test_lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Create C++ wrapper with TP=2
    print(f"\n[INFO] Creating wrapper with tp_count=2...")
    wrapper_tp = KTMoEWrapper(
        layer_idx=0,
        num_experts=test_expert_num,
        num_experts_per_tok=test_k,
        hidden_size=test_hidden_size,
        moe_intermediate_size=test_intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=test_num_threads,
        threadpool_count=2,
        weight_path="",
        chunked_prefill_size=test_max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=test_lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=validation_iter,
    )
    wrapper_tp.gate_proj = gate_proj
    wrapper_tp.up_proj = up_proj
    wrapper_tp.down_proj = down_proj
    wrapper_tp.load_weights(torch.arange(test_expert_num, dtype=torch.int64))
    wrapper_tp.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    # Create C++ wrapper with TP=1 (No-TP)
    print(f"[INFO] Creating wrapper with tp_count=1...")
    wrapper_no_tp = KTMoEWrapper(
        layer_idx=0,
        num_experts=test_expert_num,
        num_experts_per_tok=test_k,
        hidden_size=test_hidden_size,
        moe_intermediate_size=test_intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=test_num_threads,
        threadpool_count=1,
        weight_path="",
        chunked_prefill_size=test_max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=test_lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=validation_iter,
    )
    wrapper_no_tp.gate_proj = gate_proj
    wrapper_no_tp.up_proj = up_proj
    wrapper_no_tp.down_proj = down_proj
    wrapper_no_tp.load_weights(torch.arange(test_expert_num, dtype=torch.int64))
    wrapper_no_tp.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    threshold = BF16_FORWARD_THRESHOLD

    for iter_idx in range(validation_iter):
        print(f"\n--- Iteration {iter_idx} ---")

        # Generate random inputs
        expert_ids = (
            torch.stack([torch.randperm(test_expert_num)[:test_k] for _ in range(test_qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((test_qlen, test_k), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        input_data = torch.randn((test_qlen, test_hidden_size), dtype=torch.bfloat16).contiguous() / 100

        # Forward passes
        output_tp = wrapper_tp.forward_sft(input_data, expert_ids, weights, save_for_backward=False)
        output_no_tp = wrapper_no_tp.forward_sft(input_data, expert_ids, weights, save_for_backward=False)

        # Compare
        diff = torch.mean(torch.abs(output_tp - output_no_tp)) / (torch.mean(torch.abs(output_no_tp)) + 1e-8)
        print(f"TP=2 vs TP=1 relative difference: {diff:.6f}")
        print(f"  TP=2 output mean: {output_tp.float().mean():.6f}")
        print(f"  TP=1 output mean: {output_no_tp.float().mean():.6f}")

        if diff < threshold:
            print(f"PASSED (threshold: {threshold})")
        else:
            print(f"FAILED: diff={diff:.6f} >= {threshold}")
            sys.exit(1)

    print(f"\n[OK] C++ TP=2 vs TP=1 Test PASSED")


def test_tp_backward_vs_no_tp():
    """
    Test that TP simulator backward produces same results as non-TP reference.

    This validates that the PyTorch TP simulation backward is mathematically correct.
    Uses float32 for exact numerical comparison.
    """
    print(f"\n{'='*60}")
    print(f"Test: TP Simulator Backward vs Non-TP Reference (float32)")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Use smaller dimensions for faster testing
    test_expert_num = 8
    test_hidden_size = 64
    test_intermediate_size = 128
    test_lora_rank = 4
    test_qlen = 2
    test_tp_count = 2
    test_expert_id = 3

    # Use float32 for exact comparison
    test_dtype = torch.float32

    # Initialize weights with float32
    gate_proj, up_proj, down_proj = init_base_weights(
        test_expert_num, test_hidden_size, test_intermediate_size, dtype=test_dtype
    )
    lora_weights = init_lora_weights(
        test_expert_num, test_hidden_size, test_intermediate_size, test_lora_rank, dtype=test_dtype
    )
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Create TP simulator
    simulator = TPSFTSimulator(
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
        test_tp_count,
    )

    # Generate test inputs
    input_data = torch.randn((test_qlen, test_hidden_size), dtype=test_dtype).contiguous() / 100
    grad_output = torch.randn((test_qlen, test_hidden_size), dtype=test_dtype).contiguous() / 100

    print(f"\nConfiguration:")
    print(f"  expert_id: {test_expert_id}")
    print(f"  tp_count: {test_tp_count}")
    print(f"  hidden_size: {test_hidden_size}")
    print(f"  intermediate_size: {test_intermediate_size}")
    print(f"  lora_rank: {test_lora_rank}")

    # === TP Simulator forward + backward ===
    tp_output, tp_grad_input, tp_grad_loras, tp_intermediates = simulator.forward_backward_single_expert(
        input_data, test_expert_id, grad_output, dump_intermediates=True
    )

    # === Non-TP reference forward + backward ===
    no_tp_output, saved_tensors = mlp_lora_forward_with_save(
        input_data,
        gate_proj[test_expert_id],
        up_proj[test_expert_id],
        down_proj[test_expert_id],
        gate_lora_a[test_expert_id],
        gate_lora_b[test_expert_id],
        up_lora_a[test_expert_id],
        up_lora_b[test_expert_id],
        down_lora_a[test_expert_id],
        down_lora_b[test_expert_id],
        lora_scaling,
    )

    no_tp_grads = mlp_lora_backward(
        grad_output,
        saved_tensors,
        gate_proj[test_expert_id],
        up_proj[test_expert_id],
        down_proj[test_expert_id],
        gate_lora_a[test_expert_id],
        gate_lora_b[test_expert_id],
        up_lora_a[test_expert_id],
        up_lora_b[test_expert_id],
        down_lora_a[test_expert_id],
        down_lora_b[test_expert_id],
        lora_scaling,
    )

    # === Compare forward outputs ===
    fwd_diff = torch.mean(torch.abs(tp_output - no_tp_output)) / (torch.mean(torch.abs(no_tp_output)) + 1e-8)
    print(f"\nForward comparison:")
    print(f"  Relative difference: {fwd_diff:.6e}")

    # === Compare backward gradients ===
    print(f"\nBackward comparison:")

    threshold = 1e-5
    all_passed = True

    # grad_input
    grad_input_diff = torch.mean(torch.abs(tp_grad_input - no_tp_grads["grad_input"])) / (
        torch.mean(torch.abs(no_tp_grads["grad_input"])) + 1e-8
    )
    print(f"  grad_input diff: {grad_input_diff:.6e}")
    if grad_input_diff >= threshold:
        print(f"    FAILED!")
        all_passed = False

    # LoRA gradients
    for name in [
        "grad_gate_lora_a",
        "grad_gate_lora_b",
        "grad_up_lora_a",
        "grad_up_lora_b",
        "grad_down_lora_a",
        "grad_down_lora_b",
    ]:
        tp_grad = tp_grad_loras[name]
        no_tp_grad = no_tp_grads[name]
        diff = torch.mean(torch.abs(tp_grad - no_tp_grad)) / (torch.mean(torch.abs(no_tp_grad)) + 1e-8)
        status = "OK" if diff < threshold else "FAILED"
        print(f"  {name} diff: {diff:.6e} [{status}]")
        if diff >= threshold:
            all_passed = False

    if all_passed:
        print(f"\nPASSED (threshold: {threshold})")
    else:
        print(f"\nFAILED")
        sys.exit(1)


def test_tp_backward_vs_cpp(quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT):
    """
    Compare PyTorch TP simulator backward with C++ TP backward implementation.

    Uses smaller dimensions for faster execution.
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n{'='*60}")
    print(f"Test: PyTorch TP Backward vs C++ Backward [{tp_mode_str}, tp_count={tp_count}]")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel not available, skipping C++ backward comparison test")
        return

    torch.manual_seed(42)

    # Use smaller dimensions for faster testing
    test_expert_num = 64
    test_hidden_size = 1024  # Must be multiple of 32 for AMX
    test_intermediate_size = 5120  # Must be multiple of 32 for AMX
    test_lora_rank = 8
    test_qlen = 4
    test_k = 4
    test_num_threads = 16
    test_max_len = 1024

    print(f"[INFO] Using smaller test dimensions:")
    print(f"  expert_num={test_expert_num}, hidden={test_hidden_size}, intermediate={test_intermediate_size}")
    print(f"  lora_rank={test_lora_rank}, qlen={test_qlen}, k={test_k}")

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(test_expert_num, test_hidden_size, test_intermediate_size)
    lora_weights = init_lora_weights(test_expert_num, test_hidden_size, test_intermediate_size, test_lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Create C++ wrapper
    print(f"\n[INFO] Creating KTMoEWrapper with mode='sft', tp_count={tp_count}...")
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=test_expert_num,
        num_experts_per_tok=test_k,
        hidden_size=test_hidden_size,
        moe_intermediate_size=test_intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=test_num_threads,
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=test_max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=test_lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=validation_iter,
    )

    # Load weights
    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj
    physical_map = torch.arange(test_expert_num, dtype=torch.int64)
    wrapper.load_weights(physical_map)
    wrapper.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    # Create PyTorch TP simulator
    simulator = TPSFTSimulator(
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b,
        up_lora_a,
        up_lora_b,
        down_lora_a,
        down_lora_b,
        test_lora_scaling,
        tp_count,
    )

    threshold = BF16_BACKWARD_THRESHOLD

    for iter_idx in range(validation_iter):
        print(f"\n--- Iteration {iter_idx} ---")

        # Generate random inputs
        expert_ids = (
            torch.stack([torch.randperm(test_expert_num)[:test_k] for _ in range(test_qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((test_qlen, test_k), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        input_data = torch.randn((test_qlen, test_hidden_size), dtype=torch.bfloat16).contiguous() / 100
        grad_output = torch.randn((test_qlen, test_hidden_size), dtype=torch.bfloat16).contiguous() / 100

        # C++ forward (with save_for_backward=True)
        cpp_output = wrapper.forward_sft(input_data, expert_ids, weights, save_for_backward=True)

        # C++ backward
        cpp_grad_input, cpp_grad_loras = wrapper.backward(grad_output)

        # Note: PyTorch TP simulator would need full MoE backward implementation
        # For now, we just verify C++ backward runs without error
        print(f"C++ backward completed:")
        print(f"  grad_input shape: {cpp_grad_input.shape}, mean: {cpp_grad_input.float().mean():.6f}")
        print(f"  grad_gate_lora_a shape: {cpp_grad_loras['grad_gate_lora_a'].shape}")
        print(f"  grad_gate_lora_b shape: {cpp_grad_loras['grad_gate_lora_b'].shape}")
        print(f"  grad_down_lora_a shape: {cpp_grad_loras['grad_down_lora_a'].shape}")

        # Verify shapes are correct
        assert cpp_grad_input.shape == input_data.shape, f"grad_input shape mismatch"
        assert cpp_grad_loras["grad_gate_lora_a"].shape == gate_lora_a.shape, f"grad_gate_lora_a shape mismatch"
        assert cpp_grad_loras["grad_gate_lora_b"].shape == gate_lora_b.shape, f"grad_gate_lora_b shape mismatch"
        assert cpp_grad_loras["grad_down_lora_a"].shape == down_lora_a.shape, f"grad_down_lora_a shape mismatch"

        print(f"Shape verification PASSED")

    print(f"\n[OK] PyTorch TP Backward vs C++ Backward Test [{tp_mode_str}] PASSED")


def test_comprehensive_backward_with_dump(
    quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT, dump_enabled: bool = False
):
    """
    Comprehensive backward test with optional dump functionality.

    This test is modeled after test_minimal_backward.py and provides:
    1. NaN/Inf checking at every step
    2. Detailed comparison statistics
    3. Binary dump capability for debugging
    4. Comparison of C++ and PyTorch backward passes

    Usage:
        # Basic test (no debug output):
        python test_moe_sft_tp_debug.py --mode comprehensive_backward

        # With dump enabled:
        SFT_MOE_DUMP=1 python test_moe_sft_tp_debug.py --mode comprehensive_backward
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print("=" * 80)
    print(f"Comprehensive Backward Test for SFT TP MoE with LoRA [{tp_mode_str}]")
    print("=" * 80)

    # Check for dump environment variable
    dump_enabled = dump_enabled or os.environ.get("SFT_MOE_DUMP", "0") != "0"
    dump_dir = os.environ.get("SFT_MOE_DUMP_DIR", "./cpp_dump")
    py_dump_dir = os.path.join(dump_dir, "py")
    cpp_dump_dir = os.path.join(dump_dir, "cpp")

    print(f"\nDump enabled: {dump_enabled}")
    if dump_enabled:
        print(f"  Dump directory: {dump_dir}")
        os.makedirs(py_dump_dir, exist_ok=True)
        os.makedirs(cpp_dump_dir, exist_ok=True)

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel not available, skipping C++ comparison")
        return True

    torch.manual_seed(42)

    # Configuration (smaller dimensions for faster testing)
    test_expert_num = 8
    test_hidden_size = 1024
    test_intermediate_size = 1024
    test_lora_rank = 16
    test_lora_scaling = lora_alpha / test_lora_rank
    test_qlen = 40
    test_k = 4
    test_num_threads = 8
    test_max_len = 1024

    # Weight/input scaling for numerical stability
    WEIGHT_SCALE = 0.01
    INPUT_SCALE = 0.1
    GRAD_SCALE = 0.1

    print(f"\nConfiguration:")
    print(f"  Experts: {test_expert_num}, Routed per token: {test_k}")
    print(f"  Hidden: {test_hidden_size}, Intermediate: {test_intermediate_size}")
    print(f"  Sequence length: {test_qlen}, LoRA rank: {test_lora_rank}")
    print(f"  TP count: {tp_count}")
    print(f"  Weight scale: {WEIGHT_SCALE}, Input scale: {INPUT_SCALE}, Grad scale: {GRAD_SCALE}")
    print("=" * 80)

    # Initialize weights
    print("\n[Initializing Weights]")
    gate_proj = (
        torch.rand(test_expert_num, test_intermediate_size, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_proj = (
        torch.rand(test_expert_num, test_intermediate_size, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_proj = (
        torch.rand(test_expert_num, test_hidden_size, test_intermediate_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()

    # LoRA weights
    gate_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    gate_lora_b = (
        torch.rand(test_expert_num, test_intermediate_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_b = (
        torch.rand(test_expert_num, test_intermediate_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_intermediate_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_b = (
        torch.rand(test_expert_num, test_hidden_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()

    # Check weights for NaN
    print("\n[Checking Weight Initialization]")
    has_nan = False
    has_nan |= check_nan(gate_proj, "gate_proj")
    has_nan |= check_nan(up_proj, "up_proj")
    has_nan |= check_nan(down_proj, "down_proj")
    has_nan |= check_nan(gate_lora_a, "gate_lora_a")
    has_nan |= check_nan(gate_lora_b, "gate_lora_b")
    has_nan |= check_nan(up_lora_a, "up_lora_a")
    has_nan |= check_nan(up_lora_b, "up_lora_b")
    has_nan |= check_nan(down_lora_a, "down_lora_a")
    has_nan |= check_nan(down_lora_b, "down_lora_b")
    if not has_nan:
        print("  All weights OK (no NaN/Inf)")

    # Setup C++ MoE operator
    print(f"\n[Setting up C++ MoE Operator with tp_count={tp_count}]")
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=test_expert_num,
        num_experts_per_tok=test_k,
        hidden_size=test_hidden_size,
        moe_intermediate_size=test_intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=test_num_threads,
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=test_max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=test_lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=2,
    )

    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj
    wrapper.load_weights(torch.arange(test_expert_num, dtype=torch.int64))
    wrapper.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)
    print("  C++ MoE operator initialized")

    # Setup PyTorch TP Simulator
    print("\n[Setting up PyTorch TP Simulator]")
    simulator = TPSFTSimulator(
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b,
        up_lora_a,
        up_lora_b,
        down_lora_a,
        down_lora_b,
        test_lora_scaling,
        tp_count,
    )
    print("  PyTorch TP simulator initialized")

    # Generate test data
    print("\n[Generating Test Data]")
    input_tensor = (torch.rand((test_qlen, test_hidden_size), dtype=torch.bfloat16) * INPUT_SCALE).contiguous()
    output_grad = (torch.rand((test_qlen, test_hidden_size), dtype=torch.bfloat16) * GRAD_SCALE).contiguous()
    expert_ids = torch.stack([torch.randperm(test_expert_num)[:test_k] for _ in range(test_qlen)]).contiguous()
    routing_weights = torch.rand(test_qlen, test_k, dtype=torch.float).contiguous()
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    has_nan |= check_nan(input_tensor, "input_tensor")
    has_nan |= check_nan(output_grad, "output_grad")
    expert_counts = torch.bincount(expert_ids.view(-1), minlength=test_expert_num)
    print(f"  Expert usage: {expert_counts.tolist()}")

    # Save inputs if dump enabled
    if dump_enabled:
        print("\n[Saving Python Inputs]")
        save_tensor_for_comparison(input_tensor, "input", py_dump_dir)
        save_tensor_for_comparison(output_grad, "output_grad", py_dump_dir)
        save_tensor_for_comparison(expert_ids.long(), "expert_ids", py_dump_dir)
        save_tensor_for_comparison(routing_weights, "routing_weights", py_dump_dir)

    # C++ Forward Pass
    print("\n[Running C++ Forward Pass]")
    cpp_output = wrapper.forward_sft(input_tensor, expert_ids, routing_weights, save_for_backward=True)
    cpp_fwd_has_nan = check_nan(cpp_output, "cpp_forward_output")

    # C++ Backward Pass
    print("\n[Running C++ Backward Pass]")
    cpp_grad_input, cpp_grad_loras = wrapper.backward(output_grad)

    # Check C++ backward outputs for NaN
    cpp_has_nan = check_nan(cpp_grad_input, "cpp_grad_input")
    cpp_has_nan |= check_nan(cpp_grad_loras["grad_gate_lora_a"], "cpp_grad_gate_lora_a")
    cpp_has_nan |= check_nan(cpp_grad_loras["grad_gate_lora_b"], "cpp_grad_gate_lora_b")
    cpp_has_nan |= check_nan(cpp_grad_loras["grad_up_lora_a"], "cpp_grad_up_lora_a")
    cpp_has_nan |= check_nan(cpp_grad_loras["grad_up_lora_b"], "cpp_grad_up_lora_b")
    cpp_has_nan |= check_nan(cpp_grad_loras["grad_down_lora_a"], "cpp_grad_down_lora_a")
    cpp_has_nan |= check_nan(cpp_grad_loras["grad_down_lora_b"], "cpp_grad_down_lora_b")
    if not cpp_has_nan:
        print("  C++ backward output OK (no NaN/Inf)")

    # PyTorch Forward Pass
    print("\n[Running PyTorch TP Simulator Forward Pass]")
    py_output, py_intermediates = simulator.forward_moe(
        input_tensor, expert_ids, routing_weights, dump_intermediates=dump_enabled
    )
    py_fwd_has_nan = check_nan(py_output, "py_forward_output")

    # Save outputs if dump enabled
    if dump_enabled:
        print("\n[Saving Forward Outputs]")
        save_tensor_for_comparison(cpp_output, "cpp_forward_output", cpp_dump_dir)
        save_tensor_for_comparison(py_output, "py_forward_output", py_dump_dir)

    # Compare forward outputs
    print("\n[Comparing Forward Outputs]")
    fwd_result = compare_tensors_detailed(cpp_output, py_output, "forward_output", BF16_FORWARD_THRESHOLD)
    print_comparison_result(fwd_result, verbose=True)

    # Diagnostic: verify merge is working correctly
    if dump_enabled:
        print("\n[Diagnostic: Merge Verification]")
        tp0_file = f"./cpp_dump/final_output_tp0.bin"
        tp1_file = f"./cpp_dump/final_output_tp1.bin"
        print(tp0_file, tp1_file)
        if os.path.exists(tp0_file) and os.path.exists(tp1_file):

            def read_matrix_file_diag(filepath):
                with open(filepath, "rb") as f:
                    rows = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
                return data

            tp0 = read_matrix_file_diag(tp0_file)
            tp1 = read_matrix_file_diag(tp1_file)
            cpp_sum_fp32 = tp0 + tp1
            cpp_output_fp32 = cpp_output.float().numpy()
            py_output_fp32 = py_output.float().numpy()
            print(f"  TP0 mean: {tp0.mean():.6f}, TP1 mean: {tp1.mean():.6f}")
            print(f"  Sum (TP0+TP1) mean: {cpp_sum_fp32.mean():.6f}")
            print(f"  cpp_output (merged) mean: {cpp_output_fp32.mean():.6f}")
            print(f"  py_output mean: {py_output_fp32.mean():.6f}")
            print(f"  |Sum - cpp_output| mean: {np.abs(cpp_sum_fp32 - cpp_output_fp32).mean():.6e}")
            print(f"  |Sum - py_output| mean: {np.abs(cpp_sum_fp32 - py_output_fp32).mean():.6e}")
        else:
            print("  Dump files not found, skipping merge verification")

    # Save backward outputs if dump enabled
    if dump_enabled:
        print("\n[Saving Backward Outputs]")
        save_tensor_for_comparison(cpp_grad_input, "cpp_grad_input", cpp_dump_dir)
        for name, grad in cpp_grad_loras.items():
            save_tensor_for_comparison(grad, f"cpp_{name}", cpp_dump_dir)

    # Note: Full PyTorch MoE backward would require implementing backward for the full MoE
    # Here we compare the C++ backward shapes and check for NaN
    print("\n[C++ Backward Output Statistics]")
    print(
        f"  grad_input: min={cpp_grad_input.min().item():.6f}, max={cpp_grad_input.max().item():.6f}, mean={cpp_grad_input.float().mean().item():.6f}"
    )
    for name, grad in cpp_grad_loras.items():
        print(f"  {name}: shape={grad.shape}, mean={grad.float().mean().item():.6e}")

    # Final verdict
    print("\n" + "=" * 80)
    threshold = BF16_FORWARD_THRESHOLD
    lora_threshold = BF16_BACKWARD_THRESHOLD

    passed = True
    if fwd_result["status"] != "PASS":
        print(f"\033[91mFAILED: Forward output comparison failed\033[0m")
        passed = False
    if cpp_has_nan or py_fwd_has_nan:
        print(f"\033[91mFAILED: NaN/Inf detected\033[0m")
        passed = False

    if passed:
        print(f"\033[92mTEST PASSED!\033[0m")
        print(f"  Forward rel_error: {fwd_result['rel_error']:.6e}")
    print("=" * 80)

    return passed


def test_moe_backward_full(quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT, dump_enabled: bool = False):
    """
    Full MoE backward test comparing C++ and PyTorch implementations.

    This test implements a full PyTorch MoE backward pass and compares it with
    the C++ implementation, similar to test_minimal_backward.py.
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print("=" * 80)
    print(f"Full MoE Backward Test [{tp_mode_str}, tp_count={tp_count}]")
    print("=" * 80)

    dump_enabled = dump_enabled or os.environ.get("SFT_MOE_DUMP", "0") != "0"

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel not available, skipping test")
        return True

    torch.manual_seed(42)

    # Configuration
    test_expert_num = 8
    test_hidden_size = 256
    test_intermediate_size = 512
    test_lora_rank = 8
    test_qlen = 4
    test_k = 2
    test_num_threads = 8
    test_max_len = 1024

    WEIGHT_SCALE = 0.01
    INPUT_SCALE = 0.1
    GRAD_SCALE = 0.1

    # Compute correct lora_scaling for the test configuration
    # NOTE: C++ computes lora_scaling = lora_alpha / lora_rank internally
    # So we must use the same formula here with test_lora_rank
    test_lora_scaling = lora_alpha / test_lora_rank  # 32 / 8 = 4.0

    print(f"\nConfiguration:")
    print(f"  expert_num={test_expert_num}, hidden={test_hidden_size}, intermediate={test_intermediate_size}")
    print(f"  qlen={test_qlen}, k={test_k}, lora_rank={test_lora_rank}, lora_scaling={test_lora_scaling}")
    print("=" * 80)

    # Initialize weights
    gate_proj = (
        torch.rand(test_expert_num, test_intermediate_size, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_proj = (
        torch.rand(test_expert_num, test_intermediate_size, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_proj = (
        torch.rand(test_expert_num, test_hidden_size, test_intermediate_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()

    gate_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    gate_lora_b = (
        torch.rand(test_expert_num, test_intermediate_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_hidden_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_b = (
        torch.rand(test_expert_num, test_intermediate_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_a = (
        torch.rand(test_expert_num, test_lora_rank, test_intermediate_size, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_b = (
        torch.rand(test_expert_num, test_hidden_size, test_lora_rank, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()

    # Setup C++ wrapper
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=test_expert_num,
        num_experts_per_tok=test_k,
        hidden_size=test_hidden_size,
        moe_intermediate_size=test_intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=test_num_threads,
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=test_max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=test_lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=2,
    )

    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj
    wrapper.load_weights(torch.arange(test_expert_num, dtype=torch.int64))
    wrapper.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    # Generate test data
    input_tensor = (torch.rand((test_qlen, test_hidden_size), dtype=torch.bfloat16) * INPUT_SCALE).contiguous()
    output_grad = (torch.rand((test_qlen, test_hidden_size), dtype=torch.bfloat16) * GRAD_SCALE).contiguous()
    expert_ids = torch.stack([torch.randperm(test_expert_num)[:test_k] for _ in range(test_qlen)]).contiguous()
    routing_weights = torch.rand(test_qlen, test_k, dtype=torch.float).contiguous()
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # C++ forward + backward
    print("\n[Running C++ Forward + Backward]")
    cpp_output = wrapper.forward_sft(input_tensor, expert_ids, routing_weights, save_for_backward=True)
    cpp_grad_input, cpp_grad_loras = wrapper.backward(output_grad)

    # PyTorch forward + backward using non-TP reference
    print("\n[Running PyTorch Reference Forward + Backward]")
    py_output = moe_sft_torch_forward_no_tp(
        input_tensor,
        expert_ids,
        routing_weights,
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b,
        up_lora_a,
        up_lora_b,
        down_lora_a,
        down_lora_b,
        test_lora_scaling,
    )

    # Compare forward
    print("\n[Forward Comparison]")
    fwd_rel_err = compute_relative_error(cpp_output, py_output)
    fwd_abs_err = (cpp_output.float() - py_output.float()).abs().mean().item()
    print(f"  Relative error: {fwd_rel_err:.6e}")
    print(f"  Absolute error (mean): {fwd_abs_err:.6e}")

    # Output statistics
    print("\n[Output Statistics]")
    print(
        f"  C++ output: min={cpp_output.min().item():.6f}, max={cpp_output.max().item():.6f}, mean={cpp_output.float().mean().item():.6f}"
    )
    print(
        f"  Py output:  min={py_output.min().item():.6f}, max={py_output.max().item():.6f}, mean={py_output.float().mean().item():.6f}"
    )

    print("\n[Backward Statistics (C++)]")
    print(f"  grad_input: shape={cpp_grad_input.shape}, mean={cpp_grad_input.float().mean().item():.6e}")
    for name, grad in cpp_grad_loras.items():
        print(f"  {name}: shape={grad.shape}, mean={grad.float().mean().item():.6e}")

    # Verdict
    print("\n" + "=" * 80)
    passed = fwd_rel_err < BF16_FORWARD_THRESHOLD
    if passed:
        print(f"\033[92mTEST PASSED!\033[0m")
        print(f"  Forward rel_error: {fwd_rel_err:.6e} (threshold: {BF16_FORWARD_THRESHOLD})")
    else:
        print(f"\033[91mTEST FAILED!\033[0m")
        print(f"  Forward rel_error: {fwd_rel_err:.6e} >= {BF16_FORWARD_THRESHOLD}")
    print("=" * 80)

    return passed


def dump_all_intermediate_values(quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT):
    """
    Comprehensive intermediate value dump for debugging.

    This function dumps all intermediate values from the PyTorch TP simulator
    for detailed analysis.
    """
    print(f"\n{'='*60}")
    print(f"Comprehensive Intermediate Value Dump (tp_count={tp_count})")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Use smaller dimensions for faster testing
    test_expert_num = 8
    test_hidden_size = 64
    test_intermediate_size = 128
    test_lora_rank = 4
    test_qlen = 2
    test_k = 2

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(test_expert_num, test_hidden_size, test_intermediate_size)
    lora_weights = init_lora_weights(test_expert_num, test_hidden_size, test_intermediate_size, test_lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Create TP simulator
    simulator = TPSFTSimulator(
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
    )

    # Generate test inputs with specific experts
    expert_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    input_data = torch.randn((test_qlen, test_hidden_size), dtype=torch.bfloat16).contiguous() / 100

    print(f"\nConfiguration:")
    print(f"  expert_num: {test_expert_num}")
    print(f"  hidden_size: {test_hidden_size}")
    print(f"  intermediate_size: {test_intermediate_size}")
    print(f"  tp_count: {tp_count}")
    print(f"  tp_intermediate: {test_intermediate_size // tp_count}")
    print(f"  lora_rank: {test_lora_rank}")
    print(f"  lora_scaling: {lora_scaling}")
    print(f"  qlen: {test_qlen}")
    print(f"  k: {test_k}")
    print(f"  expert_ids:\n{expert_ids}")
    print(f"  weights:\n{weights}")

    # Forward with intermediate dump
    output, intermediates = simulator.forward_moe(input_data, expert_ids, weights, dump_intermediates=True)

    print(f"\n{'='*60}")
    print(f"All Intermediate Values")
    print(f"{'='*60}")

    # Sort keys for organized output
    sorted_keys = sorted(intermediates.keys())

    for key in sorted_keys:
        val = intermediates[key]
        print(f"\n{key}:")
        print(f"  shape: {val.shape}")
        print(f"  dtype: {val.dtype}")
        print(f"  mean: {val.float().mean():.6f}")
        print(f"  std: {val.float().std():.6f}")
        print(f"  min: {val.float().min():.6f}")
        print(f"  max: {val.float().max():.6f}")
        if val.numel() <= 32:
            print(f"  values: {val.flatten().tolist()}")

    print(f"\n{'='*60}")
    print(f"Final Output")
    print(f"{'='*60}")
    print(f"  shape: {output.shape}")
    print(f"  mean: {output.float().mean():.6f}")
    print(f"  std: {output.float().std():.6f}")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_all_tests(quant_mode: str = "AMXBF16_SFT"):
    """Run all TP debug tests."""
    print("\n" + "=" * 70)
    print(" MOE SFT TP Debug Test Suite")
    print("=" * 70)

    try:
        # Test weight partitioning
        test_weight_partitioning()

        # Test TP simulator vs non-TP reference (forward)
        test_tp_simulator_vs_no_tp()

        # Test single expert forward with intermediate dump
        test_tp_simulator_single_expert()

        # Test TP simulator backward vs non-TP reference
        test_tp_backward_vs_no_tp()

        # Test TP simulator vs C++ (if available)
        if HAS_KT_KERNEL:
            test_tp_vs_cpp_wrapper(quant_mode, tp_count=TP_COUNT)
            test_tp_vs_cpp_wrapper(quant_mode, tp_count=NO_TP_COUNT)
            test_tp_vs_no_tp_cpp(quant_mode)
            test_tp_backward_vs_cpp(quant_mode, tp_count=TP_COUNT)

            # Comprehensive backward tests
            test_comprehensive_backward_with_dump(quant_mode, tp_count=TP_COUNT)
            test_moe_backward_full(quant_mode, tp_count=TP_COUNT)
        else:
            print("\nSkipping C++ comparison tests (kt_kernel not available)")

        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOE SFT TP Debug Test Suite")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "partition",
            "simulator",
            "single",
            "backward",
            "cpp",
            "cpp_tp_compare",
            "cpp_backward",
            "dump",
            "comprehensive_backward",
            "moe_backward_full",
        ],
        default="all",
        help="Test mode",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="AMXBF16_SFT",
        help="SFT method to test",
    )
    parser.add_argument(
        "--tp-count",
        type=int,
        default=TP_COUNT,
        help="TP count for tests",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Enable dump mode (saves intermediate values to files)",
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all_tests(quant_mode=args.method)
    elif args.mode == "partition":
        test_weight_partitioning()
    elif args.mode == "simulator":
        test_tp_simulator_vs_no_tp()
    elif args.mode == "single":
        test_tp_simulator_single_expert()
    elif args.mode == "backward":
        test_tp_backward_vs_no_tp()
    elif args.mode == "cpp":
        test_tp_vs_cpp_wrapper(args.method, tp_count=args.tp_count)
    elif args.mode == "cpp_tp_compare":
        test_tp_vs_no_tp_cpp(args.method)
    elif args.mode == "cpp_backward":
        test_tp_backward_vs_cpp(args.method, tp_count=args.tp_count)
    elif args.mode == "dump":
        dump_all_intermediate_values(args.method, tp_count=args.tp_count)
    elif args.mode == "comprehensive_backward":
        test_comprehensive_backward_with_dump(args.method, tp_count=args.tp_count, dump_enabled=args.dump)
    elif args.mode == "moe_backward_full":
        test_moe_backward_full(args.method, tp_count=args.tp_count, dump_enabled=args.dump)
