"""
Weight loaders for different formats.

This module provides loaders for:
- SafeTensor format (for AMX quantized weights)
- GGUF format (for Llamafile quantized weights)
"""

from __future__ import annotations

import os
import numpy as np
import torch
from enum import IntEnum
from safetensors import safe_open
from gguf.gguf_reader import GGUFReader


class GGMLQuantizationType(IntEnum):
    """GGML quantization type enumeration"""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


def translate_name_to_gguf(name):
    """
    Translate PyTorch tensor name to GGUF format
    """
    name = name.replace("lm_head.", "output.")
    name = name.replace("model.embed_tokens.", "token_embd.")
    name = name.replace("model.norm.", "output_norm.")
    name = name.replace("model.layers.", "blk.")
    name = name.replace(".input_layernorm", ".attn_norm")
    name = name.replace(".mlp.down_proj", ".ffn_down")
    name = name.replace(".mlp.gate_proj", ".ffn_gate")
    name = name.replace(".mlp.up_proj", ".ffn_up")
    name = name.replace(".post_attention_layernorm", ".ffn_norm")
    name = name.replace(".self_attn.q_proj", ".attn_q")
    name = name.replace(".self_attn.k_proj", ".attn_k")
    name = name.replace(".self_attn.v_proj", ".attn_v")
    name = name.replace(".self_attn.o_proj", ".attn_output")
    name = name.replace(".self_attn.qkv_proj", ".attn_qkv")
    name = name.replace(".self_attn.kv_a_proj_with_mqa", ".attn_kv_a_mqa")
    name = name.replace(".self_attn.kv_a_layernorm", ".attn_kv_a_norm")
    name = name.replace(".self_attn.kv_b_proj", ".attn_kv_b")
    name = name.replace(".self_attn.q_a_proj", ".attn_q_a")
    name = name.replace(".self_attn.q_a_layernorm", ".attn_q_a_norm")
    name = name.replace(".self_attn.q_b_proj", ".attn_q_b")
    name = name.replace(".self_attn.q_norm", ".attn_q_norm")
    name = name.replace(".self_attn.k_norm", ".attn_k_norm")
    name = name.replace(".shared_expert.", ".shared_experts.")
    name = name.replace(".shared_expert_", ".shared_experts_")
    name = name.replace(".gate_up_proj.", ".up_proj")
    name = name.replace(".mlp.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".mlp.gate.e_score_correction_bias", ".exp_probs_b.bias")
    name = name.replace(".mlp.gate", ".ffn_gate_inp")
    name = name.replace(".mlp.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".mlp.shared_experts.up_proj", ".ffn_up_shexp")
    name = name.replace(".mlp.shared_experts_gate", ".ffn_gate_inp_shexp")
    name = name.replace(".mlp.experts", "")
    name = name.replace(".mlp.experts.ffn_down_exps", ".ffn_down_exps")
    name = name.replace(".mlp.experts.ffn_gate_exps", ".ffn_gate_exps")
    name = name.replace(".mlp.experts.ffn_up_exps", ".ffn_up_exps")
    name = name.replace(".block_sparse_moe.gate.", ".ffn_gate_inp.")
    name = name.replace(".block_sparse_moe.experts", "")
    name = name.replace(".feed_forward.experts", "")
    name = name.replace(".feed_forward.router", ".ffn_gate_inp")
    name = name.replace(".feed_forward.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".feed_forward.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".feed_forward.shared_experts.up_proj", ".ffn_up_shexp")
    return name


class SafeTensorLoader:
    """
    SafeTensor format loader for AMX quantized weights.

    Supports loading tensors from .safetensors files with NUMA-sharded expert weights.
    """

    tensor_file_map: dict
    tensor_type_map: dict
    file_handle_map: dict
    tensor_device_map: dict

    def __init__(self, file_path: str):
        self.__load_tensor_file_map(file_path)

    def __load_tensor_file_map(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")
        if os.path.isfile(file_path):
            folder_path = os.path.dirname(file_path)
        else:
            folder_path = file_path
        self.file_handle_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework="pt")
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f"Error opening Safetensor file {file_path}: {e}")
                            continue

                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f"Error reading Safetensor file {file_path}: {e}")

        if not found_safetensor:
            raise FileNotFoundError(f"No Safetensor files found in {folder_path}")

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)

    def close_all_handles(self):
        for handle in self.file_handle_map.values():
            handle.close()
        self.file_handle_map.clear()

    def load_experts(self, base_key: str, device: str = "cpu"):
        """
        Load expert weights from SafeTensor files.

        Expected format:
        - blk.{layer_index}.ffn_[up, down, gate]_exps.{expert_id}.numa.{numa_id}.weight
        - blk.{layer_index}.ffn_[up, down, gate]_exps.{expert_id}.numa.{numa_id}.scale

        Args:
            base_key: Base key like "blk.{layer_index}"
            device: Target device for tensors

        Returns:
            Dictionary with keys: up, gate, down, up_scale, gate_scale, down_scale
            Each value is a list of lists: [numa_id][expert_id] -> numpy array
        """
        up_base_key = f"{base_key}.ffn_up_exps"
        gate_base_key = f"{base_key}.ffn_gate_exps"
        down_base_key = f"{base_key}.ffn_down_exps"
        max_numa_id = -1
        max_experts_count = -1
        while self.has_tensor(f"{up_base_key}.{max_experts_count+1}.numa.{0}.weight"):
            max_experts_count += 1
        if max_experts_count == 0:
            raise ValueError(f"No experts found for key {base_key}")
        while self.has_tensor(f"{up_base_key}.{0}.numa.{max_numa_id+1}.weight"):
            max_numa_id += 1
        # Initialize empty lists to store tensors for each projection type
        up_weights = [[] for _ in range(max_numa_id + 1)]
        gate_weights = [[] for _ in range(max_numa_id + 1)]
        down_weights = [[] for _ in range(max_numa_id + 1)]
        up_scales = [[] for _ in range(max_numa_id + 1)]
        gate_scales = [[] for _ in range(max_numa_id + 1)]
        down_scales = [[] for _ in range(max_numa_id + 1)]
        for numa_id in range(max_numa_id + 1):
            for expert_id in range(max_experts_count + 1):
                up_key = f"{up_base_key}.{expert_id}.numa.{numa_id}.weight"
                gate_key = f"{gate_base_key}.{expert_id}.numa.{numa_id}.weight"
                down_key = f"{down_base_key}.{expert_id}.numa.{numa_id}.weight"
                up_scale_key = f"{up_base_key}.{expert_id}.numa.{numa_id}.scale"
                gate_scale_key = f"{gate_base_key}.{expert_id}.numa.{numa_id}.scale"
                down_scale_key = f"{down_base_key}.{expert_id}.numa.{numa_id}.scale"
                # make sure contiguous
                up_tensor = self.load_tensor(up_key, device).numpy()
                gate_tensor = self.load_tensor(gate_key, device).numpy()
                down_tensor = self.load_tensor(down_key, device).numpy()
                up_scale_tensor = self.load_tensor(up_scale_key, device).numpy()
                gate_scale_tensor = self.load_tensor(gate_scale_key, device).numpy()
                down_scale_tensor = self.load_tensor(down_scale_key, device).numpy()

                up_weights[numa_id].append(up_tensor)
                gate_weights[numa_id].append(gate_tensor)
                down_weights[numa_id].append(down_tensor)
                up_scales[numa_id].append(up_scale_tensor)
                gate_scales[numa_id].append(gate_scale_tensor)
                down_scales[numa_id].append(down_scale_tensor)
        return {
            "up": up_weights,
            "gate": gate_weights,
            "down": down_weights,
            "up_scale": up_scales,
            "gate_scale": gate_scales,
            "down_scale": down_scales,
        }

    def has_tensor(self, name: str):
        return name in self.tensor_file_map


class FP8SafeTensorLoader(SafeTensorLoader):
    """Loader for FP8 expert weights with auto-detection of naming formats.

    Supported formats:
    - DeepSeek style: {base}.mlp.experts.{id}.{gate,up,down}_proj.weight
    - Mixtral/MiniMax style: {base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight

    Supported scale formats (auto-detected):
    - Block-wise: weight_scale_inv (DeepSeek FP8)
    - Per-channel: weight_scale (GLM-4.7-FP8)

    The format is auto-detected during initialization.
    """

    # Known MoE naming formats: (experts_path_template, gate_name, up_name, down_name)
    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str, scale_suffix: str = None):
        """Initialize FP8 loader with optional scale suffix override.

        Args:
            file_path: Path to safetensor files
            scale_suffix: Optional scale key suffix. If None, auto-detect between
                         'weight_scale_inv' (block-wise) and 'weight_scale' (per-channel).
        """
        super().__init__(file_path)
        self._detected_format = None
        self._scale_suffix = scale_suffix  # None means auto-detect
        # Set per_channel based on explicit scale_suffix if provided
        if scale_suffix == "weight_scale":
            self._is_per_channel = True
        elif scale_suffix == "weight_scale_inv":
            self._is_per_channel = False
        else:
            self._is_per_channel = False  # Will be updated in _detect_format if auto-detect
        self._is_vl_model = False
        self._detect_format()

    def _detect_format(self):
        """Auto-detect the MoE naming format and scale format by checking tensor keys."""
        # Sample some tensor names to detect format
        sample_keys = list(self.tensor_file_map.keys())[:1000]

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            # Check if any key matches this format pattern
            # Look for pattern like: model.layers.0.{experts_path}.0.{gate_name}.weight
            for key in sample_keys:
                if ".experts." in key and f".{gate}.weight" in key:
                    # Verify the path template matches
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
            if self._detected_format:
                break

        # Default to deepseek if no format detected
        if not self._detected_format:
            self._detected_format = "deepseek"
            print("[FP8SafeTensorLoader] No MoE format detected, defaulting to: deepseek")

        # Auto-detect scale suffix if not specified
        if self._scale_suffix is None:
            _, gate, _, _ = self.MOE_FORMATS[self._detected_format]
            # Check for per-channel scale (weight_scale) vs block-wise (weight_scale_inv)
            for key in sample_keys:
                if f".{gate}.weight_scale_inv" in key:
                    self._scale_suffix = "weight_scale_inv"
                    self._is_per_channel = False
                    print("[FP8SafeTensorLoader] Detected scale format: block-wise (weight_scale_inv)")
                    if key.startswith("model.language_model.") and self._detected_format == "deepseek":
                        # VL models(Qwen3.5): model.layers.{N} -> model.language_model.layers.{N}
                        self._is_vl_model = True
                        print("[FP8SafeTensorLoader] Detected VL model")
                        return
                elif f".{gate}.weight_scale" in key and "weight_scale_inv" not in key:
                    self._scale_suffix = "weight_scale"
                    self._is_per_channel = True
                    print("[FP8SafeTensorLoader] Detected scale format: per-channel (weight_scale)")
                    return
            # Default to weight_scale_inv
            self._scale_suffix = "weight_scale_inv"
            self._is_per_channel = False
            print("[FP8SafeTensorLoader] No scale format detected, defaulting to: weight_scale_inv")
        else:
            # Scale suffix was explicitly provided
            scale_type = "per-channel" if self._is_per_channel else "block-wise"
            print(f"[FP8SafeTensorLoader] Using explicit scale format: {scale_type} ({self._scale_suffix})")

    def _get_experts_prefix(self, base_key: str) -> str:
        """Get the experts prefix based on detected format."""
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        if self._is_vl_model:
            base_key = base_key.replace("model.layers", "model.language_model.layers")
        return path_tpl.format(base=base_key)

    def _get_proj_names(self):
        """Get projection names (gate, up, down) based on detected format."""
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        if device == "cpu":
            return tensor
        return tensor.to(device)

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load FP8 expert weights and their scale tensors.

        Supports both block-wise (weight_scale_inv) and per-channel (weight_scale) formats.
        Per-channel scales are squeezed from [N, 1] to [N] if needed.
        """
        experts_prefix = self._get_experts_prefix(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        while self.has_tensor(f"{experts_prefix}.{expert_count}.{gate_name}.weight"):
            expert_count += 1

        if expert_count == 0:
            raise ValueError(f"No experts found for key {experts_prefix}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            gate_w_key = f"{experts_prefix}.{exp_id}.{gate_name}.weight"
            up_w_key = f"{experts_prefix}.{exp_id}.{up_name}.weight"
            down_w_key = f"{experts_prefix}.{exp_id}.{down_name}.weight"
            gate_s_key = f"{experts_prefix}.{exp_id}.{gate_name}.{self._scale_suffix}"
            up_s_key = f"{experts_prefix}.{exp_id}.{up_name}.{self._scale_suffix}"
            down_s_key = f"{experts_prefix}.{exp_id}.{down_name}.{self._scale_suffix}"

            gate_weights[exp_id] = self.load_tensor(gate_w_key, device).contiguous()
            up_weights[exp_id] = self.load_tensor(up_w_key, device).contiguous()
            down_weights[exp_id] = self.load_tensor(down_w_key, device).contiguous()

            gate_scale = self.load_tensor(gate_s_key, device)
            up_scale = self.load_tensor(up_s_key, device)
            down_scale = self.load_tensor(down_s_key, device)

            # For per-channel scales, squeeze [N, 1] -> [N] if needed
            if self._is_per_channel:
                if gate_scale.dim() == 2 and gate_scale.shape[1] == 1:
                    gate_scale = gate_scale.squeeze(1)
                if up_scale.dim() == 2 and up_scale.shape[1] == 1:
                    up_scale = up_scale.squeeze(1)
                if down_scale.dim() == 2 and down_scale.shape[1] == 1:
                    down_scale = down_scale.squeeze(1)

            gate_scales[exp_id] = gate_scale.contiguous()
            up_scales[exp_id] = up_scale.contiguous()
            down_scales[exp_id] = down_scale.contiguous()

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }

    def is_per_channel(self) -> bool:
        """Return True if using per-channel quantization, False for block-wise."""
        return self._is_per_channel



class BF16SafeTensorLoader(SafeTensorLoader):
    """Loader for native BF16 expert weights (no quantization, no scales).

    Supported formats:
    - DeepSeek style: {base}.mlp.experts.{id}.{gate,up,down}_proj.weight
    - Mixtral/MiniMax style: {base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight

    The format is auto-detected during initialization.
    """

    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._detected_format = None
        self._detect_format()

    def _detect_format(self):
        """Auto-detect the MoE naming format by checking tensor keys."""
        sample_keys = list(self.tensor_file_map.keys())[:1000]

        # Check for packed format first (Qwen3.5 MoE style: all experts in one 3D tensor)
        for key in sample_keys:
            if key.endswith(".mlp.experts.gate_up_proj"):
                self._detected_format = "packed"
                print("[BF16SafeTensorLoader] Detected format: packed (Qwen3.5 MoE style)")
                return

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            for key in sample_keys:
                if ".experts." in key and f".{gate}.weight" in key:
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return

        self._detected_format = "deepseek"
        print("[BF16SafeTensorLoader] No MoE format detected, defaulting to: deepseek")

    def _get_experts_prefix(self, base_key: str) -> str:
        """Get the experts prefix based on detected format."""
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        return path_tpl.format(base=base_key)

    def _get_proj_names(self):
        """Get projection names (gate, up, down) based on detected format."""
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        if device == "cpu":
            return tensor
        return tensor.to(device)

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load BF16 expert weights (no scales needed)."""
        if self._detected_format == "packed":
            return self._load_experts_packed(base_key, device)

        experts_prefix = self._get_experts_prefix(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        while self.has_tensor(f"{experts_prefix}.{expert_count}.{gate_name}.weight"):
            expert_count += 1

        if expert_count == 0:
            raise ValueError(f"No experts found for key {experts_prefix}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count

        for exp_id in range(expert_count):
            gate_w_key = f"{experts_prefix}.{exp_id}.{gate_name}.weight"
            up_w_key = f"{experts_prefix}.{exp_id}.{up_name}.weight"
            down_w_key = f"{experts_prefix}.{exp_id}.{down_name}.weight"

            gate_weights[exp_id] = self.load_tensor(gate_w_key, device).contiguous()
            up_weights[exp_id] = self.load_tensor(up_w_key, device).contiguous()
            down_weights[exp_id] = self.load_tensor(down_w_key, device).contiguous()

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
        }

    def _resolve_packed_experts_prefix(self, base_key: str) -> str:
        """Resolve the experts prefix for packed format, trying fallbacks."""
        # Direct: model.layers.{N}.mlp.experts
        experts_prefix = f"{base_key}.mlp.experts"
        if self.has_tensor(f"{experts_prefix}.gate_up_proj"):
            return experts_prefix

        # VL models: model.layers.{N} -> model.language_model.layers.{N}
        parts = base_key.split(".", 1)
        if len(parts) == 2:
            alt_base = f"{parts[0]}.language_model.{parts[1]}"
            experts_prefix = f"{alt_base}.mlp.experts"
            if self.has_tensor(f"{experts_prefix}.gate_up_proj"):
                return experts_prefix

        raise ValueError(f"No packed experts found for base_key '{base_key}'.")

    def _load_experts_packed(self, base_key: str, device: str = "cpu"):
        """Load packed expert weights (Qwen3.5 MoE style).

        Packed format stores all experts in stacked 3D tensors:
        - gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
        - down_proj:    [num_experts, hidden_size, intermediate_size]
        """
        experts_prefix = self._resolve_packed_experts_prefix(base_key)

        gate_up_key = f"{experts_prefix}.gate_up_proj"
        down_key = f"{experts_prefix}.down_proj"

        gate_up = self.load_tensor(gate_up_key, device)  # [E, 2*I, H]
        down = self.load_tensor(down_key, device)  # [E, H, I]

        mid = gate_up.shape[1] // 2
        gate_list = [gate_up[i, :mid, :].contiguous() for i in range(gate_up.shape[0])]
        up_list = [gate_up[i, mid:, :].contiguous() for i in range(gate_up.shape[0])]
        down_list = [down[i].contiguous() for i in range(down.shape[0])]

        return {
            "gate": gate_list,
            "up": up_list,
            "down": down_list,
        }


class CompressedSafeTensorLoader(SafeTensorLoader):
    """Loader for compressed SafeTensor layouts (RAWINT4 weights)."""

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load raw expert weights stored in compressed safetensor format."""

        experts_prefix = f"{base_key}.mlp.experts"

        expert_idx = 0
        while self.has_tensor(f"{experts_prefix}.{expert_idx}.up_proj.weight_packed"):
            expert_idx += 1

        if expert_idx == 0:
            experts_prefix = f"language_model.{base_key}.mlp.experts"
            expert_idx = 0
            while self.has_tensor(f"{experts_prefix}.{expert_idx}.up_proj.weight_packed"):
                expert_idx += 1
            if expert_idx == 0:
                raise ValueError(f"No experts found for key {experts_prefix}")

        def load_projection(proj_name: str):
            weight_entries = []
            scale_entries = []

            for exp_id in range(expert_idx):
                weight_key = f"{experts_prefix}.{exp_id}.{proj_name}_proj.weight_packed"
                scale_key = f"{experts_prefix}.{exp_id}.{proj_name}_proj.weight_scale"

                if not self.has_tensor(weight_key):
                    raise KeyError(f"Missing tensor: {weight_key}")
                if not self.has_tensor(scale_key):
                    raise KeyError(f"Missing tensor: {scale_key}")

                weight_tensor = self.load_tensor(weight_key, device).contiguous()
                scale_tensor = self.load_tensor(scale_key, device).contiguous()

                weight_entries.append(weight_tensor)
                scale_entries.append(scale_tensor)

            return weight_entries, scale_entries

        gate_weights, gate_scales = load_projection("gate")
        up_weights, up_scales = load_projection("up")
        down_weights, down_scales = load_projection("down")

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class GGUFLoader:
    """
    GGUF format loader using the official gguf library (gguf.gguf_reader.GGUFReader)

    This is a cleaner implementation compared to manual binary parsing.
    """

    def __init__(self, gguf_path: str):
        """
        Initialize GGUF loader from a file or directory

        Args:
            gguf_path: Path to a single GGUF file or a directory containing GGUF files
        """
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF path not found: {gguf_path}")

        self.tensor_info = {}
        self.metadata = {}
        self.tensor_file_map = {}
        self.file_data_map = {}

        if os.path.isfile(gguf_path) and gguf_path.endswith(".gguf"):
            print(f"\n[GGUFLoader] Loading single GGUF file : {os.path.basename(gguf_path)}")
            self._load_single_file(gguf_path)
        elif os.path.isdir(gguf_path):
            print(f"\n[GGUFLoader] Loading GGUF files from directory: {gguf_path}")
            self._load_directory(gguf_path)
        else:
            raise ValueError(f"Path must be a .gguf file or a directory: {gguf_path}")

        print(f"[GGUFLoader] Summary:")
        print(f"  Files loaded: {len(self.file_data_map)}")
        print(f"  Total tensors: {len(self.tensor_info)}")
        print(f"  Metadata keys: {len(self.metadata)}")
        tensors = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"]
        for key in tensors:
            if key in self.tensor_info:
                info = self.tensor_info[key]
                print(f" {'.'.join(key.split('.')[2:-1])}, Dtype: {info['dtype'].name}")

    def _load_single_file(self, file_path: str):
        """Load a single GGUF file"""
        reader = GGUFReader(file_path)

        for key, field in reader.fields.items():
            value = field.parts[field.data[0]]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
                try:
                    value = bytes(value).decode("utf-8")
                except:
                    pass
            self.metadata[key] = value

        for tensor in reader.tensors:
            self.tensor_info[tensor.name] = {
                "shape": list(reversed(tensor.shape)),  # Reverse to match PyTorch order
                "dtype": tensor.tensor_type,
                "offset": tensor.data_offset,
                "n_elements": tensor.n_elements,
            }
            self.tensor_file_map[tensor.name] = file_path

        self.file_data_map[file_path] = np.memmap(file_path, mode="r")

    def _load_directory(self, dir_path: str):
        """Load all GGUF files from a directory (non-recursive)"""
        found_gguf = False

        for file in sorted(os.listdir(dir_path)):
            if file.endswith(".gguf"):
                found_gguf = True
                file_path = os.path.join(dir_path, file)
                print(f"  Loading: {file}")

                reader = GGUFReader(file_path)

                for key, field in reader.fields.items():
                    value = field.parts[field.data[0]]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
                        try:
                            value = bytes(value).decode("utf-8")
                        except:
                            pass
                    self.metadata[key] = value

                for tensor in reader.tensors:
                    self.tensor_info[tensor.name] = {
                        "shape": list(reversed(tensor.shape)),
                        "dtype": tensor.tensor_type,
                        "offset": tensor.data_offset,
                        "n_elements": tensor.n_elements,
                    }
                    self.tensor_file_map[tensor.name] = file_path

                self.file_data_map[file_path] = np.memmap(file_path, mode="r")

        if not found_gguf:
            raise FileNotFoundError(f"No .gguf files found in directory: {dir_path}")

    def get_model_config(self, layer_idx: int = 0):
        """
        Extract model configuration from GGUF metadata and tensor shapes.

        Args:
            layer_idx: Layer index to inspect (default: 0)

        Returns:
            dict with keys: num_experts, num_experts_per_tok, hidden_size, moe_intermediate_size
        """
        config = {}

        arch = self.metadata.get("general.architecture", "unknown")

        num_experts = None
        for key_suffix in [
            "expert_count",
            "expert.count",
            "moe.expert_count",
            "expert_feed_forward_length",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                num_experts = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        num_experts_per_tok = None
        for key_suffix in [
            "expert_used_count",
            "expert.used_count",
            "moe.num_experts_per_tok",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                num_experts_per_tok = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        hidden_size = None
        for key_suffix in [
            "embedding_length",
            "embed_length",
            "hidden_size",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                hidden_size = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        moe_intermediate_size = None
        for key_suffix in [
            "expert_feed_forward_length",
            "feed_forward_length",
            "ffn_length",
            "intermediate_size",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                moe_intermediate_size = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        if any(v is None for v in [num_experts, hidden_size, moe_intermediate_size]):

            base_key = f"blk.{layer_idx}.ffn_gate_exps.weight"
            if base_key in self.tensor_info:
                gate_shape = self.tensor_info[base_key]["shape"]
                print(f"  Found tensor '{base_key}' with shape: {gate_shape}")

                if len(gate_shape) >= 3:
                    if num_experts is None:
                        num_experts = int(gate_shape[0])
                    if moe_intermediate_size is None:
                        moe_intermediate_size = int(gate_shape[1])
                    if hidden_size is None:
                        hidden_size = int(gate_shape[2])

        config = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "hidden_size": hidden_size,
            "moe_intermediate_size": moe_intermediate_size,
        }

        return config

    def print_metadata(self, filter_keywords=None):
        """
        Print GGUF file metadata for debugging.

        Args:
            filter_keywords: Optional list of keywords to filter metadata keys
        """
        print(f"\n[GGUFLoader] GGUF Metadata:")
        print(f"  Total metadata entries: {len(self.metadata)}")

        if filter_keywords:
            filtered = {
                k: v for k, v in self.metadata.items() if any(kw.lower() in k.lower() for kw in filter_keywords)
            }
            for k, v in sorted(filtered.items()):
                print(f"  {k}: {v}")
        else:
            for k, v in sorted(self.metadata.items()):
                print(f"  {k}: {v}")

    def has_tensor(self, name: str):
        """Check if tensor exists"""
        name = translate_name_to_gguf(name)
        return name in self.tensor_info

    def get_ggml_type(self, name: str):
        """Get GGML type of a tensor"""
        name = translate_name_to_gguf(name)
        if name not in self.tensor_info:
            raise KeyError(f"Tensor '{name}' not found in GGUF files")
        return self.tensor_info[name]["dtype"]

    def get_undequanted_tensor_and_ggml_type(self, name: str):
        """
        Get tensor data and its GGML type without dequantizing

        Args:
            name: Tensor name (in PyTorch format, will be translated to GGUF format)

        Returns:
            (data, ggml_type): Tuple of tensor data and GGML quantization type
        """
        name = translate_name_to_gguf(name)

        if name not in self.tensor_info:
            raise KeyError(f"Tensor '{name}' not found in GGUF files")

        info = self.tensor_info[name]
        file_path = self.tensor_file_map[name]
        mmap_data = self.file_data_map[file_path]

        offset = info["offset"]
        n_elements = info["n_elements"]
        ggml_type = info["dtype"]

        GGML_QUANT_SIZES = {
            GGMLQuantizationType.F32: (1, 4),
            GGMLQuantizationType.F16: (1, 2),
            GGMLQuantizationType.BF16: (1, 2),
            GGMLQuantizationType.Q4_0: (32, 2 + 16),
            GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
            GGMLQuantizationType.Q5_0: (32, 2 + 4 + 16),
            GGMLQuantizationType.Q5_1: (32, 2 + 2 + 4 + 16),
            GGMLQuantizationType.Q8_0: (32, 2 + 32),
            GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32),
            GGMLQuantizationType.Q2_K: (256, 2 + 2 + 256 // 16 + 256 // 4),
            GGMLQuantizationType.Q3_K: (256, 2 + 256 // 4 + 256 // 8 + 12),
            GGMLQuantizationType.Q4_K: (256, 2 + 2 + 256 // 2 + 12),
            GGMLQuantizationType.Q5_K: (256, 2 + 2 + 256 // 2 + 256 // 8 + 12),
            GGMLQuantizationType.Q6_K: (256, 2 + 256 // 2 + 256 // 4 + 256 // 16),
            GGMLQuantizationType.Q8_K: (256, 4 + 256 + 256 // 8),
            GGMLQuantizationType.IQ2_XXS: (256, 2 + 256 // 4),
            GGMLQuantizationType.IQ2_XS: (256, 2 + 256 // 4 + 256 // 32),
            GGMLQuantizationType.IQ3_XXS: (256, 2 + 256 // 4 + 256 // 8),
            GGMLQuantizationType.IQ1_S: (256, 2 + 256 // 8 + 256 // 16),
            GGMLQuantizationType.IQ4_NL: (32, 2 + 16),
            GGMLQuantizationType.IQ3_S: (256, 2 + 256 // 4 + 256 // 8 + 256 // 32 + 4),
            GGMLQuantizationType.IQ2_S: (256, 2 + 256 // 4 + 256 // 16),
            GGMLQuantizationType.IQ4_XS: (256, 2 + 2 + 256 // 2 + 256 // 64),
            GGMLQuantizationType.I8: (1, 1),
            GGMLQuantizationType.I16: (1, 2),
            GGMLQuantizationType.I32: (1, 4),
            GGMLQuantizationType.I64: (1, 8),
            GGMLQuantizationType.F64: (1, 8),
            GGMLQuantizationType.IQ1_M: (256, 256 // 8 + 256 // 16 + 256 // 32),
        }

        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        n_bytes = n_elements * type_size // block_size

        data_bytes = mmap_data[offset : offset + n_bytes]
        data = torch.from_numpy(np.frombuffer(data_bytes, dtype=np.uint8).copy())

        return data, ggml_type
