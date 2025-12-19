#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict
from typing import Dict, List
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import time
import json
import sys
import glob
import numpy as np

# Add parent directory to path to import kt_kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kt_kernel import KTMoEWrapper

import triton
import triton.language as tl


Q_BITS = 4
STORAGE_BITS = 32
PACK_NUM = STORAGE_BITS // Q_BITS
NUMA_NUM = 2

REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def load_model_config(input_path: str, input_type: str = None) -> Dict:
    """Load model configuration from config.json

    Args:
        input_path: Path to directory containing config.json
        input_type: Input weight type (fp8/fp16/bf16/awq), used to validate FP8 config

    Returns:
        Dictionary with model configuration
    """
    config_path = os.path.join(input_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {input_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if "text_config" in config:
        text_cfg = config["text_config"]
    else:
        text_cfg = config

    # Extract required fields with fallbacks
    model_config = {
        "num_experts": text_cfg.get("n_routed_experts", text_cfg.get("num_experts")),
        "num_experts_per_tok": text_cfg.get("num_experts_per_tok", 2),
        "hidden_size": text_cfg.get("hidden_size"),
        "moe_intermediate_size": text_cfg.get("moe_intermediate_size", text_cfg.get("intermediate_size")),
    }

    # Validate required fields
    missing_fields = [k for k, v in model_config.items() if v is None]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")

    # For FP8 input, extract and validate quantization_config
    if input_type == "fp8":
        quant_config = config.get("quantization_config") or text_cfg.get("quantization_config")
        if quant_config is None:
            raise ValueError(
                "FP8 input type specified but 'quantization_config' not found in config.json. "
                "Expected quantization_config with weight_block_size field."
            )

        weight_block_size = quant_config.get("weight_block_size")
        if weight_block_size is None:
            raise ValueError(
                "FP8 quantization_config found but 'weight_block_size' field is missing. "
                "Expected format: 'weight_block_size': [128, 128]"
            )

        if not isinstance(weight_block_size, list) or len(weight_block_size) != 2:
            raise ValueError(
                f"Invalid weight_block_size format: {weight_block_size}. "
                "Expected a list of two integers, e.g., [128, 128]"
            )

        model_config["fp8_weight_block_size"] = weight_block_size
        print(f"FP8 quantization config detected:")
        print(f"  format: {quant_config.get('fmt', 'unknown')}")
        print(f"  weight_block_size: {weight_block_size}")
    return model_config


def pack(imatrix: torch.Tensor):
    """
    Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of packing, either "column" or "row"

    Returns:
        qmatrix (torch.Tensor): packed matrix of integers
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=imatrix.device)

    imatrix = torch.bitwise_and(imatrix, 0x0F).to(torch.int32)  # eventually correct overflow

    imatrix = imatrix.view(imatrix.shape[0], imatrix.shape[1], -1, PACK_NUM)
    qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, None, :]).sum(dim=-1)

    qmatrix = qmatrix.to(torch.int32)

    return qmatrix


def unpack(qmatrix: torch.Tensor):
    """
    Unpacks a 32-bit packed integer matrix into a 4-bit integer matrix.

    Args:
        qmatrix (torch.Tensor): matrix of packed integers
        direction (str): direction of unpacking, either "column" or "row"

    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=qmatrix.device)

    imatrix = torch.bitwise_right_shift(qmatrix[:, :, :, None], shifts[None, None, None, :]).view(
        qmatrix.shape[0], qmatrix.shape[1], -1
    )

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    return imatrix


def reverse_awq_interleaving(imatrix: torch.Tensor):
    """Reverse AWQ interleaving to get original order"""
    # Reshape to handle interleaving at pack level
    original_shape = imatrix.shape
    imatrix_reshaped = imatrix.view(original_shape[0], original_shape[1], -1, PACK_NUM)

    # Apply reverse AWQ pack order
    imatrix_reordered = imatrix_reshaped[:, :, :, REVERSE_AWQ_PACK_ORDER]

    return imatrix_reordered.view(original_shape)


def unpack_reverse_awq_interleaving(qweight: torch.Tensor, qzeros: torch.Tensor = None):
    """
    Row-major unpack AWQ I32 -> INT4 and reverse interleaving to get original order

    Args:
        qweight: Packed AWQ weights with interleaving (I32)
        qzeros: Packed AWQ zeros with interleaving (I32, optional)

    Returns:
        Tuple of (unpacked_weights, unpacked_zeros) in row major order (original)
    """
    # Step 1: Row-major unpack I32 to INT4
    iweights = unpack(qweight)  # Use row direction for row-major

    if qzeros is not None:
        izeros = unpack(qzeros)  # Use row direction for row-major
    else:
        izeros = None

    # Step 2: Reverse AWQ interleaving to get original row-major order
    iweights_original = reverse_awq_interleaving(iweights)

    if izeros is not None:
        izeros_original = reverse_awq_interleaving(izeros)
    else:
        izeros_original = None

    return iweights_original, izeros_original


def pack_column_major_1d(iweights: torch.Tensor, izeros: torch.Tensor = None):
    """
    Pack INT4 -> I32 then flatten to 1D with different logic for weights vs zeros

    Args:
        iweights: Unpacked weights in row major order (INT4)
        izeros: Unpacked zeros in row major order (INT4, optional)

    Returns:
        Tuple of (packed_weights, packed_zeros) as 1D tensors
    """
    # qweight: transpose to column-major then pack
    iweights_transposed = iweights.transpose(1, 2).contiguous()
    qweight = pack(iweights_transposed)
    # qweight = qweight_2d.flatten()  # Flatten to 1D

    # qzeros: NO transpose, keep original shape, pack with original interleaving (01234567)
    if izeros is not None:
        qzeros = pack(izeros)  # Keep original shape, original interleaving
        # qzeros = qzeros_2d.flatten()  # Flatten to 1D
    else:
        qzeros = None

    return qweight, qzeros


class ConverterBase:
    """Base class for converting model weights.

    Subclasses must implement `_convert_layer_experts` to handle the expert
    tensor transformation for a given quantization method (e.g., awq, int4, int8).
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_config: Dict,
        cpuinfer_threads: int = 60,
        threadpool_count: int = 2,
        input_type: str = None,
        merge_to_safetensor: bool = True,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.model_config = model_config
        self.cpuinfer_threads = cpuinfer_threads
        self.threadpool_count = threadpool_count
        self.input_type = input_type
        self.merge_to_safetensor = merge_to_safetensor
        self.tensor_file_map: Dict[str, str] = {}  # key -> filename
        self.tensor_key_map: Dict[str, str] = {}  # old key -> new key
        self.file_handle_map: Dict[str, any] = {}  # filename -> file

        # Extract commonly used config values for convenience
        self.num_experts = model_config["num_experts"]
        self.num_experts_per_tok = model_config["num_experts_per_tok"]
        self.hidden_size = model_config["hidden_size"]
        self.moe_intermediate_size = model_config["moe_intermediate_size"]
        self.layout = "base"

        # Load input safetensors files
        self._load_input_files()

    def _load_input_files(self):
        """Load all safetensors files from input directory"""
        print(f"Loading safetensors files from {self.input_path}")

        found_safetensor = False
        for root, _, files in os.walk(self.input_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    try:
                        handle = safe_open(file_path, framework="pt")
                        self.file_handle_map[file] = handle
                        renamed = False
                        for key in handle.keys():
                            if "language_model" in key:
                                key_ = key.replace("language_model.", "")
                                # print("  Renaming key:", key, "->", key_)
                                renamed = True
                            else:
                                key_ = key
                            self.tensor_key_map[key_] = key
                            self.tensor_file_map[key_] = file
                        print(
                            f"  Loaded: {file} ({len(list(handle.keys()))} tensors){' (renamed keys)' if renamed else ''}"
                        )
                    except Exception as e:
                        print(f"  Error loading {file}: {e}")

        if not found_safetensor:
            raise FileNotFoundError(f"No safetensors files found in {self.input_path}")

        print(f"Total tensors loaded: {len(self.tensor_file_map)}")

    def _load_tensor(self, key: str) -> torch.Tensor:
        """Load tensor by key"""
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found")

        file = self.tensor_file_map[key]
        handle = self.file_handle_map[file]
        return handle.get_tensor(self.tensor_key_map.get(key, key))

    # layers_id -> list[experts_id]
    def _find_expert_layers(self) -> Dict[int, List[int]]:
        """Find all layers and experts in the model"""
        layers = defaultdict(set)

        # detect layout
        for key in self.tensor_file_map.keys():
            if "mlp.experts" in key and "gate_up" in key:
                self.layout = "fused"
                break

        if self.layout == "fused":  # Pattern: model.layers.{layer}.mlp.experts.{proj}
            layers = set()
            for key in self.tensor_file_map.keys():
                if "model.layers." in key and ".mlp.experts." in key:
                    parts = key.split(".")
                    if len(parts) >= 6:
                        layer_idx = int(parts[2])
                        layers.add(layer_idx)

            result: Dict[int, List[int]] = {}
            for layer_idx in sorted(layers):
                result[layer_idx] = [-1]

            print(f"Found {len(result)} layers with fused MoE experts")
            return result

        # Pattern: model.layers.{layer}.mlp.experts.{expert}.{proj}.{type}
        for key in self.tensor_file_map.keys():
            if "model.layers." in key and ".mlp.experts." in key:
                parts = key.split(".")
                if len(parts) >= 6:
                    layer_idx = int(parts[2])
                    expert_idx = int(parts[5])
                    layers[layer_idx].add(expert_idx)

        # Convert to sorted lists
        result: Dict[int, List[int]] = {}
        for layer_idx, expert_set in layers.items():
            result[layer_idx] = sorted(list(expert_set))

        print(f"Found {len(result)} layers with MoE experts:")
        for layer_idx, experts in sorted(result.items()):
            print(f"  Layer {layer_idx}: {len(experts)} experts (0-{max(experts)})")

        return result

    def _convert_layer_experts(self, layer_idx: int, expert_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Subclasses must implement expert conversion for a given layer.

        Expected to return a mapping from output tensor keys to tensors.
        """
        raise NotImplementedError("Subclasses must implement _convert_layer_experts")

    def convert(self, resume_layer: int = 0):
        """Convert all expert layers using subclass-specific logic.

        Args:
            resume_layer (int, optional): The layer index to resume conversion from.
                Layers with an index lower than this will be skipped. Defaults to 0.
        """
        print("Starting conversion...")
        print(f"Input: {self.input_path}")
        print(f"Output: {self.output_path}")
        if resume_layer > 0:
            print(f"Resuming from layer: {resume_layer}")

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Find all expert layers
        expert_layers = self._find_expert_layers()

        if not expert_layers:
            print("No MoE expert layers found in input!")
            return

        # Convert each layer with memory management
        all_tensors: Dict[str, torch.Tensor] = {}

        # Enable memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Process layers with memory cleanup
        for i, (layer_idx, expert_ids) in enumerate(sorted(expert_layers.items())):
            if layer_idx < resume_layer:
                continue
            print(f"Processing layer {layer_idx} ({i+1}/{len(expert_layers)})...")

            layer_tensors = self._convert_layer_experts(layer_idx, expert_ids)
            all_tensors.update(layer_tensors)

            # Periodic garbage collection to free memory
            if (i + 1) % 5 == 0:  # Every 5 layers
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  Memory cleanup after layer {layer_idx}")

        if self.merge_to_safetensor:
            # Copy non-expert tensors (embeddings, norms, etc.)
            print("Copying non-expert tensors...")
            for key in self.tensor_file_map.keys():
                if not (".mlp.experts." in key):
                    # Convert key format for consistency
                    if key.startswith("model."):
                        # Convert model.layers.X -> blk.X for non-expert layers
                        new_key = key.replace("model.layers.", "blk.").replace("model.", "")
                        all_tensors[new_key] = self._load_tensor(key)
                    else:
                        all_tensors[key] = self._load_tensor(key)

            # Save all tensors
            print(f"Saving {len(all_tensors)} tensors...")

            # Split into multiple files if too large
            max_tensors_per_file = 3000  # Adjust based on memory constraints
            tensor_items = list(all_tensors.items())

            if len(tensor_items) <= max_tensors_per_file:
                # Single file
                output_file = os.path.join(self.output_path, "model.safetensors")
                save_file(dict(tensor_items), output_file)
                print(f"Saved to: {output_file}")
            else:
                # Multiple files
                for i in range(0, len(tensor_items), max_tensors_per_file):
                    batch = dict(tensor_items[i : i + max_tensors_per_file])
                    output_file = os.path.join(self.output_path, f"model-{i//max_tensors_per_file + 1:05d}.safetensors")
                    save_file(batch, output_file)
                    print(f"Saved batch to: {output_file}")

            # Copy config files
            self._copy_config_files()

            print("Conversion completed successfully!")
        else:
            print("Skipping safetensor merge, weights kept in layer folder structure")
            print("Conversion completed successfully!")

    def _copy_config_files(self):
        """Copy configuration files to output directory"""
        config_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

        for config_file in config_files:
            src_path = os.path.join(self.input_path, config_file)
            if os.path.exists(src_path):
                import shutil

                dst_path = os.path.join(self.output_path, config_file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {config_file}")

    def close(self):
        """Close all file handles"""
        self.file_handle_map.clear()


class AWQToColumnMajorConverter(ConverterBase):
    """Convert raw AWQ safetensors to NUMA-sliced column-major format."""

    # NOTE: Only this method differs across quantization methods.
    def _convert_layer_experts(self, layer_idx: int, expert_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Convert all experts in a layer to column major format with optimized AWQ processing"""
        output_tensors = {}

        start_time = time.time()
        print(f"Converting layer {layer_idx} with {len(expert_ids)} experts...")

        # Pre-compute projection name mappings
        proj_mappings = {"up_proj": "ffn_up_exps", "gate_proj": "ffn_gate_exps", "down_proj": "ffn_down_exps"}

        # Batch process all experts to reduce nested loops
        for proj_name, out_proj in proj_mappings.items():
            # Load all expert tensors for this projection at once
            expert_qweights = []
            expert_qzeros = []
            expert_scales = []
            valid_experts = []

            for expert_id in expert_ids:
                qweight_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{proj_name}.qweight"
                qzeros_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{proj_name}.qzeros"
                scales_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{proj_name}.scales"

                if qweight_key in self.tensor_file_map:
                    qweight = self._load_tensor(qweight_key)
                    qzeros = self._load_tensor(qzeros_key) if qzeros_key in self.tensor_file_map else None
                    scales = self._load_tensor(scales_key) if scales_key in self.tensor_file_map else None

                    expert_qweights.append(qweight)
                    expert_qzeros.append(qzeros)
                    expert_scales.append(scales)
                    valid_experts.append(expert_id)

            if not valid_experts:
                continue

            print(f"  Processing {proj_name}: {len(valid_experts)} experts")

            qweights_stack = torch.stack([w for w in expert_qweights if w is not None], dim=0)
            qzeros_stack = torch.stack([z for z in expert_qzeros if z is not None], dim=0)

            batch_size = 128

            for batch_start in range(0, len(valid_experts), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_experts))
                qweights_batch = qweights_stack[batch_start:batch_end].to("cuda")
                qzeros_batch = qzeros_stack[batch_start:batch_end].to("cuda")
                iweights_batch, izeros_batch = unpack_reverse_awq_interleaving(qweights_batch, qzeros_batch)
                qweights_1d_batch, qzeros_1d_batch = pack_column_major_1d(iweights_batch, izeros_batch)

                for idx in range(batch_start, batch_end):
                    expert_id = valid_experts[idx]
                    batch_idx = idx - batch_start
                    output_tensors[f"blk.{layer_idx}.{out_proj}.{expert_id}.scale"] = expert_scales[idx].flatten()
                    output_tensors[f"blk.{layer_idx}.{out_proj}.{expert_id}.weight"] = qweights_1d_batch[
                        batch_idx
                    ].cpu()
                    if qzeros_1d_batch is not None:
                        output_tensors[f"blk.{layer_idx}.{out_proj}.{expert_id}.qzeros"] = qzeros_1d_batch[
                            batch_idx
                        ].cpu()

            gc.collect()

        elapsed = time.time() - start_time
        print(f"  Generated {len(output_tensors)} column-major 1D tensors in {elapsed:.2f}s")
        return output_tensors


class OnlineQuantConverter(ConverterBase):
    """Convert FP8/FP16/BF16 weights to quantized format using AMXMoEWrapper.

    Performs online quantization (FP8/FP16/BF16 -> INT4/INT8) using AMXMoEWrapper
    with NUMA-aware memory management and automatic weight saving.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_config: Dict,
        cpuinfer_threads: int = 60,
        threadpool_count: int = 2,
        input_type: str = None,
        quant_method: str = "int4",
        merge_to_safetensor: bool = True,
    ):
        super().__init__(
            input_path, output_path, model_config, cpuinfer_threads, threadpool_count, input_type, merge_to_safetensor
        )
        self.quant_method = quant_method

        # For FP8, get block size from model_config
        if input_type == "fp8":
            self.fp8_block_size = model_config.get("fp8_weight_block_size", [128, 128])
        else:
            self.fp8_block_size = None

    def _dequantize_fp8_blockwise(self, fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 weight with block-wise scaling.

        Args:
            fp8_weight: FP8 weight tensor of shape [H, W]
            scale_inv: Scale inverse tensor of shape [H//block_size, W//block_size]

        Returns:
            Dequantized BF16 weight tensor of shape [H, W]
        """
        H, W = fp8_weight.shape
        num_blocks_h, num_blocks_w = scale_inv.shape

        # Infer block size from shapes
        block_h = H // num_blocks_h
        block_w = W // num_blocks_w

        # Reshape fp8_weight to [num_blocks_h, block_h, num_blocks_w, block_w]
        fp8_reshaped = fp8_weight.view(num_blocks_h, block_h, num_blocks_w, block_w)

        # Reshape scale_inv to [num_blocks_h, 1, num_blocks_w, 1] for broadcasting
        scale_inv_reshaped = scale_inv.view(num_blocks_h, 1, num_blocks_w, 1)

        # Dequantize: convert to bf16 and multiply by scale_inv
        dequantized = fp8_reshaped.to(torch.bfloat16) * scale_inv_reshaped

        # Reshape back to [H, W]
        dequantized = dequantized.view(H, W).contiguous()

        return dequantized

    def _load_binary_tensor(self, file_path: str) -> torch.Tensor:
        """Load .kt format binary tensor file

        Args:
            file_path: Path to .kt binary file

        Returns:
            torch.Tensor: Loaded tensor
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            binary_data = f.read()

        # Determine dtype based on file name
        if "scale" in file_path:
            # Scale tensors are typically float32
            np_array = np.frombuffer(binary_data, dtype=np.float32)
        else:
            # Quant tensors are typically int8
            np_array = np.frombuffer(binary_data, dtype=np.int8)

        tensor = torch.from_numpy(np_array.copy())
        return tensor

    def _load_layer_tensors_from_disk(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load all quantized tensors from _layer_{layer_idx} folder

        Args:
            layer_idx: Layer index

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys in format:
                'blk.{layer}.ffn_{proj}_exps.{expert}.numa.{numa_idx}.{weight|scale}'
        """
        layer_path = os.path.join(self.output_path, f"_layer_{layer_idx}")
        if not os.path.exists(layer_path):
            raise FileNotFoundError(f"Layer folder not found: {layer_path}")

        tensors = {}

        # Get AMX method from quant_method parameter (INT4/INT8)
        # Map quant_method to AMX_METHOD format
        quant_to_amx_map = {
            "int4": "INT4",
            "int8": "INT8",
            "moe_int4": "MOE_INT4",
            "moe_int8": "MOE_INT8",
        }
        amx_method = quant_to_amx_map.get(self.quant_method, "INT4")

        # Iterate through all NUMA folders
        for numa_idx in range(self.threadpool_count):
            numa_folder = os.path.join(layer_path, f"_numa_{numa_idx}")
            if not os.path.exists(numa_folder):
                print(f"  Warning: NUMA folder not found: {numa_folder}, skipping...")
                continue

            # Iterate through all experts
            for expert_id in range(self.num_experts):
                # For each projection (down, gate, up)
                proj_mappings = [("down", "ffn_down_exps"), ("gate", "ffn_gate_exps"), ("up", "ffn_up_exps")]

                for proj_name, proj_key in proj_mappings:
                    # Build file patterns
                    quant_pattern = os.path.join(numa_folder, f"{amx_method}_{proj_name}_{expert_id}_*Byte_quant_.kt")
                    scale_pattern = os.path.join(numa_folder, f"{amx_method}_{proj_name}_{expert_id}_*Byte_scale_.kt")

                    # Find files using glob
                    quant_files = glob.glob(quant_pattern)
                    scale_files = glob.glob(scale_pattern)

                    # Build keys (following merge_small_tensor.py format)
                    weight_key = f"blk.{layer_idx}.{proj_key}.{expert_id}.numa.{numa_idx}.weight"
                    scale_key = f"blk.{layer_idx}.{proj_key}.{expert_id}.numa.{numa_idx}.scale"

                    # Load quant tensor
                    if quant_files:
                        if len(quant_files) > 1:
                            raise ValueError(f"Multiple quant files found: {quant_files}")
                        tensors[weight_key] = self._load_binary_tensor(quant_files[0])

                    # Load scale tensor
                    if scale_files:
                        if len(scale_files) > 1:
                            raise ValueError(f"Multiple scale files found: {scale_files}")
                        tensors[scale_key] = self._load_binary_tensor(scale_files[0])

        return tensors

    def _remove_layer_folder(self, layer_idx: int):
        """Remove _layer_{layer_idx} folder and all its contents

        Args:
            layer_idx: Layer index
        """
        import shutil

        layer_path = os.path.join(self.output_path, f"_layer_{layer_idx}")
        if os.path.exists(layer_path):
            shutil.rmtree(layer_path)
            print(f"  Removed temporary folder: {layer_path}")

    def _convert_layer_experts(self, layer_idx: int, expert_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Convert all experts in a layer using online quantization via AMXMoEWrapper"""
        start_time = time.time()
        print(
            f"Converting layer {layer_idx} with {len(expert_ids) if self.layout == 'base' else 'fused'} experts via online quantization..."
        )
        # Load all expert weights for this layer
        if self.layout == "fused":
            if self.input_type not in ["bf16", "fp16"]:
                raise ValueError(f"Fused path currently supports bf16/fp16 only, got input_type={self.input_type}")

            proj_set = set()
            prefix = f"model.layers.{layer_idx}.mlp.experts."
            for key in self.tensor_file_map.keys():
                if key.startswith(prefix):
                    parts = key.split(".")
                    if len(parts) >= 6:
                        proj_set.add(parts[5])

            if not proj_set:
                raise ValueError(f"[Fused] No fused MoE experts found for layer {layer_idx} under 'model.layers'")

            projs = sorted(proj_set)
            print(f"  [Fused] layer {layer_idx} fused proj keys: {projs}")
            if len(projs) < 2:
                raise ValueError(
                    f"[Fused] Expect at least 2 fused tensors (down & gate_up) in layer {layer_idx}, got {len(projs)}"
                )

            fused_tensors = []
            for p in projs:
                key = f"model.layers.{layer_idx}.mlp.experts.{p}"
                if key not in self.tensor_file_map:
                    raise KeyError(f"[Fused] Missing fused tensor {key} for layer {layer_idx}")
                w = self._load_tensor(key)
                if self.input_type == "fp16":
                    w = w.to(torch.bfloat16)
                print(f"    [Fused] tensor {p} shape: {tuple(w.shape)}")
                fused_tensors.append(w)

            #   fused_tensors[0] : down-like, [E, I, H]
            #   fused_tensors[1] : gate_up-like, [E, H, 2I]
            down_fused = fused_tensors[0]
            gate_up_fused = fused_tensors[1]

            #    gate_up_fused: [E, H, 2I] -> [E, 2I, H] -> gate / up
            if gate_up_fused.dim() != 3:
                raise ValueError(
                    f"[Fused] Expect gate_up fused tensor to be 3D, got shape {tuple(gate_up_fused.shape)}"
                )
            E, H, twoI = gate_up_fused.shape
            if twoI % 2 != 0:
                raise ValueError(f"[Fused] gate_up last dim (2I) not even: {twoI}")
            I = twoI // 2

            gate_up_T = gate_up_fused.transpose(1, 2).contiguous()  # [E, 2I, H]
            gate_proj = gate_up_T[:, :I, :]  # [E, I, H]
            up_proj = gate_up_T[:, I:, :]  # [E, I, H]

            if down_fused.dim() != 3:
                raise ValueError(f"[Fused] Expect down fused tensor to be 3D, got shape {tuple(down_fused.shape)}")
            if down_fused.shape[0] != E:
                raise ValueError(f"[Fused] down_fused expert dim mismatch: {down_fused.shape[0]} vs gate_up {E}")
            down_proj = down_fused.transpose(1, 2).contiguous()  # [E, H, I]
            del fused_tensors
            del gate_up_fused
            del down_fused
        else:
            gate_weights = []
            up_weights = []
            down_weights = []

            for expert_id in expert_ids:
                gate_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"
                up_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"
                down_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"

                if gate_key not in self.tensor_file_map:
                    raise KeyError(f"Missing gate weight for layer {layer_idx}, expert {expert_id}")
                if up_key not in self.tensor_file_map:
                    raise KeyError(f"Missing up weight for layer {layer_idx}, expert {expert_id}")
                if down_key not in self.tensor_file_map:
                    raise KeyError(f"Missing down weight for layer {layer_idx}, expert {expert_id}")

                # Load weights based on input type
                if self.input_type == "fp8":
                    # Load FP8 weights and their scale_inv tensors
                    gate_scale_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight_scale_inv"
                    up_scale_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight_scale_inv"
                    down_scale_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight_scale_inv"

                    if gate_scale_key not in self.tensor_file_map:
                        raise KeyError(f"Missing gate weight_scale_inv for layer {layer_idx}, expert {expert_id}")
                    if up_scale_key not in self.tensor_file_map:
                        raise KeyError(f"Missing up weight_scale_inv for layer {layer_idx}, expert {expert_id}")
                    if down_scale_key not in self.tensor_file_map:
                        raise KeyError(f"Missing down weight_scale_inv for layer {layer_idx}, expert {expert_id}")

                    # Load FP8 weights and scales
                    gate_fp8 = self._load_tensor(gate_key).to("cuda")
                    up_fp8 = self._load_tensor(up_key).to("cuda")
                    down_fp8 = self._load_tensor(down_key).to("cuda")

                    gate_scale_inv = self._load_tensor(gate_scale_key).to("cuda")
                    up_scale_inv = self._load_tensor(up_scale_key).to("cuda")
                    down_scale_inv = self._load_tensor(down_scale_key).to("cuda")

                    # Dequantize FP8 to BF16 using block-wise scaling
                    gate_weight = weight_dequant(gate_fp8, gate_scale_inv).to("cpu").to(torch.bfloat16).contiguous()
                    up_weight = weight_dequant(up_fp8, up_scale_inv).to("cpu").to(torch.bfloat16).contiguous()
                    down_weight = weight_dequant(down_fp8, down_scale_inv).to("cpu").to(torch.bfloat16).contiguous()

                elif self.input_type == "fp16":
                    # Load FP16 and convert to BF16
                    gate_weight = self._load_tensor(gate_key).to(torch.bfloat16)
                    up_weight = self._load_tensor(up_key).to(torch.bfloat16)
                    down_weight = self._load_tensor(down_key).to(torch.bfloat16)

                elif self.input_type == "bf16":
                    # Load BF16 directly
                    gate_weight = self._load_tensor(gate_key)
                    up_weight = self._load_tensor(up_key)
                    down_weight = self._load_tensor(down_key)

                else:
                    raise ValueError(f"Unsupported input_type for INT4 conversion: {self.input_type}")

                gate_weights.append(gate_weight)
                up_weights.append(up_weight)
                down_weights.append(down_weight)

            # Stack weights into single tensors: [num_experts, ...]
            gate_proj = torch.stack(gate_weights, dim=0).contiguous()
            up_proj = torch.stack(up_weights, dim=0).contiguous()
            down_proj = torch.stack(down_weights, dim=0).contiguous()
            del gate_weights, up_weights, down_weights

        print(f"  Loaded weights shapes:")
        print(f"    gate_proj: {gate_proj.shape}")
        print(f"    up_proj: {up_proj.shape}")
        print(f"    down_proj: {down_proj.shape}")

        # Create physical_to_logical_map: identity mapping where position i maps to expert i
        physical_to_logical_map = torch.arange(self.num_experts, dtype=torch.int64)

        # Map quant_method to AMX method format
        quant_to_amx_map = {
            "int4": "AMXINT4",
            "int8": "AMXINT8",
            "moe_int4": "MOE_INT4",
            "moe_int8": "MOE_INT8",
        }
        amx_method = quant_to_amx_map.get(self.quant_method, "AMXINT4")

        # Create AMXMoEWrapper instance for this layer
        # num_gpu_experts=0 since we're converting all experts to CPU format
        wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_gpu_experts=0,  # All experts on CPU for conversion
            cpuinfer_threads=self.cpuinfer_threads,
            threadpool_count=self.threadpool_count,
            weight_path=self.output_path,  # Output path for quantized weights
            chunked_prefill_size=512,  # Arbitrary value, not critical for conversion
            cpu_save=True,  # Enable saving quantized weights to output
            method=amx_method,  # Specify quantization method (AMXINT4 or AMXINT8)
        )

        # Load and quantize weights from tensors
        # This triggers the quantization process and saves to disk
        wrapper.load_weights_from_tensors(gate_proj, up_proj, down_proj, physical_to_logical_map)

        # Clean up to free memory
        del gate_proj, up_proj, down_proj
        gc.collect()

        elapsed = time.time() - start_time

        if self.merge_to_safetensor:
            # Load quantized tensors from disk
            print(f"  Loading quantized tensors from disk...")
            layer_tensors = self._load_layer_tensors_from_disk(layer_idx)
            print(f"  Loaded {len(layer_tensors)} tensors")

            # Remove temporary layer folder
            self._remove_layer_folder(layer_idx)

            print(f"  Layer {layer_idx} quantized and saved in {elapsed:.2f}s")

            # Return loaded tensors
            return layer_tensors
        else:
            # Keep layer folders, return empty dict
            print(f"  Layer {layer_idx} quantized and saved in {elapsed:.2f}s")
            print(f"  Keeping layer folder structure at {self.output_path}/_layer_{layer_idx}")
            return {}


"""
Example usage(test passed):
python convert_cpu_weights.py --input-path /mnt/data3/models/DeepSeek-R1-0528/ --input-type fp8 --output /mnt/data3/models/DeepSeek-R1-0528-INT4-test --quant-method int4 --cpuinfer-threads 60 --threadpool-count 2
python convert_cpu_weights.py --input-path /mnt/data3/models/DeepSeek-R1-0528/ --input-type fp8 --output /mnt/data3/models/DeepSeek-R1-0528-INT8-test --quant-method int8 --cpuinfer-threads 60 --threadpool-count 2
python convert_cpu_weights.py --input-path /mnt/data2/models/Qwen3-Next-80B-A3B-Instruct --input-type bf16 --output /mnt/data2/models/Qwen3-Next-80B-A3B-Instruct-INT4-test --quant-method int4 --cpuinfer-threads 60 --threadpool-count 2
"""


def main():
    parser = argparse.ArgumentParser(description="Convert SafeTensors to column major 1D format")
    parser.add_argument("--input-path", "-i", required=True, help="Input directory with safetensors")
    parser.add_argument(
        "--input-type",
        choices=["awq", "fp8", "fp16", "bf16"],
        required=True,
        help="Input weight type (awq/fp8/fp16/bf16)",
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory for converted safetensors")
    parser.add_argument(
        "--quant-method",
        choices=["int4", "int8", "awq", "moe_int4", "moe_int8"],
        default="int4",
        help="Quantization method for output (default: int4)",
    )
    parser.add_argument(
        "--cpuinfer-threads",
        type=int,
        default=60,
        help="Number of CPU inference threads (default: 60)",
    )
    parser.add_argument(
        "--threadpool-count",
        type=int,
        default=2,
        help="Number of NUMA subpools for thread distribution (default: 2)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for conversion if available")
    parser.add_argument(
        "--no-merge-safetensor",
        action="store_true",
        default=False,
        help="Keep layer folders without merging to safetensor files (default: False)",
    )
    parser.add_argument(
        "--resume-layer",
        type=int,
        default=0,
        help="Resume conversion starting at this layer index (default: 0)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_path):
        print(f"Error: Input path does not exist: {args.input_path}")
        return 1
    try:
        # Load model configuration from config.json
        print("Loading model configuration...")
        model_config = load_model_config(args.input_path, args.input_type)
        print(f"Model config: {model_config}")
        print(f"  num_experts: {model_config['num_experts']}")
        print(f"  num_experts_per_tok: {model_config['num_experts_per_tok']}")
        print(f"  hidden_size: {model_config['hidden_size']}")
        print(f"  moe_intermediate_size: {model_config['moe_intermediate_size']}")
        print(f"CPU inference config:")
        print(f"  cpuinfer_threads: {args.cpuinfer_threads}")
        print(f"  threadpool_count: {args.threadpool_count}")
        print()

        # Create converter by quantization method
        quant_method = args.quant_method.lower()
        merge_to_safetensor = not args.no_merge_safetensor

        if quant_method == "awq":
            converter = AWQToColumnMajorConverter(
                args.input_path,
                args.output,
                model_config,
                args.cpuinfer_threads,
                args.threadpool_count,
                input_type=None,
                merge_to_safetensor=merge_to_safetensor,
            )
        elif quant_method in ["int4", "int8", "moe_int4", "moe_int8"] and args.input_type in ["fp8", "fp16", "bf16"]:
            # Use OnlineQuantConverter for both INT4 and INT8 quantization
            converter = OnlineQuantConverter(
                args.input_path,
                args.output,
                model_config,
                args.cpuinfer_threads,
                args.threadpool_count,
                args.input_type,
                quant_method,
                merge_to_safetensor,
            )
        else:
            raise ValueError(
                f"Unsupported quant_method: {args.quant_method} or incompatible input_type: {args.input_type}"
            )

        # Run conversion
        converter.convert(resume_layer=args.resume_layer)

        # Cleanup
        converter.close()
        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
