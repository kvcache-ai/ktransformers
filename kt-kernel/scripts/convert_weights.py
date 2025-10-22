#!/usr/bin/env python3
"""
Convert Raw AWQ SafeTensors to Column Major format

Usage:
    python convert_awq_to_numa.py --input /path/to/raw/awq --output /path/to/column_major/awq

Input Format:  model.layers.3.mlp.experts.21.down_proj.qweight
Output Format: blk.3.ffn_down_exps.21.weight
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors import pack_to_int32, unpack_from_int32
import gc
import time
import json
import sys
import glob
import numpy as np

# Add parent directory to path to import kt_kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kt_kernel import AMXMoEWrapper

import cpuinfer_ext


Q_BITS = 4
STORAGE_BITS = 32
PACK_NUM = STORAGE_BITS // Q_BITS
NUMA_NUM = 2

REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # Extract required fields with fallbacks
    model_config = {
        "num_experts": config.get("n_routed_experts", config.get("num_experts")),
        "num_experts_per_tok": config.get("num_experts_per_tok", 2),
        "hidden_size": config.get("hidden_size"),
        "moe_intermediate_size": config.get("moe_intermediate_size", config.get("intermediate_size")),
    }

    # Validate required fields
    missing_fields = [k for k, v in model_config.items() if v is None]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")

    # For FP8 input, extract and validate quantization_config
    if input_type == "fp8":
        quant_config = config.get("quantization_config")
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
        subpool_count: int = 2,
        input_type: str = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.model_config = model_config
        self.cpuinfer_threads = cpuinfer_threads
        self.subpool_count = subpool_count
        self.input_type = input_type
        self.tensor_file_map: Dict[str, str] = {}  # key -> filename
        self.file_handle_map: Dict[str, any] = {}  # filename -> file

        # Extract commonly used config values for convenience
        self.num_experts = model_config["num_experts"]
        self.num_experts_per_tok = model_config["num_experts_per_tok"]
        self.hidden_size = model_config["hidden_size"]
        self.moe_intermediate_size = model_config["moe_intermediate_size"]

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
                        for key in handle.keys():
                            self.tensor_file_map[key] = file
                        print(f"  Loaded: {file} ({len(list(handle.keys()))} tensors)")
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
        return handle.get_tensor(key)


    # layers_id -> list[experts_id]
    def _find_expert_layers(self) -> Dict[int, List[int]]:
        """Find all layers and experts in the model"""
        layers = defaultdict(set)

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

    def convert(self):
        """Convert all expert layers using subclass-specific logic."""
        print("Starting conversion...")
        print(f"Input: {self.input_path}")
        print(f"Output: {self.output_path}")

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
            print(f"Processing layer {layer_idx} ({i+1}/{len(expert_layers)})...")

            layer_tensors = self._convert_layer_experts(layer_idx, expert_ids)
            all_tensors.update(layer_tensors)

            # Periodic garbage collection to free memory
            if (i + 1) % 5 == 0:  # Every 5 layers
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  Memory cleanup after layer {layer_idx}")

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
        # for handle in self.file_handle_map.values():
        #     handle.close()
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
        subpool_count: int = 2,
        input_type: str = None,
        quant_method: str = "int4",
    ):
        super().__init__(input_path, output_path, model_config, cpuinfer_threads, subpool_count, input_type)
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

        with open(file_path, 'rb') as f:
            binary_data = f.read()

        # Determine dtype based on file name
        if 'scale' in file_path:
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
        }
        amx_method = quant_to_amx_map.get(self.quant_method, "INT4")

        # Iterate through all NUMA folders
        for numa_idx in range(self.subpool_count):
            numa_folder = os.path.join(layer_path, f"_numa_{numa_idx}")
            if not os.path.exists(numa_folder):
                continue

            # Iterate through all experts
            for expert_id in range(self.num_experts):
                # For each projection (down, gate, up)
                proj_mappings = [
                    ('down', 'ffn_down_exps'),
                    ('gate', 'ffn_gate_exps'),
                    ('up', 'ffn_up_exps')
                ]

                for proj_name, proj_key in proj_mappings:
                    # Build file patterns
                    quant_pattern = os.path.join(
                        numa_folder,
                        f'{amx_method}_{proj_name}_{expert_id}_*Byte_quant_.kt'
                    )
                    scale_pattern = os.path.join(
                        numa_folder,
                        f'{amx_method}_{proj_name}_{expert_id}_*Byte_scale_.kt'
                    )

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
        print(f"Converting layer {layer_idx} with {len(expert_ids)} experts via online quantization...")

        # Load all expert weights for this layer
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
                gate_fp8 = self._load_tensor(gate_key)
                up_fp8 = self._load_tensor(up_key)
                down_fp8 = self._load_tensor(down_key)

                gate_scale_inv = self._load_tensor(gate_scale_key)
                up_scale_inv = self._load_tensor(up_scale_key)
                down_scale_inv = self._load_tensor(down_scale_key)

                # Dequantize FP8 to BF16 using block-wise scaling
                gate_weight = self._dequantize_fp8_blockwise(gate_fp8, gate_scale_inv)
                up_weight = self._dequantize_fp8_blockwise(up_fp8, up_scale_inv)
                down_weight = self._dequantize_fp8_blockwise(down_fp8, down_scale_inv)

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

        print(f"  Loaded weights shapes:")
        print(f"    gate_proj: {gate_proj.shape}")
        print(f"    up_proj: {up_proj.shape}")
        print(f"    down_proj: {down_proj.shape}")

        # Create physical_to_logical_map: identity mapping where position i maps to expert i
        physical_to_logical_map = torch.arange(self.num_experts, dtype=torch.int64)

        # Create AMXMoEWrapper instance for this layer
        # num_gpu_experts=0 since we're converting all experts to CPU format
        wrapper = AMXMoEWrapper(
            layer_idx=layer_idx,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_gpu_experts=0,  # All experts on CPU for conversion
            cpuinfer_threads=self.cpuinfer_threads,
            subpool_count=self.subpool_count,
            amx_weight_path=self.output_path,  # Output path for quantized weights
            chunked_prefill_size=512,  # Arbitrary value, not critical for conversion
            cpu_save=True,  # Enable saving quantized weights to output
        )

        # Load and quantize weights from tensors
        # This triggers the quantization process and saves to disk
        wrapper.load_weights_from_tensors(gate_proj, up_proj, down_proj, physical_to_logical_map)

        # Clean up to free memory
        del gate_weights, up_weights, down_weights
        del gate_proj, up_proj, down_proj
        gc.collect()

        # Load quantized tensors from disk
        print(f"  Loading quantized tensors from disk...")
        layer_tensors = self._load_layer_tensors_from_disk(layer_idx)
        print(f"  Loaded {len(layer_tensors)} tensors")

        # Remove temporary layer folder
        self._remove_layer_folder(layer_idx)

        elapsed = time.time() - start_time
        print(f"  Layer {layer_idx} quantized and saved in {elapsed:.2f}s")

        # Return loaded tensors
        return layer_tensors

"""
Example usage(test passed):
python convert_weights.py --input-path /mnt/data3/models/DeepSeek-V3.1 --input-type fp8 --output /mnt/data3/models/DeepSeek-V3.1-INT4-test --quant-method int4 --cpuinfer-threads 60 --subpool-count 2
python convert_weights.py --input-path /mnt/data2/models/Qwen3-Next-80B-A3B-Instruct --input-type bf16 --output /mnt/data2/models/Qwen3-Next-80B-A3B-Instruct-INT4-test --quant-method int4 --cpuinfer-threads 60 --subpool-count 2
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
        choices=["int4", "int8", "awq"],
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
        "--subpool-count",
        type=int,
        default=2,
        help="Number of NUMA subpools for thread distribution (default: 2)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for conversion if available")

    args = parser.parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

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
        print(f"  subpool_count: {args.subpool_count}")
        print()

        # Create converter by quantization method
        quant_method = args.quant_method.lower()
        if quant_method == "awq":
            converter = AWQToColumnMajorConverter(
                args.input_path, args.output, model_config, args.cpuinfer_threads, args.subpool_count
            )
        elif quant_method in ["int4", "int8"] and args.input_type in ["fp8", "fp16", "bf16"]:
            # Use OnlineQuantConverter for both INT4 and INT8 quantization
            converter = OnlineQuantConverter(
                args.input_path, args.output, model_config, args.cpuinfer_threads, args.subpool_count, args.input_type, quant_method
            )
        else:
            raise ValueError(
                f"Unsupported quant_method: {args.quant_method} or incompatible input_type: {args.input_type}"
            )

        # Run conversion
        converter.convert()

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
