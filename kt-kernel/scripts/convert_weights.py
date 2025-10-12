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
import cpuinfer_ext


Q_BITS = 4
STORAGE_BITS = 32
PACK_NUM = STORAGE_BITS // Q_BITS
NUMA_NUM = 2

REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def __init__(self, input_path: str, output_path: str, bf16_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.bf16_path = bf16_path
        self.tensor_file_map: Dict[str, str] = {}
        self.file_handle_map: Dict[str, any] = {}

        # Load input safetensors files
        self._load_input_files()
        if bf16_path:
            self.tensor_file_map_bf16: Dict[str, str] = {}
            self.file_handle_map_bf16: Dict[str, any] = {}
            self._load_bf16_files()

    def _load_bf16_files(self):
        """Load all bf16 safetensors files from bf16 directory"""
        print(f"Loading bf16 safetensors files from {self.bf16_path}")

        found_safetensor = False
        for root, _, files in os.walk(self.bf16_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    try:
                        handle = safe_open(file_path, framework="pt")
                        self.file_handle_map_bf16[file] = handle
                        for key in handle.keys():
                            self.tensor_file_map_bf16[key] = file
                        print(f"  Loaded: {file} ({len(list(handle.keys()))} tensors)")
                    except Exception as e:
                        print(f"  Error loading {file}: {e}")

        if not found_safetensor:
            raise FileNotFoundError(f"No safetensors files found in {self.bf16_path}")

        print(f"Total tensors loaded: {len(self.tensor_file_map)}")

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

    def _load_tensor_bf16(self, key: str) -> torch.Tensor:
        """Load tensor by key"""
        if key not in self.tensor_file_map_bf16:
            raise KeyError(f"Key {key} not found")

        file = self.tensor_file_map_bf16[key]
        handle = self.file_handle_map_bf16[file]
        return handle.get_tensor(key)

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
        max_tensors_per_file = 2000  # Adjust based on memory constraints
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


class Int4ToColumnMajorConverter(ConverterBase):
    """Convert raw INT4 safetensors to NUMA-sliced column-major format.

    NOTE: Implement `_convert_layer_experts` with the correct INT4 packing rules.
    """

    def _convert_layer_experts(self, layer_idx: int, expert_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Convert all experts in a layer to our numa int4 format"""
        output_tensors = {}

        start_time = time.time()
        print(f"Converting layer {layer_idx} with {len(expert_ids)} experts...")

        # Pre-compute projection name mappings
        proj_mappings = {"up_proj": "ffn_up_exps", "gate_proj": "ffn_gate_exps", "down_proj": "ffn_down_exps"}
        for expert_id in expert_ids:
            # Load expert's all tensors for this projection at once
            # up_expert_weights_out = torch.tensor()
            # gate_expert_weights_out = torch.tensor()
            # down_expert_weights_out = torch.tensor()
            for proj_name, out_proj in proj_mappings.items():
                weight_key = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{proj_name}.weight"

                if weight_key in self.tensor_file_map_bf16:
                    weight = self._load_tensor_bf16(weight_key)
                    if proj_name == "up_proj":
                        up_expert_weights = weight
                        up_output_tensor = torch.empty(weight.numel(), dtype=torch.uint8).continuous()
                        output_tensors[f"blk.{layer_idx}.{out_proj}.{expert_id}.weight"] = up_output_tensor
                    elif proj_name == "gate_proj":
                        gate_expert_weights = weight
                        gate_output_tensor = torch.empty(weight.numel(), dtype=torch.uint8).continuous()
                        output_tensors[f"blk.{layer_idx}.{out_proj}.{expert_id}.weight"] = gate_output_tensor
                    elif proj_name == "down_proj":
                        down_expert_weights = weight
                        down_output_tensor = torch.empty(weight.numel(), dtype=torch.uint8).continuous()
                        output_tensors[f"blk.{layer_idx}.{out_proj}.{expert_id}.weight"] = down_output_tensor

            # call c++ api to get qweights and scales

            gc.collect()
        elapsed = time.time() - start_time
        print(f"  Generated {len(output_tensors)} tensors in {elapsed:.2f}s")
        return output_tensors


class Int8ToColumnMajorConverter(ConverterBase):
    """Convert raw INT8 safetensors to NUMA-sliced column-major format.

    NOTE: Implement `_convert_layer_experts` with the correct INT8 transformation rules.
    """

    def _convert_layer_experts(self, layer_idx: int, expert_ids: List[int]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("INT8 converter not implemented yet. Please implement transformation logic.")


def main():
    parser = argparse.ArgumentParser(description="Convert AWQ SafeTensors to column major 1D format")
    parser.add_argument("--input", "-i", required=True, help="Input directory with raw AWQ safetensors")
    parser.add_argument("--bf16_path", help="Path to bf16 weights if needed for mixed precision(amx for int4&int8)")
    parser.add_argument("--output", "-o", required=True, help="Output directory for hybrid safetensors")
    parser.add_argument(
        "--quant_method",
        choices=["int4", "int8", "awq"],
        default="int4",
        help="Quantization method used in input (default: int4)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for conversion if available")

    args = parser.parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return 1
    try:

        # Create converter by quantization method
        quant_method = args.quant_method.lower()
        if quant_method == "awq":
            converter = AWQToColumnMajorConverter(args.input, args.output)
        elif quant_method == "int4" and args.bf16_path:
            converter = Int4ToColumnMajorConverter(args.input, args.output, args.bf16_path)
        elif quant_method == "int8" and args.bf16_path:
            converter = Int8ToColumnMajorConverter(args.input, args.output, args.bf16_path)
        else:
            raise ValueError(f"Unsupported quant_method: {args.quant_method} or missing bf16_path for int4/int8")

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
