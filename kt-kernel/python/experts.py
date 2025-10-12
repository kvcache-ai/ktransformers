# Wrapper for AMX MoE CPU inference operations
# This module encapsulates CPU inference engine, weight loading, and buffer management
# SPDX-License-Identifier: Apache-2.0

"""
Expert wrappers for CPU-based MoE inference.

This module provides high-level Python wrappers around the low-level C++ kernel
implementations, handling weight loading, buffer management, and forward inference.
"""

from __future__ import annotations

import torch
from typing import List, Dict
from safetensors import safe_open
import os
import ctypes

# Import the C++ extension module (compiled as cpuinfer_ext)
import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, AMXInt4_MOE, AMXInt8_MOE


class SafeTensorLoader:
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
        # base_key: blk.{layer_index}
        # blk.{layer_index}.ffn_[up, down, gate]_exps.{expert_id}.numa.{numa_id}.weight
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


class KExpertsCPUBuffer:
    capture_bs: List = list()
    capture_buffers: Dict = dict()
    temp_bs: int = 0
    temp_buffer: tuple = tuple()

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_size)
        batch_size, hidden_size = hidden_states.shape

        if batch_size in KExpertsCPUBuffer.capture_buffers:
            return KExpertsCPUBuffer.capture_buffers[batch_size]
        if batch_size == KExpertsCPUBuffer.temp_bs:
            return KExpertsCPUBuffer.temp_buffer

        input_tensor_cpu = torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
        expert_ids_cpu = torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
        weights_cpu = torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
        output_cpu = torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
        bsz_tensor_cpu = torch.tensor((batch_size), device="cpu", dtype=torch.int32, pin_memory=True)
        output_gpu = torch.zeros_like(hidden_states)

        cur_buffer = (input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu)
        if batch_size in KExpertsCPUBuffer.capture_bs:
            KExpertsCPUBuffer.capture_buffers[batch_size] = cur_buffer
        KExpertsCPUBuffer.temp_bs = batch_size
        KExpertsCPUBuffer.temp_buffer = cur_buffer
        return cur_buffer


class AMXMoEWrapper:
    """
    Wrapper for AMX MoE CPU inference operations.
    Manages CPU inference engine, weight loading, and buffer management.
    """

    _cpu_infer_instance = None
    _safetensor_loader_instance = None

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_gpu_experts: int,
        cpuinfer_threads: int,
        subpool_count: int,
        amx_weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
    ):
        """
        Initialize AMX MoE Wrapper.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            num_gpu_experts: Number of experts to run on GPU
            cpuinfer_threads: Number of CPU inference threads
            subpool_count: Number of NUMA subpools
            amx_weight_path: Path to AMX weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
        """

        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_gpu_experts = num_gpu_experts
        self.amx_weight_path = amx_weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpu_save = cpu_save

        # Initialize CPU inference engine (singleton)
        if AMXMoEWrapper._cpu_infer_instance is None:
            worker_config = cpuinfer_ext.WorkerPoolConfig()

            subpool_numa_map = list(range(subpool_count))
            subpool_thread_count = [
                cpuinfer_threads // subpool_count + (1 if i < cpuinfer_threads % subpool_count else 0)
                for i in range(subpool_count)
            ]

            worker_config.subpool_count = subpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            AMXMoEWrapper._cpu_infer_instance = cpuinfer_ext.CPUInfer(worker_config)

        self.cpu_infer = AMXMoEWrapper._cpu_infer_instance

        # Check if we should load merged safetensor weights
        self.load_merged_weight = False
        import glob

        if glob.glob(os.path.join(amx_weight_path, "*.safetensors")):
            self.load_merged_weight = True

        # Initialize SafeTensor loader (singleton)
        if self.load_merged_weight:
            if AMXMoEWrapper._safetensor_loader_instance is None:
                AMXMoEWrapper._safetensor_loader_instance = SafeTensorLoader(amx_weight_path)
            self.safetensor_loader = AMXMoEWrapper._safetensor_loader_instance

        self.moe = None
        self.gate_weights = None
        self.up_weights = None
        self.down_weights = None
        self.gate_scales = None
        self.up_scales = None
        self.down_scales = None

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        gate_ptr = 0
        up_ptr = 0
        down_ptr = 0

        gate_ptrs = []
        up_ptrs = []
        down_ptrs = []

        gate_scale_ptrs = []
        up_scale_ptrs = []
        down_scale_ptrs = []

        if self.load_merged_weight:
            base_key = f"blk.{self.layer_idx}"
            w = self.safetensor_loader.load_experts(base_key)

            self.gate_weights = w["gate"]
            self.up_weights = w["up"]
            self.down_weights = w["down"]
            self.gate_scales = w["gate_scale"]
            self.up_scales = w["up_scale"]
            self.down_scales = w["down_scale"]

            # Get pointers to weight arrays
            gate_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.gate_weights
            ]

            up_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.up_weights
            ]

            down_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.down_weights
            ]

            gate_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.gate_scales
            ]

            up_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.up_scales
            ]

            down_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.down_scales
            ]

        # Configure MoE
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.num_gpu_experts,
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size

        moe_config.gate_proj = gate_ptr
        moe_config.up_proj = up_ptr
        moe_config.down_proj = down_ptr
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs

        if self.cpu_save:
            moe_config.save = True
            moe_config.load = False
            base_key = f"model.layers.{self.layer_idx}"
            w = self.safetensor_loader.load_experts(base_key)

            self.gate_proj = torch.cat(w["gate_weight"], dim=0).contiguous()
            self.up_proj = torch.cat(w["up_weight"], dim=0).contiguous()
            self.down_proj = torch.cat(w["down_weight"], dim=0).contiguous()

            moe_config.gate_proj = self.gate_proj.data_ptr()
            moe_config.up_proj = self.up_proj.data_ptr()
            moe_config.down_proj = self.down_proj.data_ptr()
        else:
            moe_config.load = True

        if not self.load_merged_weight:
            moe_config.path = self.amx_weight_path

        # Create MoE module based on AMX method
        amx_method = os.environ.get("AMX_METHOD", "AMXINT4")
        if amx_method == "AMXINT4":
            self.moe = AMXInt4_MOE(moe_config)
        elif amx_method == "AMXINT8":
            self.moe = AMXInt8_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported AMX method: {amx_method}")

        # Load weights
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()

        # Clean up temporary weight storage if using merged weights
        if self.load_merged_weight:
            del self.gate_weights
            del self.up_weights
            del self.down_weights
            del self.gate_scales
            del self.up_scales
            del self.down_scales

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ):
        """
        Submit forward inference task to CPU (non-blocking).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization
        """
        # Get CPU buffers
        (
            input_tensor_cpu,
            expert_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(hidden_states, self.num_experts_per_tok)

        # Copy data to CPU
        topk_ids = topk_ids.to(torch.long)
        input_tensor_cpu.copy_(hidden_states, non_blocking=True)
        expert_ids_cpu.copy_(topk_ids, non_blocking=True)
        weights_cpu.copy_(topk_weights, non_blocking=True)

        # Submit task
        self.cpu_infer.submit_with_cuda_stream(
            cuda_stream,
            self.moe.forward_task(
                bsz_tensor_cpu.data_ptr(),
                expert_ids_cpu.size(-1),
                expert_ids_cpu.data_ptr(),
                weights_cpu.data_ptr(),
                input_tensor_cpu.data_ptr(),
                output_cpu.data_ptr(),
                False,
            ),
        )

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream) -> torch.Tensor:
        """
        Synchronize and retrieve forward inference results.

        Args:
            hidden_states: Original input hidden states (for getting buffer)
            cuda_stream: CUDA stream for synchronization

        Returns:
            output_gpu: Output tensor on GPU
        """
        (
            input_tensor_cpu,
            expert_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(hidden_states, self.num_experts_per_tok)

        self.cpu_infer.sync_with_cuda_stream(cuda_stream)
        output_gpu.copy_(output_cpu, non_blocking=True)
        return output_gpu

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ) -> torch.Tensor:
        """
        Execute forward inference synchronously (submit + sync).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization

        Returns:
            Output tensor on GPU
        """
        self.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        return self.sync_forward(hidden_states, cuda_stream)

    @staticmethod
    def set_capture_batch_sizes(capture_bs: List[int]):
        """
        Set batch sizes to capture and cache buffers for.

        This allows pre-allocation of CPU buffers for specific batch sizes,
        improving performance by avoiding buffer re-allocation during inference.

        Args:
            capture_bs: List of batch sizes to capture (e.g., [1, 2, 4, 8, 16])

        Example:
            >>> AMXMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])
        """
        KExpertsCPUBuffer.capture_bs = capture_bs

    @staticmethod
    def get_capture_batch_sizes() -> List[int]:
        """
        Get currently configured capture batch sizes.

        Returns:
            List of batch sizes that are being captured
        """
        return KExpertsCPUBuffer.capture_bs

    @staticmethod
    def clear_buffer_cache():
        """
        Clear all cached buffers.

        This frees up memory by clearing the buffer cache. Useful when you want
        to reset the buffer state or free memory.
        """
        KExpertsCPUBuffer.capture_buffers.clear()
        KExpertsCPUBuffer.temp_bs = 0
        KExpertsCPUBuffer.temp_buffer = tuple()
