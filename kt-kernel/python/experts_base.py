# Base classes for MoE CPU inference operations
# SPDX-License-Identifier: Apache-2.0

"""
Base infrastructure for CPU-based MoE inference.

This module contains base classes and utilities shared across all backend implementations.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import os
from kt_kernel import kt_kernel_ext

_PIN_MEMORY = torch.cuda.is_available()


def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
) -> torch.Tensor:
    """
    Generate GPU experts masks based on activation frequency.

    Selects the top `num_gpu_experts` experts with highest activation frequency
    across all layers to be placed on GPU.

    Args:
        activation_freq: Activation frequency table of shape (num_layers, num_experts).
                         Higher values indicate more frequently activated experts.
        num_gpu_experts: Total number of experts to place on GPU across all layers.

    Returns:
        gpu_experts_masks: Boolean mask of shape (num_layers, num_experts) on CPU.
                           True means the expert should be on GPU.

    Example:
        >>> activation_freq = torch.tensor([
        ...     [0.1, 0.5, 0.3, 0.8],  # layer 0
        ...     [0.2, 0.4, 0.9, 0.1],  # layer 1
        ... ])
        >>> masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=3)
        >>> # Top 3: layer0-expert3 (0.8), layer1-expert2 (0.9), layer0-expert1 (0.5)
        >>> masks
        tensor([[False,  True, False,  True],
                [False, False,  True, False]])
    """
    num_layers, num_experts_per_layer = activation_freq.shape
    total_experts = num_layers * num_experts_per_layer

    # Clamp num_gpu_experts to valid range
    num_gpu_experts = min(num_gpu_experts, total_experts)
    num_gpu_experts = max(num_gpu_experts, 0)

    if num_gpu_experts == 0:
        return torch.zeros(num_layers, num_experts_per_layer, dtype=torch.bool, device="cpu")

    # Flatten and find top-k indices
    flat_freq = activation_freq.view(-1).to(device="cpu")
    _, top_indices = torch.topk(flat_freq, k=num_gpu_experts, largest=True, sorted=False)

    # Create mask
    gpu_experts_masks = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    gpu_experts_masks[top_indices] = True

    # Reshape to (num_layers, num_experts)
    gpu_experts_masks = gpu_experts_masks.view(num_layers, num_experts_per_layer)

    return gpu_experts_masks


class KExpertsCPUBuffer:
    """
    CPU buffer management for expert computation.

    Manages pinned memory buffers for efficient GPU-CPU data transfer.
    """

    capture_bs: List = list()
    capture_buffers: Dict = dict()
    temp_bs: int = 0
    temp_buffer: tuple = tuple()
    buffer_depth: int = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        hidden_size = hidden_states.shape[-1]
        batch_size = hidden_states.shape[0]

        pin_memory = True

        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size == cls.temp_bs:
            return cls.temp_buffer

        input_tensor_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        immediate_experts_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        deferred_experts_ids_cpu = [
            torch.full((batch_size, num_experts_per_tok), -1, device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        bsz_tensor_cpu = [
            torch.full((1,), batch_size, device="cpu", dtype=torch.int32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        )
        if batch_size in cls.capture_bs:
            cls.capture_buffers[batch_size] = cur_buffer
        cls.temp_bs = batch_size
        cls.temp_buffer = cur_buffer
        return cur_buffer


class _MoEBase:
    """
    Shared base class for inference and SFT MoE wrappers.

    Provides:
    - CPUInfer singleton management
    - Basic configuration validation

    This class is shared between BaseMoEWrapper (inference) and BaseSFTMoEWrapper (SFT).
    """

    _cpu_infer_instance = None

    @classmethod
    def _get_cpu_infer(
        cls,
        cpuinfer_threads: int,
        threadpool_count: int,
        numa_nodes=None,
    ):
        """
        Get or create the CPUInfer singleton instance.

        Args:
            cpuinfer_threads: Total number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            numa_nodes: Explicit list of NUMA node IDs. If None, defaults to sequential.

        Returns:
            CPUInfer singleton instance
        """
        if cls._cpu_infer_instance is None:
            worker_config = kt_kernel_ext.WorkerPoolConfig()

            if numa_nodes is not None:
                if len(numa_nodes) != threadpool_count:
                    raise ValueError(
                        f"numa_nodes length ({len(numa_nodes)}) must match " f"threadpool_count ({threadpool_count})"
                    )
                subpool_numa_map = list(numa_nodes)
            else:
                subpool_numa_map = list(range(threadpool_count))
            subpool_thread_count = [
                cpuinfer_threads // threadpool_count + (1 if i < cpuinfer_threads % threadpool_count else 0)
                for i in range(threadpool_count)
            ]

            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            cls._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)

        return cls._cpu_infer_instance

    @staticmethod
    def _validate_base_config(
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts_per_tok: int,
    ) -> None:
        """
        Validate basic configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if moe_intermediate_size <= 0:
            raise ValueError(f"moe_intermediate_size must be positive, got {moe_intermediate_size}")
        if num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok must be positive, got {num_experts_per_tok}")
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed " f"num_experts ({num_experts})"
            )


class BaseMoEWrapper(_MoEBase, ABC):
    """
    Base class for MoE CPU inference operations.
    Provides common functionality for all backend implementations.
    """

    _layer_has_pending_deferred: Dict[int, bool] = {}
    # MESH class state and helpers are installed from kt_kernel.python.utils.mesh.runtime_helpers

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "AMXINT4",
        numa_nodes: Optional[List[int]] = None,
        weight_strategy: str = "auto",
        max_tier0_experts: Optional[int] = None,
        num_moe_layers: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        """
        Initialize base MoE Wrapper.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            gpu_experts_mask: Boolean mask indicating which experts are on GPU.
                              Shape: [num_experts], dtype: torch.bool.
                              mask[i] = True means expert i is on GPU.
                              If None, all experts are on CPU.
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA subpools
            numa_nodes: Explicit NUMA node IDs for the CPU subpools. If None,
                        use detected NUMA nodes in ascending order.
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer on this layer. Defaults to 0 (no defer).
            method: Backend method string
            weight_strategy: Weight residency strategy. "auto" chooses between
                             full-resident legacy mode and mmap+tier0 adaptive mode,
                             "legacy" forces malloc+copy, "tiered" forces mmap baseline
                             with hot experts promoted into Tier0 NUMA buffers.
            max_tier0_experts: Maximum number of expert IDs promoted to Tier 0.
                               Defaults to an auto-derived value based on the
                               current cgroup/host memory scope when omitted.
            num_moe_layers: Total number of MoE layers in the model. Used when
                            auto Tier0 sizing is enabled. If None, estimated
                            from registered layers.
            swiglu_limit: MXFP4 SwiGLU clamp limit. 0.0 disables the clamp.
        """
        self.layer_idx = layer_idx
        self.num_moe_layers = int(num_moe_layers) if num_moe_layers is not None else None
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.cpuinfer_threads = cpuinfer_threads
        self.threadpool_count = threadpool_count

        self._configure_base_runtime(
            kt_kernel_ext,
            method=method,
            weight_strategy=weight_strategy,
            max_tier0_experts=max_tier0_experts,
            num_moe_layers=num_moe_layers,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            numa_nodes=numa_nodes,
        )

        # Process gpu_experts_mask: convert to bool tensor on CPU, pinned memory for async copy
        # This mask is shared between C and Python (C uses uint8_t*), both can read/write it
        if gpu_experts_mask is None:
            # No GPU experts - all experts on CPU
            self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu", pin_memory=_PIN_MEMORY)
        else:
            # Create a new pinned tensor and copy data into it
            self.gpu_experts_mask = torch.empty(num_experts, dtype=torch.bool, device="cpu", pin_memory=_PIN_MEMORY)
            self.gpu_experts_mask.copy_(gpu_experts_mask)

        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())

        # GPU copy for mask operations in forward pass (e.g., mask_cpu_expert_ids)
        # This will be lazily initialized when needed
        self._gpu_experts_mask_gpu: Optional[torch.Tensor] = None
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpu_save = cpu_save
        env_max_deferred_experts = os.environ.get("KT_MAX_DEFERRED_EXPERTS_PER_TOKEN")
        if env_max_deferred_experts is not None:
            try:
                max_deferred_experts_per_token = int(env_max_deferred_experts)
            except ValueError:
                print(
                    f"[KTDeferredExperts] ignoring invalid KT_MAX_DEFERRED_EXPERTS_PER_TOKEN="
                    f"{env_max_deferred_experts!r}; using {max_deferred_experts_per_token or 0}"
                )
        self.max_deferred_experts_per_token = (
            int(max_deferred_experts_per_token) if max_deferred_experts_per_token is not None else 0
        )
        print(
            f"[KTDeferredExperts] layer={layer_idx} max_deferred_experts_per_token={self.max_deferred_experts_per_token}"
        )

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        self.method = method
        # V4-Flash 2604B SwiGLU clamp limit; 0.0 = disabled. NativeMoEWrapper
        # (MXFP4 path) reads this in load_weights() and writes it into
        # MOEConfig.swiglu_limit. Other backends ignore it (C++ act_fn skips
        # the clamp branch when limit==0). Origin: kt-sglang 耦合.
        self.swiglu_limit = float(swiglu_limit)

        # Backend-specific initialization happens in subclasses
        self.moe = None
        self._closed = False
        BaseMoEWrapper._wrappers_by_layer[self.layer_idx] = self
        BaseMoEWrapper._active_wrapper_count += 1

    @abstractmethod
    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        """
        Load and quantize weights from BF16/FP16 tensors (online quantization).

        Args:
            gate_proj: Gate projection weights [num_experts, intermediate_size, hidden_size]
            up_proj: Up projection weights [num_experts, intermediate_size, hidden_size]
            down_proj: Down projection weights [num_experts, hidden_size, intermediate_size]
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

    def select_deferred_experts(
        self,
        expert_ids: torch.Tensor,
        expert_scores: torch.Tensor,
        protected_k: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, topk = expert_ids.shape
        device = expert_ids.device

        protected_k = max(0, min(int(protected_k), topk))
        if protected_k == 0:
            deferred_ids = expert_ids.clone()
            immediate_ids = torch.full_like(expert_ids, -1)
            return immediate_ids, deferred_ids

        topk_result = torch.topk(expert_scores, k=protected_k, dim=-1, largest=True, sorted=False)
        protected_indices = topk_result.indices
        protected_ids = torch.gather(expert_ids, -1, protected_indices)

        protected_flag = torch.zeros((self.num_experts,), dtype=torch.int32, device=device)
        protected_flag.scatter_(0, protected_ids.reshape(-1), 1)

        protected_mask_flat = torch.gather(protected_flag, 0, expert_ids.reshape(-1)).ne(0)
        protected_mask = protected_mask_flat.view(batch, topk)

        immediate_ids = expert_ids.clone().masked_fill(~protected_mask, -1)
        deferred_ids = expert_ids.clone().masked_fill(protected_mask, -1)

        return immediate_ids, deferred_ids

    # MESH CPUInfer and runtime helpers are installed from kt_kernel.python.utils.mesh.runtime_helpers
    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
        router_scores: Optional[torch.Tensor] = None,
    ):
        """
        Submit forward inference task to CPU (non-blocking).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization
        """
        return self._mesh_submit_forward_impl(hidden_states, topk_ids, topk_weights, cuda_stream, router_scores)

    def sync_forward(self, hidden_states: torch.Tensor, topk_ids_or_stream=None, cuda_stream=None) -> torch.Tensor:
        """
        Synchronize and retrieve forward inference results.

        Args:
            hidden_states: Original input hidden states (for getting buffer)
            topk_ids: Top-k expert IDs from this forward pass (for recording activations)
            cuda_stream: CUDA stream for synchronization

        Returns:
            output_gpu: Output tensor on GPU
        """
        return self._mesh_sync_forward_impl(hidden_states, topk_ids_or_stream, cuda_stream)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
        router_scores: Optional[torch.Tensor] = None,
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
        return self._mesh_forward_impl(hidden_states, topk_ids, topk_weights, cuda_stream, router_scores)

    @staticmethod
    def set_capture_batch_sizes(capture_bs: List[int]):
        """
        Set batch sizes to capture and cache buffers for.

        This allows pre-allocation of CPU buffers for specific batch sizes,
        improving performance by avoiding buffer re-allocation during inference.

        Args:
            capture_bs: List of batch sizes to capture (e.g., [1, 2, 4, 8, 16])

        Example:
            >>> BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])
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


from .utils.mesh.runtime_helpers import install_base_moe_helpers as _install_mesh_base_moe_helpers

_install_mesh_base_moe_helpers(BaseMoEWrapper, KExpertsCPUBuffer, _PIN_MEMORY)
