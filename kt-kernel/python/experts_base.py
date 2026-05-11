# Base classes for MoE CPU inference operations
# SPDX-License-Identifier: Apache-2.0

"""
Base infrastructure for CPU-based MoE inference.

This module contains base classes and utilities shared across all backend implementations.
"""

from __future__ import annotations

import torch
import json
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import os
import ctypes
import numpy as np
from pathlib import Path

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

    _cpu_infer_instance = None
    _cpu_infer_signature: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
    _active_wrapper_count: int = 0
    _layer_has_pending_deferred: Dict[int, bool] = {}
    _tiered_provider = None  # Singleton TieredWeightProvider
    _tiered_provider_signature: Optional[Tuple[int, int, str, str]] = None
    _prev_topk_ids_by_layer: Dict[int, np.ndarray] = {}
    _wrappers_by_layer: Dict[int, "BaseMoEWrapper"] = {}
    _full_gate_batch_states: Dict[Tuple[str, int, int, int], dict] = {}
    _full_gate_skip_logged: set = set()
    _mesh_bootstrap_done: bool = False
    _mesh_bootstrap_log_count: int = 0
    _provider_unsupported_logged: set = set()
    _cpu_infer_stream_fallback_logged: bool = False
    # Backends whose C++ MOE objects expose promote/demote hooks for Tier0 management.
    _provider_backends = frozenset({"LLAMAFILE", "BF16", "AMXINT4", "AMXINT8", "MOE_INT4", "MOE_INT8"})
    _debug_moe_layers: Optional[set] = None
    _debug_logged_layers: set = set()

    @classmethod
    def _should_debug_layer(cls, layer_idx: int) -> bool:
        raw = os.environ.get("KT_DEBUG_MOE_LAYERS")
        if not raw:
            return False
        if cls._debug_moe_layers is None:
            values = set()
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    values.add(int(part))
                except ValueError:
                    pass
            cls._debug_moe_layers = values
        return layer_idx in cls._debug_moe_layers and layer_idx not in cls._debug_logged_layers

    @staticmethod
    def _debug_stats_tensor(t: torch.Tensor) -> dict:
        x = t.detach().float()
        finite = torch.isfinite(x)
        return {
            "shape": list(x.shape),
            "min": float(x.min().item()) if x.numel() else 0.0,
            "max": float(x.max().item()) if x.numel() else 0.0,
            "mean": float(x.mean().item()) if x.numel() else 0.0,
            "std": float(x.std().item()) if x.numel() > 1 else 0.0,
            "l2": float(torch.linalg.vector_norm(x).item()) if x.numel() else 0.0,
            "finite_ratio": float(finite.float().mean().item()) if x.numel() else 1.0,
        }

    def _debug_log_moe_once(
        self,
        hidden_states: torch.Tensor,
        topk_ids: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        output_tensor: torch.Tensor,
    ) -> None:
        if not BaseMoEWrapper._should_debug_layer(self.layer_idx):
            return
        BaseMoEWrapper._debug_logged_layers.add(self.layer_idx)
        line = {
            "layer": self.layer_idx,
            "weight_strategy": self.weight_strategy,
            "hidden": self._debug_stats_tensor(hidden_states),
            "output": self._debug_stats_tensor(output_tensor),
        }
        if topk_ids is not None and torch.is_tensor(topk_ids):
            ids = topk_ids.detach().cpu().reshape(-1)
            line["topk_ids_head"] = ids[:32].tolist()
            line["topk_ids_unique"] = int(ids.unique().numel()) if ids.numel() else 0
        if topk_weights is not None and torch.is_tensor(topk_weights):
            line["topk_weights"] = self._debug_stats_tensor(topk_weights)
            line["topk_weights_head"] = topk_weights.detach().cpu().reshape(-1)[:32].tolist()
        debug_path = os.environ.get("KT_DEBUG_MOE_LOG", "/tmp/kt_moe_debug.log")
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        dump_dir = os.environ.get("KT_DEBUG_MOE_DUMP_DIR")
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"layer_{self.layer_idx}.pt")
            payload = {
                "layer": self.layer_idx,
                "weight_strategy": self.weight_strategy,
                "hidden_states": hidden_states.detach().cpu(),
                "topk_ids": topk_ids.detach().cpu() if torch.is_tensor(topk_ids) else None,
                "topk_weights": topk_weights.detach().cpu() if torch.is_tensor(topk_weights) else None,
                "output_tensor": output_tensor.detach().cpu(),
            }
            torch.save(payload, dump_path)

    @staticmethod
    def _detect_available_numa_node_ids() -> List[int]:
        """Return actual Linux NUMA node ids instead of assuming 0..N-1."""
        node_root = Path("/sys/devices/system/node")
        node_ids: List[int] = []
        if node_root.exists():
            for entry in node_root.iterdir():
                if entry.name.startswith("node") and entry.name[4:].isdigit():
                    node_ids.append(int(entry.name[4:]))
        if not node_ids:
            return [0]
        return sorted(set(node_ids))

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
        numa_nodes: Optional[List[int]] = None,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "AMXINT4",
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

        # Allow environment override so runtime experiments can force
        # legacy/tiered without needing a dedicated launch_server flag.
        env_weight_strategy = os.environ.get("KT_WEIGHT_STRATEGY")
        if env_weight_strategy:
            weight_strategy = env_weight_strategy

        requested_weight_strategy = weight_strategy or "auto"
        from .utils.weight_provider import (
            backend_supports_tiered_strategy,
            normalize_residency_policy_name,
            resolve_backend_weight_strategy,
        )

        backend_has_tiered = backend_supports_tiered_strategy(method)
        if requested_weight_strategy in {"auto", "tiered"}:
            from .utils.weight_provider import get_available_ram_bytes

            effective_num_layers = num_moe_layers or 60
            available_ram_bytes = get_available_ram_bytes()
            resolved_strategy, estimated_model_bytes, total_ram_bytes = resolve_backend_weight_strategy(
                method,
                requested_weight_strategy,
                num_layers=effective_num_layers,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                available_ram_bytes=available_ram_bytes,
            )
            self.weight_strategy = resolved_strategy
            if requested_weight_strategy == "auto":
                if backend_has_tiered:
                    tiered_mode = (
                        "mmap + adaptive Tier0" if method in BaseMoEWrapper._provider_backends else "mmap baseline"
                    )
                    print(
                        "[TieredWeightProvider] weight_strategy=auto -> "
                        f"{self.weight_strategy} "
                        f"(backend={method}, mode={tiered_mode}, "
                        f"estimated_moe_weights={estimated_model_bytes / 1024**3:.1f}GB, "
                        f"available_ram={available_ram_bytes / 1024**3:.1f}GB, "
                        f"total_ram={total_ram_bytes / 1024**3:.1f}GB)"
                    )
                else:
                    print(
                        "[TieredWeightProvider] weight_strategy=auto -> legacy "
                        f"(backend={method} uses resident weights; mmap-backed tiered loading is unavailable)"
                    )
            elif not backend_has_tiered and self.weight_strategy != requested_weight_strategy:
                print(
                    "[TieredWeightProvider] weight_strategy=tiered requested but downgraded to legacy "
                    f"(backend={method} does not support mmap-backed expert weights)"
                )
        else:
            self.weight_strategy = requested_weight_strategy

        print(
            "[KTWeightStrategy] "
            f"env={env_weight_strategy!r} requested={requested_weight_strategy!r} "
            f"resolved={self.weight_strategy!r} method={method!r} layer={layer_idx}"
        )

        env_residency_policy = os.environ.get("KT_RESIDENCY_POLICY")
        self.residency_policy = normalize_residency_policy_name(env_residency_policy or "baseline")
        print(
            "[KTResidencyPolicy] "
            f"env={env_residency_policy!r} resolved={self.residency_policy!r} "
            f"method={method!r} layer={layer_idx}"
        )

        # Read io_backend from environment
        env_io_backend = os.environ.get("KT_IO_BACKEND", "MMAP").upper()
        self.io_backend = env_io_backend
        print(f"[KTIOBackend] io_backend={self.io_backend} method={method!r} layer={layer_idx}")

        # Read enable_cache_stats from environment
        env_enable_cache_stats = os.environ.get("KT_ENABLE_CACHE_STATS", "0")
        self.enable_cache_stats = env_enable_cache_stats in ("1", "true", "True", "TRUE")
        if self.enable_cache_stats:
            print(f"[KTCacheStats] Cache statistics collection enabled for layer={layer_idx}")

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
        # MOEConfig.swiglu_limit. Other backends ignore it.
        self.swiglu_limit = float(swiglu_limit)

        if max_tier0_experts is not None:
            parsed_max_tier0 = int(max_tier0_experts)
            max_tier0_experts = None if parsed_max_tier0 <= 0 else parsed_max_tier0
        self.max_tier0_experts = int(max_tier0_experts) if max_tier0_experts is not None else 0
        self.max_resident_experts = (
            int(os.environ["KT_MAX_RESIDENT_EXPERTS"]) if "KT_MAX_RESIDENT_EXPERTS" in os.environ else 0
        )

        if threadpool_count <= 0:
            raise ValueError(f"threadpool_count must be positive, got {threadpool_count}")
        if cpuinfer_threads < threadpool_count:
            raise ValueError(f"cpuinfer_threads ({cpuinfer_threads}) must be >= threadpool_count ({threadpool_count})")

        if numa_nodes is not None:
            available_numa_ids = [int(node) for node in numa_nodes]
            if len(available_numa_ids) == 0:
                raise ValueError("numa_nodes must not be empty when provided")
        else:
            available_numa_ids = BaseMoEWrapper._detect_available_numa_node_ids()
        if threadpool_count > len(available_numa_ids):
            raise ValueError(
                f"threadpool_count ({threadpool_count}) exceeds detected NUMA nodes " f"{available_numa_ids}"
            )
        if numa_nodes is not None and threadpool_count != len(available_numa_ids):
            raise ValueError(
                f"threadpool_count ({threadpool_count}) must match explicit numa_nodes {available_numa_ids}"
            )

        subpool_numa_map = available_numa_ids[:threadpool_count]
        subpool_thread_count = [
            cpuinfer_threads // threadpool_count + (1 if i < cpuinfer_threads % threadpool_count else 0)
            for i in range(threadpool_count)
        ]
        runtime_signature = (tuple(subpool_numa_map), tuple(subpool_thread_count))

        # Initialize CPU inference engine (singleton per worker topology).
        if BaseMoEWrapper._cpu_infer_instance is None:
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            BaseMoEWrapper._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)
            BaseMoEWrapper._cpu_infer_signature = runtime_signature
        elif BaseMoEWrapper._cpu_infer_signature != runtime_signature:
            if BaseMoEWrapper._active_wrapper_count > 0:
                raise RuntimeError(
                    "CPUInfer is already initialized with a different NUMA/thread layout. "
                    "Destroy existing MoE wrappers before creating a new runtime topology."
                )
            BaseMoEWrapper.reset_runtime_state(force=True)
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            BaseMoEWrapper._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)
            BaseMoEWrapper._cpu_infer_signature = runtime_signature

        self.cpu_infer = BaseMoEWrapper._cpu_infer_instance

        # Initialize tiered weight provider only for backends that truly support
        # mmap baseline + Tier0 promotion.
        # NOTE: self._provider is initially None for ALL backends.
        provider_backend = method in BaseMoEWrapper._provider_backends
        tiered_policy_config_raw = os.environ.get("KT_RESIDENCY_POLICY_CONFIG", "")
        if provider_backend and self.weight_strategy == "tiered" and BaseMoEWrapper._tiered_provider is None:
            from .utils.weight_provider import (
                TieredWeightProvider,
                compute_max_tier0_experts,
                get_available_ram_bytes,
                method_bytes_per_element,
                resolve_auto_tier0_budget_bytes,
            )

            # Fallback to environment variables if parameters not provided
            if max_tier0_experts is None and "KT_MAX_TIER0_EXPERTS" in os.environ:
                raw_max_tier0 = os.environ["KT_MAX_TIER0_EXPERTS"].strip().lower()
                if raw_max_tier0 in ("", "auto", "none", "0", "-1"):
                    max_tier0_experts = None
                else:
                    parsed_max_tier0 = int(raw_max_tier0)
                    max_tier0_experts = None if parsed_max_tier0 <= 0 else parsed_max_tier0

            # Auto-detect Tier0 budget from the effective memory scope when no
            # explicit expert cap is provided.
            if max_tier0_experts is None:
                available_gb = get_available_ram_bytes() / (1024**3)
                effective_num_layers = num_moe_layers or 60
                from .utils.weight_provider import estimate_model_weight_bytes

                model_bytes = estimate_model_weight_bytes(
                    num_layers=effective_num_layers,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    intermediate_size=moe_intermediate_size,
                    bytes_per_element=method_bytes_per_element(method),
                )
                safety_gb = float(
                    os.environ.get("KT_TIER0_AUTO_SAFETY_GB", "3" if self.io_backend == "IOURING" else "4")
                )
                safety_bytes = int(safety_gb * 1024**3)
                if self.io_backend == "IOURING":
                    # io_uring + O_DIRECT has no mmap page-cache copy to reserve
                    # for. Use the cgroup-visible headroom directly, keeping only
                    # a KV-cache/runtime safety margin for SGLang, CUDA metadata,
                    # and transient request buffers.
                    tier0_bytes = max(0, get_available_ram_bytes() - safety_bytes)
                else:
                    tier0_bytes = resolve_auto_tier0_budget_bytes(
                        model_bytes=model_bytes,
                        safety_bytes=safety_bytes,
                    )
                effective_max_tier0 = compute_max_tier0_experts(
                    tier0_memory_bytes=tier0_bytes,
                    num_layers=effective_num_layers,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    intermediate_size=moe_intermediate_size,
                    bytes_per_element=method_bytes_per_element(method),
                )
                print(
                    f"[TieredWeightProvider] Auto-adapted tier0 budget: "
                    f"backend={self.io_backend}, available_ram={available_gb:.1f}GB, "
                    f"safety={safety_gb:.1f}GB, tier0={tier0_bytes / (1024**3):.1f}GB"
                )
                print(
                    f"[TieredWeightProvider] auto max_tier0_experts={effective_max_tier0} "
                    f"(~{effective_max_tier0 * effective_num_layers * 3 * hidden_size * moe_intermediate_size * method_bytes_per_element(method) / 1024**3:.1f}GB)"
                )
            else:
                effective_max_tier0 = int(max_tier0_experts)
                print(
                    f"[TieredWeightProvider] explicit max_tier0_experts={effective_max_tier0} "
                    f"(~{effective_max_tier0 * (num_moe_layers or 60) * 3 * hidden_size * moe_intermediate_size * method_bytes_per_element(method) / 1024**3:.1f}GB)"
                )

            self.max_tier0_experts = effective_max_tier0

            BaseMoEWrapper._tiered_provider = TieredWeightProvider(
                num_experts=num_experts,
                num_layers=1,  # dict-based storage, supports arbitrary layer_idx
                max_tier0_experts=effective_max_tier0,
                residency_policy=self.residency_policy,
            )
            BaseMoEWrapper._tiered_provider_signature = (
                int(num_experts),
                int(effective_max_tier0),
                self.residency_policy,
                tiered_policy_config_raw,
            )
        elif provider_backend and self.weight_strategy == "tiered" and BaseMoEWrapper._tiered_provider is not None:
            provider = BaseMoEWrapper._tiered_provider
            desired_max_tier0 = provider.max_tier0_experts
            if max_tier0_experts is not None:
                desired_max_tier0 = int(max_tier0_experts)

            desired_provider_signature = (
                int(num_experts),
                int(desired_max_tier0),
                self.residency_policy,
                tiered_policy_config_raw,
            )
            if BaseMoEWrapper._tiered_provider_signature != desired_provider_signature:
                if BaseMoEWrapper._active_wrapper_count > 0:
                    raise RuntimeError(
                        "TieredWeightProvider is already initialized with different tiered settings. "
                        "Destroy existing MoE wrappers before loading another MoE topology."
                    )
                provider.stop_promotion_thread()
                BaseMoEWrapper._tiered_provider = None
                from .utils.weight_provider import TieredWeightProvider

                BaseMoEWrapper._tiered_provider = TieredWeightProvider(
                    num_experts=num_experts,
                    num_layers=1,
                    max_tier0_experts=desired_max_tier0,
                    residency_policy=self.residency_policy,
                )
                BaseMoEWrapper._tiered_provider_signature = desired_provider_signature
            self.max_tier0_experts = BaseMoEWrapper._tiered_provider.max_tier0_experts
        self._provider = None  # Set by subclass via _register_moe_with_provider()

        # NOTE: promotion thread is NOT started here — it's started lazily
        # in register_moe() after the first MOE object is registered.
        # Starting here would run the thread with an empty moe_refs dict.

        # Backend-specific initialization happens in subclasses
        self.moe = None
        self._closed = False
        BaseMoEWrapper._wrappers_by_layer[self.layer_idx] = self
        BaseMoEWrapper._active_wrapper_count += 1

    def _register_moe_with_provider(self):
        """
        Register self.moe with the tiered weight provider and enable tiered management.

        Call this from subclass load_weights() AFTER self.moe is created.
        Only backends whose C++ MOE objects support promote_expert/demote_expert
        should call this (currently Llamafile, BF16, AMXINT4, AMXINT8, MOE_INT4, and MOE_INT8 tiered).

        This sets self._provider, which enables prefetch and record_activations
        in submit_forward/sync_forward.
        """
        provider = BaseMoEWrapper._tiered_provider
        if provider is not None and self.moe is not None:
            required_hooks = ("promote_expert", "demote_expert", "is_expert_promoted")
            missing_hooks = tuple(name for name in required_hooks if not hasattr(self.moe, name))
            if missing_hooks:
                moe_name = type(self.moe).__name__
                log_key = (moe_name, missing_hooks)
                if log_key not in BaseMoEWrapper._provider_unsupported_logged:
                    print(
                        f"[TieredWeightProvider] disabled for {moe_name}: "
                        f"missing C++ hooks {', '.join(missing_hooks)}"
                    )
                    BaseMoEWrapper._provider_unsupported_logged.add(log_key)
                return
            provider.register_moe(
                self.layer_idx,
                self.moe,
                gpu_experts_mask=self.gpu_experts_mask.detach().cpu().numpy(),
            )
            self._provider = provider

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

    def _submit_cpuinfer_task(self, task, cuda_stream=None):
        """Submit a CPUInfer task, tolerating CPU-only extension builds."""
        if cuda_stream is None:
            self.cpu_infer.submit(task)
            return

        submit_with_stream = getattr(self.cpu_infer, "submit_with_cuda_stream", None)
        if submit_with_stream is not None:
            submit_with_stream(cuda_stream, task)
            return

        self._sync_external_cuda_stream(cuda_stream)
        self.cpu_infer.submit(task)

    def _sync_cpuinfer(self, allow_pending: int = 0, cuda_stream=None):
        """Synchronize CPUInfer, using stream-aware hooks when available."""
        if cuda_stream is None:
            self.cpu_infer.sync(allow_pending)
            return

        sync_with_stream = getattr(self.cpu_infer, "sync_with_cuda_stream", None)
        if sync_with_stream is not None:
            sync_with_stream(cuda_stream, allow_pending)
            return

        self.cpu_infer.sync(allow_pending)

    @classmethod
    def _sync_external_cuda_stream(cls, cuda_stream):
        if not cls._cpu_infer_stream_fallback_logged:
            print("[CPUInfer] submit_with_cuda_stream unavailable; " "falling back to synchronous CUDA stream handoff")
            cls._cpu_infer_stream_fallback_logged = True
        torch.cuda.ExternalStream(cuda_stream).synchronize()

    def _mesh_score_transform_id(self) -> int:
        raw = os.environ.get("KT_MESH_SCORE_TRANSFORM", "softmax").strip().lower()
        if raw in ("none", "identity", "raw", "0"):
            return 0
        if raw in ("sigmoid", "2"):
            return 2
        return 1

    def _mesh_resident_capacity(self) -> int:
        if int(self.max_resident_experts) > 0:
            return int(self.max_resident_experts)
        return int(self.max_tier0_experts)

    def _mesh_current_cpu_expert_count(self, cols: Optional[int] = None) -> int:
        width = int(self.num_experts if cols is None else min(int(cols), int(self.num_experts)))
        if width <= 0:
            return 0
        src = getattr(self, "gpu_experts_mask", None)
        if src is None or src.numel() <= 0:
            return width
        n = min(width, int(src.numel()))
        gpu_count = int(src[:n].sum().item()) if n > 0 else 0
        return max(0, width - gpu_count)

    def _mesh_cpu_expert_mask(self, cols: int) -> torch.Tensor:
        """Return a CPU bool mask where True means MESH may manage this expert."""
        width = max(0, int(cols))
        mask = torch.zeros(width, dtype=torch.bool, device="cpu")
        if width <= 0:
            return mask
        valid = min(width, int(self.num_experts))
        if valid <= 0:
            return mask

        src = getattr(self, "gpu_experts_mask", None)
        if src is None or src.numel() <= 0:
            mask[:valid] = True
            return mask

        n = min(valid, int(src.numel()))
        if n > 0:
            mask[:n].copy_(~src[:n].to(dtype=torch.bool), non_blocking=False)
        if valid > n:
            mask[n:valid] = True
        return mask

    def _mesh_full_gate_observation_enabled(self) -> bool:
        if os.environ.get("KT_MESH_FULL_GATE", "1") in ("0", "false", "False", "FALSE"):
            return False
        if os.environ.get("KT_MESH_LOOKAHEAD", "1") in ("0", "false", "False", "FALSE"):
            return False
        try:
            if float(os.environ.get("KT_MESH_LOOKAHEAD_WEIGHT", "1.0")) <= 0.0:
                return False
        except ValueError:
            pass
        current_gpu_experts = int(self.gpu_experts_mask.sum().item())
        cpu_expert_count = self._mesh_current_cpu_expert_count()
        resident_capacity = self._mesh_resident_capacity()
        if cpu_expert_count <= 0 or (resident_capacity > 0 and resident_capacity >= cpu_expert_count):
            log_key = (int(self.num_experts), current_gpu_experts, int(resident_capacity))
            if log_key not in BaseMoEWrapper._full_gate_skip_logged:
                print(
                    "[MESHFullGate] skip full-router Heat observe: "
                    f"resident_capacity={resident_capacity} cpu_experts={cpu_expert_count} "
                    f"num_experts={self.num_experts} gpu_experts={current_gpu_experts}"
                )
                BaseMoEWrapper._full_gate_skip_logged.add(log_key)
            return False
        return True

    def _mesh_full_gate_batched_enabled(self) -> bool:
        return os.environ.get("KT_MESH_FULL_GATE_BATCHED", "1") not in ("0", "false", "False", "FALSE")

    @staticmethod
    def _cuda_graph_capture_active() -> bool:
        """Return True while SGLang is recording a CUDA graph warmup forward."""
        if not torch.cuda.is_available():
            return False
        checker = getattr(torch.cuda, "is_current_stream_capturing", None)
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    @staticmethod
    def _discard_full_gate_batch_slot(state: dict, slot: int) -> None:
        try:
            state["seen"][slot].clear()
            state["gpu_seen"][slot].clear()
            state["decode_seen"][slot] = False
            state["transforms"][slot].clear()
            state["keepalive"][slot] = None
            state["write_slot"] = (slot + 1) % len(state["seen"])
        except Exception:
            return

    def _mesh_total_moe_layers(self) -> int:
        registered_layers = (
            max(BaseMoEWrapper._wrappers_by_layer.keys()) + 1 if BaseMoEWrapper._wrappers_by_layer else 0
        )
        if self.num_moe_layers is not None and self.num_moe_layers > 0:
            return max(self.num_moe_layers, registered_layers, self.layer_idx + 1)
        if registered_layers > 0:
            return max(registered_layers, self.layer_idx + 1)
        return self.layer_idx + 1

    def _mesh_last_registered_layer_idx(self) -> int:
        if BaseMoEWrapper._wrappers_by_layer:
            return max(BaseMoEWrapper._wrappers_by_layer.keys())
        return self.layer_idx

    def _mesh_last_full_gate_layer_idx(self) -> int:
        last_enabled = -1
        for layer_idx, wrapper in BaseMoEWrapper._wrappers_by_layer.items():
            if wrapper._mesh_full_gate_observation_enabled():
                last_enabled = max(last_enabled, layer_idx)
        return last_enabled if last_enabled >= 0 else self._mesh_last_registered_layer_idx()

    def _prepare_router_score_vector(self, router_scores: torch.Tensor) -> Tuple[Optional[torch.Tensor], int, int]:
        scores = router_scores.detach()
        if scores.dim() == 1:
            scores = scores.view(1, -1)
        elif scores.dim() > 2:
            scores = scores.reshape(-1, scores.shape[-1])
        rows = int(scores.shape[0])
        cols = int(scores.shape[1])
        if rows <= 0 or cols <= 0:
            return None, 0, 0

        max_elements = int(os.environ.get("KT_MESH_FULL_GATE_MAX_ELEMENTS", "65536"))
        if max_elements > 0 and rows * cols > max_elements:
            return None, 0, 0

        if scores.dtype != torch.float32:
            scores = scores.float()
        if not scores.is_contiguous():
            scores = scores.contiguous()

        score_transform = self._mesh_score_transform_id()
        if rows == 1:
            return scores.view(-1), cols, score_transform

        if score_transform == 1:
            vector = torch.softmax(scores, dim=-1).amax(dim=0)
            score_transform = 0
        elif score_transform == 2:
            vector = torch.sigmoid(scores).amax(dim=0)
            score_transform = 0
        else:
            vector = scores.amax(dim=0)
        if not vector.is_contiguous():
            vector = vector.contiguous()
        return vector, cols, score_transform

    @classmethod
    def _full_gate_batch_key(cls, device: torch.device, total_layers: int, cols: int) -> Tuple[str, int, int, int]:
        return (device.type, -1 if device.index is None else int(device.index), int(total_layers), int(cols))

    def _get_full_gate_batch_state(self, vector: torch.Tensor, total_layers: int, cols: int) -> dict:
        key = BaseMoEWrapper._full_gate_batch_key(vector.device, total_layers, cols)
        state = BaseMoEWrapper._full_gate_batch_states.get(key)
        depth = max(4, KExpertsCPUBuffer.buffer_depth)
        if state is None:
            cpu = torch.empty((depth, total_layers, cols), dtype=torch.float32, device="cpu", pin_memory=_PIN_MEMORY)
            gpu = None
            if vector.device.type == "cuda":
                gpu = torch.empty((depth, total_layers, cols), dtype=torch.float32, device=vector.device)
            state = {
                "cpu": cpu,
                "gpu": gpu,
                "write_slot": 0,
                "seen": [set() for _ in range(depth)],
                "gpu_seen": [set() for _ in range(depth)],
                "decode_seen": [False for _ in range(depth)],
                "transforms": [{} for _ in range(depth)],
                "keepalive": [None for _ in range(depth)],
            }
            BaseMoEWrapper._full_gate_batch_states[key] = state
        elif vector.device.type == "cuda" and state.get("gpu") is None:
            state["gpu"] = torch.empty((depth, total_layers, cols), dtype=torch.float32, device=vector.device)
        if "decode_seen" not in state:
            state["decode_seen"] = [False for _ in range(len(state["seen"]))]
        return state

    def _mesh_full_gate_mask_batch(self, seen: List[int], cols: int) -> torch.Tensor:
        masks = torch.empty((len(seen), cols), dtype=torch.uint8, device="cpu", pin_memory=_PIN_MEMORY)
        for row, layer_idx in enumerate(seen):
            masks[row].zero_()
            wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
            if wrapper is None:
                continue
            src = getattr(wrapper, "gpu_experts_mask", None)
            if src is None:
                continue
            n = min(cols, int(src.numel()))
            if n > 0:
                masks[row, :n].copy_(src[:n].to(dtype=torch.uint8), non_blocking=False)
        return masks

    def _mesh_full_gate_batch_owner(self, seen: List[int]):
        if self.layer_idx in seen and self.moe is not None and hasattr(self.moe, "observe_router_scores_batch_task"):
            return self
        for layer_idx in reversed(seen):
            wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
            if (
                wrapper is not None
                and wrapper.moe is not None
                and hasattr(wrapper.moe, "observe_router_scores_batch_task")
            ):
                return wrapper
        return None

    def _mesh_bootstrap_prefetch_enabled(self) -> bool:
        return (
            self.io_backend == "IOURING"
            and self._env_flag("KT_MESH_BOOTSTRAP_PREFETCH", True)
            and self._mesh_resident_capacity() > 0
        )

    def _mesh_bootstrap_prefetch_budget(self, cpu_expert_count: int) -> int:
        if cpu_expert_count <= 0:
            return 0
        explicit = self._env_int("KT_MESH_BOOTSTRAP_PREFETCH_LIMIT", 0)
        capacity = self._mesh_resident_capacity()
        if capacity <= 0:
            return 0
        if explicit > 0:
            return min(explicit, cpu_expert_count)
        return min(capacity, cpu_expert_count)

    def _maybe_submit_mesh_bootstrap_from_full_gate_batch(
        self,
        state: dict,
        slot: int,
        seen: List[int],
        cols: int,
        cuda_stream,
        *,
        needs_cuda_sync: bool,
    ) -> None:
        if BaseMoEWrapper._mesh_bootstrap_done:
            return
        if not state.get("decode_seen", [False])[slot]:
            return
        if not self._env_flag("KT_MESH_BOOTSTRAP_PREFETCH", True):
            BaseMoEWrapper._mesh_bootstrap_done = True
            return

        if needs_cuda_sync and torch.cuda.is_available():
            if cuda_stream is not None:
                self._sync_external_cuda_stream(cuda_stream)
            else:
                torch.cuda.current_stream().synchronize()

        submitted_layers = 0
        candidate_total = 0
        for layer_idx in seen:
            wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
            if wrapper is None or wrapper.moe is None:
                continue
            if not wrapper._mesh_bootstrap_prefetch_enabled():
                continue

            cpu_mask = wrapper._mesh_cpu_expert_mask(cols)
            cpu_expert_count = int(cpu_mask.sum().item())
            budget = wrapper._mesh_bootstrap_prefetch_budget(cpu_expert_count)
            if budget <= 0:
                continue

            scores = state["cpu"][slot, layer_idx, :cols].detach().clone()
            scores[~cpu_mask] = float("-inf")
            candidate_scores, candidate_ids = torch.topk(scores, k=budget, largest=True, sorted=True)
            finite = torch.isfinite(candidate_scores)
            if not bool(finite.any().item()):
                continue
            candidate_ids = candidate_ids[finite].to(torch.long).contiguous()
            candidate_count = int(candidate_ids.numel())
            if candidate_count <= 0:
                continue

            protect_count = min(int(wrapper.num_experts_per_tok), candidate_count)
            protect_ids = candidate_ids[:protect_count].contiguous() if protect_count > 0 else None
            if wrapper._submit_iouring_prefetch(
                candidate_ids,
                candidate_count,
                protect_ids_cpu=protect_ids,
                protect_count=protect_count,
                max_to_submit=candidate_count,
                cuda_stream=cuda_stream,
                prefetch_kind=1,
            ):
                submitted_layers += 1
                candidate_total += candidate_count

        BaseMoEWrapper._mesh_bootstrap_done = True
        if submitted_layers > 0 and BaseMoEWrapper._mesh_bootstrap_log_count < 3:
            print(
                "[MESHBootstrap] submitted first-token CPU expert warm fill: "
                f"layers={submitted_layers} candidates={candidate_total} cols={cols}"
            )
            BaseMoEWrapper._mesh_bootstrap_log_count += 1

    def _flush_full_gate_batch(self, state: dict, slot: int, total_layers: int, cols: int, cuda_stream) -> None:
        seen = sorted(state["seen"][slot])
        if not seen:
            return
        if self._cuda_graph_capture_active():
            self._discard_full_gate_batch_slot(state, slot)
            return

        gpu_seen = state["gpu_seen"][slot]
        needs_cuda_sync = bool(gpu_seen)
        if state.get("gpu") is not None and gpu_seen:
            if len(gpu_seen) == len(seen):
                state["cpu"][slot].copy_(state["gpu"][slot], non_blocking=True)
            else:
                for layer_idx in sorted(gpu_seen):
                    state["cpu"][slot, layer_idx].copy_(state["gpu"][slot, layer_idx], non_blocking=True)

        batch_owner = self._mesh_full_gate_batch_owner(seen)
        if batch_owner is not None:
            batch_task = getattr(batch_owner.moe, "observe_router_scores_batch_task", None)
            batch_direct = getattr(batch_owner.moe, "observe_router_scores_batch", None)
            if batch_task is not None or batch_direct is not None:
                layer_indices = torch.tensor(seen, dtype=torch.int32, device="cpu")
                score_transforms = torch.tensor(
                    [
                        int(state["transforms"][slot].get(layer_idx, self._mesh_score_transform_id()))
                        for layer_idx in seen
                    ],
                    dtype=torch.int32,
                    device="cpu",
                )
                gpu_masks = self._mesh_full_gate_mask_batch(seen, cols)
                state["keepalive"][slot] = (layer_indices, score_transforms, gpu_masks)
                scores_ptr = int(state["cpu"][slot].data_ptr())
                if batch_task is not None:
                    task = batch_task(
                        scores_ptr,
                        cols,
                        cols,
                        int(layer_indices.data_ptr()),
                        int(score_transforms.data_ptr()),
                        int(gpu_masks.data_ptr()),
                        len(seen),
                    )
                    batch_owner._submit_cpuinfer_task(task, cuda_stream)
                else:
                    batch_owner._sync_external_cuda_stream(cuda_stream or torch.cuda.current_stream().cuda_stream)
                    batch_direct(
                        scores_ptr,
                        cols,
                        cols,
                        int(layer_indices.data_ptr()),
                        int(score_transforms.data_ptr()),
                        int(gpu_masks.data_ptr()),
                        len(seen),
                    )
                state["seen"][slot].clear()
                state["gpu_seen"][slot].clear()
                self._maybe_submit_mesh_bootstrap_from_full_gate_batch(
                    state,
                    slot,
                    seen,
                    cols,
                    cuda_stream,
                    needs_cuda_sync=needs_cuda_sync,
                )
                state["decode_seen"][slot] = False
                state["transforms"][slot].clear()
                state["write_slot"] = (slot + 1) % len(state["seen"])
                return

        for layer_idx in seen:
            wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
            if wrapper is None or wrapper.moe is None:
                continue
            observe_task = getattr(wrapper.moe, "observe_router_scores_task", None)
            observe_direct = getattr(wrapper.moe, "observe_router_scores", None)
            score_transform = int(state["transforms"][slot].get(layer_idx, self._mesh_score_transform_id()))
            scores_ptr = int(state["cpu"][slot, layer_idx].data_ptr())
            if observe_task is not None:
                task = observe_task(scores_ptr, 1, cols, score_transform)
                wrapper._submit_cpuinfer_task(task, cuda_stream)
            elif observe_direct is not None:
                wrapper._sync_external_cuda_stream(cuda_stream or torch.cuda.current_stream().cuda_stream)
                observe_direct(scores_ptr, 1, cols, score_transform)

        state["seen"][slot].clear()
        state["gpu_seen"][slot].clear()
        self._maybe_submit_mesh_bootstrap_from_full_gate_batch(
            state,
            slot,
            seen,
            cols,
            cuda_stream,
            needs_cuda_sync=needs_cuda_sync,
        )
        state["decode_seen"][slot] = False
        state["transforms"][slot].clear()
        state["write_slot"] = (slot + 1) % len(state["seen"])

    def _record_router_scores_for_token_batch(
        self,
        router_scores: torch.Tensor,
        cuda_stream,
        flush_on_last: bool = False,
        is_decode_token: bool = False,
    ) -> bool:
        if self._cuda_graph_capture_active():
            return True

        vector, cols, score_transform = self._prepare_router_score_vector(router_scores)
        if vector is None or cols <= 0:
            return True

        total_layers = max(self._mesh_total_moe_layers(), self.layer_idx + 1)
        state = self._get_full_gate_batch_state(vector, total_layers, cols)
        slot = int(state["write_slot"])
        if self.layer_idx >= total_layers:
            return True

        if vector.device.type == "cuda":
            state["gpu"][slot, self.layer_idx].copy_(vector, non_blocking=True)
            state["gpu_seen"][slot].add(self.layer_idx)
        else:
            state["cpu"][slot, self.layer_idx].copy_(vector, non_blocking=True)
        state["seen"][slot].add(self.layer_idx)
        if is_decode_token:
            state["decode_seen"][slot] = True
        state["transforms"][slot][self.layer_idx] = int(score_transform)

        if self.layer_idx == self._mesh_last_full_gate_layer_idx():
            if flush_on_last:
                self._flush_full_gate_batch(state, slot, total_layers, cols, cuda_stream)
            else:
                self._pending_full_gate_flush = (state, slot, total_layers, cols)
        return True

    def _flush_pending_full_gate_batch(self, cuda_stream) -> None:
        pending = getattr(self, "_pending_full_gate_flush", None)
        if pending is None:
            return
        self._pending_full_gate_flush = None
        state, slot, total_layers, cols = pending
        if self._cuda_graph_capture_active():
            self._discard_full_gate_batch_slot(state, slot)
            return
        self._flush_full_gate_batch(state, slot, total_layers, cols, cuda_stream)

    def observe_router_scores(self, router_scores: Optional[torch.Tensor], cuda_stream=None) -> None:
        """Feed the full per-token router score vector into the MESH Heat registry.

        The AMX MoE compute API only needs top-k ids/weights. MESH eviction can
        use the richer all-expert router vector when the caller provides it
        through this side channel.
        """
        if router_scores is None or not torch.is_tensor(router_scores):
            return
        if not self._mesh_full_gate_observation_enabled():
            return
        if self._cuda_graph_capture_active():
            return
        if self._mesh_full_gate_batched_enabled():
            self._record_router_scores_for_token_batch(router_scores, cuda_stream, flush_on_last=True)
            return
        observe_task = getattr(self.moe, "observe_router_scores_task", None)
        observe_direct = getattr(self.moe, "observe_router_scores", None)
        if observe_task is None and observe_direct is None:
            return

        scores = router_scores.detach()
        if scores.dim() == 1:
            scores = scores.view(1, -1)
        elif scores.dim() > 2:
            scores = scores.reshape(-1, scores.shape[-1])
        rows = int(scores.shape[0])
        cols = int(scores.shape[1])
        if rows <= 0 or cols <= 0:
            return

        max_elements = int(os.environ.get("KT_MESH_FULL_GATE_MAX_ELEMENTS", "65536"))
        if max_elements > 0 and rows * cols > max_elements:
            return

        if scores.dtype != torch.float32:
            scores = scores.float()
        if not scores.is_contiguous():
            scores = scores.contiguous()

        score_transform = self._mesh_score_transform_id()
        if scores.device.type == "cuda":
            flat = scores.view(-1)
            needed = flat.numel()
            buf = getattr(self, "_router_scores_cpu", None)
            if buf is None or buf.numel() < needed:
                buf = torch.empty(needed, dtype=torch.float32, device="cpu", pin_memory=_PIN_MEMORY)
                self._router_scores_cpu = buf
            dst = buf[:needed]
            dst.copy_(flat, non_blocking=True)
            if observe_task is not None:
                task = observe_task(dst.data_ptr(), rows, cols, score_transform)
                self._submit_cpuinfer_task(task, cuda_stream)
            else:
                self._sync_external_cuda_stream(cuda_stream or torch.cuda.current_stream(scores.device).cuda_stream)
                observe_direct(dst.data_ptr(), rows, cols, score_transform)
            return

        scores_cpu = scores.cpu() if scores.device.type != "cpu" else scores
        observe = observe_direct
        if observe is not None:
            observe(scores_cpu.data_ptr(), rows, cols, score_transform)

    def _prepare_router_scores_for_forward(
        self,
        router_scores: Optional[torch.Tensor],
        current_slot: int,
        cuda_stream=None,
        is_decode_token: bool = False,
    ) -> Tuple[int, int, int, int]:
        """Prepare full router scores for piggyback Heat update in forward_task."""
        if router_scores is None or not torch.is_tensor(router_scores):
            return 0, 0, 0, 0
        if not self._mesh_full_gate_observation_enabled():
            return 0, 0, 0, 0
        if self._cuda_graph_capture_active():
            return 0, 0, 0, 0
        if self._mesh_full_gate_batched_enabled():
            self._record_router_scores_for_token_batch(
                router_scores,
                cuda_stream,
                flush_on_last=False,
                is_decode_token=is_decode_token,
            )
            return 0, 0, 0, 0

        scores = router_scores.detach()
        if scores.dim() == 1:
            scores = scores.view(1, -1)
        elif scores.dim() > 2:
            scores = scores.reshape(-1, scores.shape[-1])
        rows = int(scores.shape[0])
        cols = int(scores.shape[1])
        if rows <= 0 or cols <= 0:
            return 0, 0, 0, 0

        max_elements = int(os.environ.get("KT_MESH_FULL_GATE_MAX_ELEMENTS", "65536"))
        if max_elements > 0 and rows * cols > max_elements:
            return 0, 0, 0, 0

        if scores.dtype != torch.float32:
            scores = scores.float()
        if not scores.is_contiguous():
            scores = scores.contiguous()

        score_transform = self._mesh_score_transform_id()
        if scores.device.type == "cuda":
            flat = scores.view(-1)
            needed = flat.numel()
            buffers = getattr(self, "_router_scores_cpu_slots", None)
            if buffers is None:
                buffers = {}
                self._router_scores_cpu_slots = buffers
            buf = buffers.get(current_slot)
            if buf is None or buf.numel() < needed:
                buf = torch.empty(needed, dtype=torch.float32, device="cpu", pin_memory=_PIN_MEMORY)
                buffers[current_slot] = buf
            dst = buf[:needed]
            dst.copy_(flat, non_blocking=True)
            return int(dst.data_ptr()), rows, cols, score_transform

        scores_cpu = scores.cpu() if scores.device.type != "cpu" else scores
        self._router_scores_cpu_direct = scores_cpu
        return int(scores_cpu.data_ptr()), rows, cols, score_transform

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw not in ("0", "false", "False", "FALSE", "no", "No", "NO")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _submit_iouring_prefetch(
        self,
        expert_ids_cpu: torch.Tensor,
        count: int,
        *,
        protect_ids_cpu: Optional[torch.Tensor] = None,
        protect_count: int = 0,
        max_to_submit: int = 0,
        cuda_stream=None,
        prefetch_kind: int = 0,
    ) -> bool:
        if self.io_backend != "IOURING" or self.moe is None or count <= 0:
            return False
        task_factory = getattr(self.moe, "prefetch_experts_task", None)
        if task_factory is None:
            return False

        expert_ids_cpu = expert_ids_cpu[:count]
        if expert_ids_cpu.dtype != torch.long:
            expert_ids_cpu = expert_ids_cpu.to(torch.long)
        if not expert_ids_cpu.is_contiguous():
            expert_ids_cpu = expert_ids_cpu.contiguous()

        protect_ptr = 0
        if protect_ids_cpu is not None and protect_count > 0:
            protect_ids_cpu = protect_ids_cpu[:protect_count]
            if protect_ids_cpu.dtype != torch.long:
                protect_ids_cpu = protect_ids_cpu.to(torch.long)
            if not protect_ids_cpu.is_contiguous():
                protect_ids_cpu = protect_ids_cpu.contiguous()
            self._mesh_prefetch_protect_keepalive = protect_ids_cpu
            protect_ptr = int(protect_ids_cpu.data_ptr())

        self._mesh_prefetch_ids_keepalive = expert_ids_cpu
        try:
            task = task_factory(
                int(expert_ids_cpu.data_ptr()),
                int(count),
                protect_ptr,
                int(protect_count),
                int(max_to_submit),
                int(prefetch_kind),
            )
        except TypeError:
            task = task_factory(
                int(expert_ids_cpu.data_ptr()),
                int(count),
                protect_ptr,
                int(protect_count),
                int(max_to_submit),
            )
        self._submit_cpuinfer_task(task, cuda_stream)
        return True

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
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth

        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        topk_ids_long = topk_ids.to(torch.long)

        # Tiered weight management: prefetch using PREVIOUS token's expert IDs
        # This avoids a GPU sync (.cpu().numpy()) in the hot path (PERF-1).
        # Previous-token IDs are a good predictor for current-token routing
        # because token sequences exhibit temporal locality in expert selection.
        prev_topk_ids = BaseMoEWrapper._prev_topk_ids_by_layer.get(self.layer_idx)
        if self._provider is not None and prev_topk_ids is not None:
            self._provider.prefetch_layer(self.layer_idx, prev_topk_ids)

        immediate_ids: torch.Tensor
        deferred_ids: Optional[torch.Tensor]
        if self.max_deferred_experts_per_token > 0:
            protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token

            immediate_ids, deferred_ids = self.select_deferred_experts(topk_ids_long, topk_weights, protected_k)
        else:
            immediate_ids = topk_ids_long
            deferred_ids = None

        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
        immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)
        if deferred_ids is not None:
            deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)

            if self._env_flag("KT_MESH_DEFER_PREFETCH", True):
                defer_prefetch_limit = self._env_int(
                    "KT_MESH_DEFER_PREFETCH_LIMIT",
                    max(1, int(self.max_deferred_experts_per_token)),
                )
                self._submit_iouring_prefetch(
                    deferred_experts_ids_cpu[current_slot],
                    int(deferred_experts_ids_cpu[current_slot].numel()),
                    protect_ids_cpu=immediate_experts_ids_cpu[current_slot],
                    protect_count=int(immediate_experts_ids_cpu[current_slot].numel()),
                    max_to_submit=defer_prefetch_limit,
                    cuda_stream=cuda_stream,
                )

        router_scores_ptr, score_rows, score_cols, score_transform = self._prepare_router_scores_for_forward(
            router_scores,
            current_slot,
            cuda_stream,
            is_decode_token=int(flat_hidden_states.shape[0]) == 1,
        )

        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        immediate_task = self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
            router_scores_ptr,
            score_rows,
            score_cols,
            score_transform,
        )
        self._submit_cpuinfer_task(immediate_task, cuda_stream)

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        if deferred_ids is not None:
            deferred_task = self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                deferred_experts_ids_cpu[current_slot].size(-1),
                deferred_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[next_slot].data_ptr(),
                False,
            )
            self._submit_cpuinfer_task(deferred_task, cuda_stream)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

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
        # Backward compatibility:
        #   sync_forward(hidden_states, cuda_stream)
        #   sync_forward(hidden_states, topk_ids, cuda_stream)
        topk_ids = topk_ids_or_stream
        if cuda_stream is None and topk_ids_or_stream is not None:
            is_stream_arg = hasattr(topk_ids_or_stream, "cuda_stream") or isinstance(topk_ids_or_stream, int)
            if is_stream_arg:
                cuda_stream = getattr(topk_ids_or_stream, "cuda_stream", topk_ids_or_stream)
                topk_ids = None

        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            _input_tensor_cpu,
            _immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            _weights_cpu,
            output_cpu,
            _bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
        self._sync_cpuinfer(allow_pending, cuda_stream)
        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
        self._debug_log_moe_once(
            flat_hidden_states,
            _immediate_experts_ids_cpu[current_slot],
            _weights_cpu[current_slot],
            output_cpu[current_slot],
        )
        if self._mesh_full_gate_batched_enabled():
            self._flush_pending_full_gate_batch(cuda_stream)

        # Record activations for hotness tracking AFTER sync (not in hot path).
        # The .cpu().numpy() transfer is acceptable here because we've already
        # synchronized with the CPU worker — no pipeline bubble introduced.
        if self._provider is not None and topk_ids is not None:
            ids_np = topk_ids.detach().cpu().numpy()
            self._provider.record_activations(self.layer_idx, ids_np)
            # Cache same-layer expert IDs for next-token prefetch without a hot-path GPU sync.
            BaseMoEWrapper._prev_topk_ids_by_layer[self.layer_idx] = ids_np

        return output_gpu[current_slot]

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
        self.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream, router_scores=router_scores)
        return self.sync_forward(hidden_states, topk_ids, cuda_stream)

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

    @staticmethod
    def reset_runtime_state(force: bool = False):
        """
        Reset global runtime singletons once no active wrappers remain.

        force=True should only be used when the caller knows no live wrappers will
        touch the old CPUInfer / provider objects afterwards.
        """
        if not force and BaseMoEWrapper._active_wrapper_count > 0:
            raise RuntimeError("Cannot reset runtime state while MoE wrappers are still active")
        provider = BaseMoEWrapper._tiered_provider
        if provider is not None:
            provider.stop_promotion_thread()
        BaseMoEWrapper._tiered_provider = None
        BaseMoEWrapper._tiered_provider_signature = None
        BaseMoEWrapper._cpu_infer_instance = None
        BaseMoEWrapper._cpu_infer_signature = None
        BaseMoEWrapper._layer_has_pending_deferred.clear()
        BaseMoEWrapper._prev_topk_ids_by_layer.clear()
        BaseMoEWrapper._wrappers_by_layer.clear()
        BaseMoEWrapper._full_gate_batch_states.clear()
        BaseMoEWrapper._full_gate_skip_logged.clear()
        BaseMoEWrapper._mesh_bootstrap_done = False
        BaseMoEWrapper._mesh_bootstrap_log_count = 0
        BaseMoEWrapper.clear_buffer_cache()

    def close(self):
        """Release layer-specific registrations and tear down globals when the last wrapper dies."""
        if getattr(self, "_closed", False):
            return

        try:
            if getattr(self, "cpu_infer", None) is not None:
                self.cpu_infer.sync()
        except Exception:
            pass

        try:
            if getattr(self, "_provider", None) is not None:
                self._provider.unregister_moe(self.layer_idx)
        except Exception:
            pass

        BaseMoEWrapper._layer_has_pending_deferred.pop(self.layer_idx, None)
        BaseMoEWrapper._prev_topk_ids_by_layer.pop(self.layer_idx, None)
        if BaseMoEWrapper._wrappers_by_layer.get(self.layer_idx) is self:
            BaseMoEWrapper._wrappers_by_layer.pop(self.layer_idx, None)

        self._provider = None
        self.moe = None
        self._closed = True

        if BaseMoEWrapper._active_wrapper_count > 0:
            BaseMoEWrapper._active_wrapper_count -= 1

        if BaseMoEWrapper._active_wrapper_count == 0:
            BaseMoEWrapper.reset_runtime_state(force=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
