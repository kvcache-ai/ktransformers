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

        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size == cls.temp_bs:
            return cls.temp_buffer

        input_tensor_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=_PIN_MEMORY, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        immediate_experts_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=_PIN_MEMORY)
            for _ in range(cls.buffer_depth)
        ]
        deferred_experts_ids_cpu = [
            torch.full((batch_size, num_experts_per_tok), -1, device="cpu", dtype=torch.long, pin_memory=_PIN_MEMORY)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=_PIN_MEMORY)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=_PIN_MEMORY, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        bsz_tensor_cpu = [
            torch.full((1,), batch_size, device="cpu", dtype=torch.int32, pin_memory=_PIN_MEMORY)
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


class BaseMoEWrapper(ABC):
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
        """
        self.layer_idx = layer_idx
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
                max_tier0_experts = int(os.environ["KT_MAX_TIER0_EXPERTS"])

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
                tier0_bytes = resolve_auto_tier0_budget_bytes(model_bytes=model_bytes)
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
                    f"available_ram={available_gb:.1f}GB, tier0={tier0_bytes / (1024**3):.1f}GB"
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
            print(
                "[CPUInfer] submit_with_cuda_stream unavailable; "
                "falling back to synchronous CUDA stream handoff"
            )
            cls._cpu_infer_stream_fallback_logged = True
        torch.cuda.ExternalStream(cuda_stream).synchronize()

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

        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        immediate_task = self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
        )
        self._submit_cpuinfer_task(immediate_task, cuda_stream)

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        if deferred_ids is not None:
            deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)
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
