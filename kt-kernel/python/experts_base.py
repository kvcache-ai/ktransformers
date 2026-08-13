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
import ctypes

from kt_kernel import kt_kernel_ext


# -----------------------------------------------------------------------------
# NPU stream-callback bypass.
#
# On Ascend NPU, `CPUInfer::submit_with_cuda_stream` calls `aclrtLaunchCallback`,
# which inserts the function into a per-stream **callback report queue**. ACL
# requires a dedicated subscriber thread (registered via `aclrtSubscribeReport`
# and continuously running `aclrtProcessReport`) to dispatch those callbacks.
#
# kt-kernel does NOT currently start such a worker (see
# `cpu_backend/vendors/ascend_npu.h` — marked TODO Phase 3). Without a
# subscriber, queued callbacks **silently never fire**, so the CPU forward
# task never runs and `output_cpu` stays all-zero. `sync_with_cuda_stream`
# likewise schedules its sync_ via the same callback queue and silently
# completes without actually syncing anything.
#
# When the ACL callback worker is active (ascend_callback_worker.cpp, started
# via ``init_ascend_callback_worker``), ``submit_with_cuda_stream`` callbacks
# are dispatched and CPU/NPU overlap works.  Set ``KT_FORCE_SYNC_SUBMIT=1`` to
# force the legacy synchronous path for debugging.
# -----------------------------------------------------------------------------


def _should_bypass_stream_callback(device: torch.device) -> bool:
    """Return True iff we must use the synchronous submit/sync path."""
    if os.environ.get("KT_FORCE_SYNC_SUBMIT", "") == "1":
        return True
    return False


def _uses_external_npu_report_subscriber() -> bool:
    return os.environ.get("KT_EXTERNAL_NPU_REPORT_SUBSCRIBER", "") == "1"


def _ensure_ascend_callback_worker() -> None:
    """Start kt-kernel ACL callback worker (idempotent)."""
    if _uses_external_npu_report_subscriber():
        return
    if not hasattr(kt_kernel_ext, "init_ascend_callback_worker"):
        return
    if getattr(_ensure_ascend_callback_worker, "_done", False):
        return
    kt_kernel_ext.init_ascend_callback_worker()
    _ensure_ascend_callback_worker._done = True  # type: ignore[attr-defined]
    if hasattr(kt_kernel_ext, "shutdown_ascend_callback_worker"):
        import atexit

        atexit.register(kt_kernel_ext.shutdown_ascend_callback_worker)


def _sglang_is_capture_mode() -> bool:
    """True when sglang is inside ``model_capture_mode()`` (graph capture).

    kt-kernel must remain importable standalone, so the sglang dependency is
    optional and any failure is treated as "not capturing".
    """
    try:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


def _wait_device(device: torch.device) -> None:
    """Block until pending async copies on `device`'s current stream finish.

    NOTE on graph capture: torch.{cuda,npu}.synchronize() raises during cuda /
    NPU graph capture (NPU returns ERR 107027 "stream is captured"). Skip sync
    while capturing; graph MoE uses ``_launch_host_func`` + pinned buffers
    (see ``kt_ep_wrapper``).
    """
    if device.type == "npu":
        try:
            if torch.npu.is_current_stream_capturing():
                return
        except Exception:
            pass
        # Defensive fallback: torch.npu.is_current_stream_capturing() reliability
        # during torch_npu graph capture is unconfirmed; if it returns False (or
        # raises) while capturing, the synchronize() below would attempt a
        # stream sync on a captured stream and crash (107027/107030). Mirror the
        # capture detection used by kt_ep_wrapper._npu_use_graph_host_callback by
        # also consulting sglang's global capture flag, which model_capture_mode()
        # sets reliably around the whole capture loop.
        if _sglang_is_capture_mode():
            return
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        if _sglang_is_capture_mode():
            return
        torch.cuda.synchronize(device)


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
            try:
                if torch.npu.is_available():  # type: ignore[attr-defined]
                    _ensure_ascend_callback_worker()
            except Exception:
                pass
            worker_config = kt_kernel_ext.WorkerPoolConfig()

            if numa_nodes is not None:
                if len(numa_nodes) != threadpool_count:
                    raise ValueError(
                        f"numa_nodes length ({len(numa_nodes)}) must match "
                        f"threadpool_count ({threadpool_count})"
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
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer on this layer. Defaults to 0 (no defer).
            method: Backend method string
            numa_nodes: Explicit list of NUMA node IDs for subpool mapping.
                        If None, defaults to [0, 1, ..., threadpool_count-1].
        """
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        # Process gpu_experts_mask: convert to bool tensor on CPU, pinned memory for async copy
        # This mask is shared between C and Python (C uses uint8_t*), both can read/write it
        if gpu_experts_mask is None:
            # No GPU experts - all experts on CPU
            self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)
        else:
            # Create a new pinned tensor and copy data into it
            self.gpu_experts_mask = torch.empty(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)
            self.gpu_experts_mask.copy_(gpu_experts_mask)

        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())

        # GPU copy for mask operations in forward pass (e.g., mask_cpu_expert_ids)
        # This will be lazily initialized when needed
        self._gpu_experts_mask_gpu: Optional[torch.Tensor] = None
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpu_save = cpu_save
        self.max_deferred_experts_per_token = (
            int(max_deferred_experts_per_token) if max_deferred_experts_per_token is not None else 0
        )

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        self.method = method
        # V4-Flash 2604B SwiGLU clamp limit; 0.0 = disabled. NativeMoEWrapper
        # (MXFP4 path) reads this in load_weights() and writes it into
        # MOEConfig.swiglu_limit. Other backends ignore it (C++ act_fn skips
        # the clamp branch when limit==0). Origin: kt-sglang 耦合.
        self.swiglu_limit = float(swiglu_limit)

        # Initialize CPU inference engine (singleton via shared base class)
        self.cpu_infer = self._get_cpu_infer(cpuinfer_threads, threadpool_count, numa_nodes=numa_nodes)

        # Backend-specific initialization happens in subclasses
        self.moe = None

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

    def _check_qlen_fits_cpp_buffers(self, hidden_states: torch.Tensor) -> None:
        """Fail loudly when qlen would overrun the C++ MoE output buffer.

        ``moe-tp.hpp`` sizes ``local_output_numa[i]`` by ``max_possible_qlen()`` =
        ``max(max_len, group_max_len)``, and both are set to ``chunked_prefill_size``
        (``utils/llamafile.py``). ``TP::forward`` then hands the *full* qlen to
        ``MOE::forward``, whose recursion only splits the internal scratch — the
        caller-supplied output pointer still advances across ``qlen * hidden_size``.
        So ``qlen > chunked_prefill_size`` writes past the allocation and corrupts the
        heap, surfacing later as an unrelated ``malloc(): unaligned tcache chunk``
        abort. Raise here instead, mirroring the SFT path (``sft/base.py``).
        """
        qlen = hidden_states.numel() // hidden_states.shape[-1]
        if qlen > self.chunked_prefill_size:
            raise ValueError(
                f"qlen ({qlen}) exceeds chunked_prefill_size ({self.chunked_prefill_size}); "
                "the C++ MoE output buffer is sized by chunked_prefill_size and would be "
                "overrun. Raise --chunked-prefill-size or reduce the prefill chunk."
            )

    def _prepare_forward_cpu_buffers(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], tuple, int, int]:
        """D2H copy into pinned CPU buffers; return deferred ids and buffer handles."""
        self._check_qlen_fits_cpp_buffers(hidden_states)
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
        if self.max_deferred_experts_per_token > 0:
            protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token
            immediate_ids, deferred_ids = self.select_deferred_experts(
                topk_ids_long, topk_weights, protected_k
            )
        else:
            immediate_ids = topk_ids_long
            deferred_ids = None

        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
        immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)
        if deferred_ids is not None:
            deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)

        buffers = (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        )
        return immediate_ids, deferred_ids, buffers, current_slot, next_slot

    def copy_inputs_to_cpu_buffers(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Copy MoE inputs to pinned CPU buffers (for NPU graph host callbacks)."""
        self._prepare_forward_cpu_buffers(hidden_states, topk_ids, topk_weights)

    def forward_on_pinned_buffers(
        self,
        hidden_states: torch.Tensor,
        cuda_stream,
    ) -> None:
        """Run CPU MoE on buffers already filled (sync or stream callback)."""
        self._check_qlen_fits_cpp_buffers(hidden_states)
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

        bypass = _should_bypass_stream_callback(hidden_states.device)
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
        if bypass:
            self.cpu_infer.submit(immediate_task)
        else:
            # Correct + fast async path. ``submit_with_cuda_stream`` enqueues the
            # CPU-MoE via an ACL host callback whose firing is not host-observable
            # (NO_BLOCK) -> a later host-side drain can race ahead of it (empty
            # queue -> stale output_cpu read by the H2D; nondeterministic on heavy
            # prefill). Enqueue synchronously on this host thread instead (work is
            # guaranteed submitted; it still runs on the WorkerPool and overlaps the
            # GPU experts queued after). Keep subscribe_ascend_stream — that stream
            # registration is what keeps decode's host-callback dispatch fast; the
            # async submit itself is not required for it.
            if (
                hidden_states.device.type == "npu"
                and not _uses_external_npu_report_subscriber()
                and hasattr(
                kt_kernel_ext, "subscribe_ascend_stream"
                )
            ):
                kt_kernel_ext.subscribe_ascend_stream(int(cuda_stream))
            self.cpu_infer.submit(immediate_task)

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        has_deferred = (
            self.max_deferred_experts_per_token > 0
            and (deferred_experts_ids_cpu[current_slot] >= 0).any().item()
        )
        if has_deferred:
            deferred_task = self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                deferred_experts_ids_cpu[current_slot].size(-1),
                deferred_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[next_slot].data_ptr(),
                False,
            )
            if bypass:
                self.cpu_infer.submit(deferred_task)
            else:
                self.cpu_infer.submit_with_cuda_stream(cuda_stream, deferred_task)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

    def run_pinned_forward_sync(
        self,
        hidden_states: torch.Tensor,
        cuda_stream,
    ) -> None:
        """Submit + sync CPU MoE on pre-filled buffers (NPU graph host callback).

        Called from ``aclrtLaunchCallback`` / ``_launch_host_func``; must not enqueue
        nested stream callbacks.
        """
        del cuda_stream  # unused — sync path only
        self._check_qlen_fits_cpp_buffers(hidden_states)
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
        self.cpu_infer.submit(immediate_task)
        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        has_deferred = (
            self.max_deferred_experts_per_token > 0
            and (deferred_experts_ids_cpu[current_slot] >= 0).any().item()
        )
        if has_deferred:
            deferred_task = self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                deferred_experts_ids_cpu[current_slot].size(-1),
                deferred_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[next_slot].data_ptr(),
                False,
            )
            self.cpu_infer.submit(deferred_task)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True
        allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
        self.cpu_infer.sync(allow_pending)

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
        _immediate_ids, deferred_ids, _buffers, current_slot, next_slot = (
            self._prepare_forward_cpu_buffers(hidden_states, topk_ids, topk_weights)
        )
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = _buffers
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        bypass = _should_bypass_stream_callback(hidden_states.device)
        # Both paths now submit the CPU-MoE synchronously on this host thread, which
        # reads input_tensor_cpu immediately -> the input D2H (queued async on the
        # stream by copy_inputs_to_cpu_buffers) MUST be finished first, else the CPU
        # MoE reads a half-copied input. Wait unconditionally (bypass already did).
        _wait_device(hidden_states.device)

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
        if bypass:
            self.cpu_infer.submit(immediate_task)
        else:
            # Correct + fast async path. ``submit_with_cuda_stream`` enqueues the
            # CPU-MoE via an ACL host callback whose firing is not host-observable
            # (NO_BLOCK) -> a later host-side drain can race ahead of it (empty
            # queue -> stale output_cpu read by the H2D; nondeterministic on heavy
            # prefill). Enqueue synchronously on this host thread instead (work is
            # guaranteed submitted; it still runs on the WorkerPool and overlaps the
            # GPU experts queued after). Keep subscribe_ascend_stream — that stream
            # registration is what keeps decode's host-callback dispatch fast; the
            # async submit itself is not required for it.
            if (
                hidden_states.device.type == "npu"
                and not _uses_external_npu_report_subscriber()
                and hasattr(
                kt_kernel_ext, "subscribe_ascend_stream"
                )
            ):
                kt_kernel_ext.subscribe_ascend_stream(int(cuda_stream))
            self.cpu_infer.submit(immediate_task)

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        if deferred_ids is not None:
            _wait_device(hidden_states.device)
            deferred_task = self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                deferred_experts_ids_cpu[current_slot].size(-1),
                deferred_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[next_slot].data_ptr(),
                False,
            )
            if bypass:
                self.cpu_infer.submit(deferred_task)
            else:
                if (
                    hidden_states.device.type == "npu"
                    and not _uses_external_npu_report_subscriber()
                    and hasattr(
                    kt_kernel_ext, "subscribe_ascend_stream"
                    )
                ):
                    kt_kernel_ext.subscribe_ascend_stream(int(cuda_stream))
                self.cpu_infer.submit(deferred_task)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

    def copy_forward_output_to_device(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Copy pinned CPU output to the device tensor (CPU work already finished)."""
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
        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
        return output_gpu[current_slot]

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream) -> torch.Tensor:
        """
        Synchronize and retrieve forward inference results.

        Args:
            hidden_states: Original input hidden states (for getting buffer)
            cuda_stream: CUDA stream for synchronization

        Returns:
            output_gpu: Output tensor on GPU
        """
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
        bypass = _should_bypass_stream_callback(hidden_states.device)
        if bypass:
            self.cpu_infer.sync(allow_pending)
        else:
            if (
                hidden_states.device.type == "npu"
                and not _uses_external_npu_report_subscriber()
                and hasattr(
                kt_kernel_ext, "subscribe_ascend_stream"
                )
            ):
                kt_kernel_ext.subscribe_ascend_stream(int(cuda_stream))
            if hidden_states.device.type == "npu":
                # ACL host callbacks do not order the following H2D copy behind
                # the WorkerPool drain.  Wait for the accelerator work that
                # overlapped CPU MoE, then drain on the host before copying the
                # completed pinned output back to the NPU.
                _wait_device(hidden_states.device)
                self.cpu_infer.sync(allow_pending)
            else:
                self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)

        if os.environ.get("KT_DEBUG_MOE_OUT", "") == "1":
            oc = output_cpu[current_slot]
            ic = _input_tensor_cpu[current_slot]
            wc = _weights_cpu[current_slot]
            ec = _immediate_experts_ids_cpu[current_slot]
            finite_oc = torch.isfinite(oc).all().item()
            finite_ic = torch.isfinite(ic).all().item()
            finite_wc = torch.isfinite(wc).all().item()
            nan_count = int(torch.isnan(oc).sum().item())
            inf_count = int(torch.isinf(oc).sum().item())
            print(
                f"[KT_DEBUG] layer={self.layer_idx} slot={current_slot} bypass={bypass} "
                f"output_cpu: finite={finite_oc} nan={nan_count}/{oc.numel()} inf={inf_count} "
                f"first16={oc.float().flatten()[:16].tolist()} | "
                f"input_cpu.finite={finite_ic} first8={ic.float().flatten()[:8].tolist()} | "
                f"weights_cpu.finite={finite_wc} first={wc.flatten()[:8].tolist()} | "
                f"expert_ids={ec.flatten()[:12].tolist()}",
                flush=True,
            )

        return self.copy_forward_output_to_device(hidden_states)

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
