# Base classes for SFT MoE operations
# SPDX-License-Identifier: Apache-2.0

"""
SFT (Supervised Fine-Tuning) MoE base classes and buffer management.

Provides:
- KExpertsSFTBuffer: Grow-only shared buffer for forward/backward passes
- BaseSFTMoEWrapper: Abstract base with concrete buffer management (template method pattern)
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from ..experts_base import _MoEBase


class KExpertsSFTBuffer:
    """
    CPU buffer management for SFT expert computation.

    Single grow-only buffer (never shrinks). Callers must use [:qlen] slicing
    since the buffer may be larger than the current batch.
    """

    _shared_buffer: Optional["KExpertsSFTBuffer"] = None

    def __init__(
        self,
        qlen: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        lora_rank: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.qlen = qlen
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.lora_rank = lora_rank
        self.dtype = dtype

        pin_memory = False

        # Forward buffers
        self.input_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.expert_ids_cpu = torch.empty(
            (qlen, num_experts_per_tok), dtype=torch.int64, device="cpu", pin_memory=pin_memory
        )
        self.weights_cpu = torch.empty(
            (qlen, num_experts_per_tok), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.output_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)

        # Backward buffers
        self.grad_output_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.grad_input_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.grad_weights = torch.empty((qlen, num_experts_per_tok), dtype=torch.float32, device="cpu")

        # Batch size tensor for C++ interface
        self.bsz_tensor = torch.tensor([qlen], dtype=torch.int32, device="cpu")

    @classmethod
    def get_buffer(
        cls,
        qlen: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        lora_rank: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "KExpertsSFTBuffer":
        """Get or grow the single shared buffer. Only reallocates when qlen exceeds capacity."""
        buf = cls._shared_buffer
        if buf is not None and qlen <= buf.qlen:
            return buf
        cls._shared_buffer = cls(
            qlen=qlen,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            lora_rank=lora_rank,
            dtype=dtype,
        )
        return cls._shared_buffer

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the shared buffer."""
        cls._shared_buffer = None


class BaseSFTMoEWrapper(_MoEBase, ABC):
    """
    Base class for SFT MoE CPU operations with concrete buffer management.

    Subclasses implement:
    - _make_forward_task(buffer, save_for_backward) -> C++ task object
    - _make_backward_task(buffer) -> C++ task object
    - load_weights(physical_to_logical_map_cpu)
    - init_lora_weights(...)
    - update_lora_weights()
    """

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_gpu_experts: int,
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        max_cache_depth: int = 1,
    ):
        self.cpu_infer = self._get_cpu_infer(cpuinfer_threads, threadpool_count)

        self._validate_base_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
        )
        self._validate_sft_config(lora_rank, lora_alpha, max_cache_depth)

        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_gpu_experts = num_gpu_experts
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.threadpool_count = threadpool_count

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank
        self.max_cache_depth = max_cache_depth

        self.gate_lora_a: Optional[torch.Tensor] = None
        self.gate_lora_b: Optional[torch.Tensor] = None
        self.up_lora_a: Optional[torch.Tensor] = None
        self.up_lora_b: Optional[torch.Tensor] = None
        self.down_lora_a: Optional[torch.Tensor] = None
        self.down_lora_b: Optional[torch.Tensor] = None

        self._weights_loaded: bool = False
        self._lora_initialized: bool = False
        self._cache_depth: int = 0
        self._is_skip_lora: bool = False

        self.moe = None

    @staticmethod
    def _validate_sft_config(lora_rank: int, lora_alpha: float, max_cache_depth: int) -> None:
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {lora_rank}")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
        if max_cache_depth <= 0:
            raise ValueError(f"max_cache_depth must be positive, got {max_cache_depth}")

    # ========== Abstract methods for subclasses ==========

    @abstractmethod
    def _make_forward_task(self, buffer: KExpertsSFTBuffer, save_for_backward: bool):
        """Construct the C++ forward task object. Backend-specific."""
        ...

    @abstractmethod
    def _make_backward_task(self, buffer: KExpertsSFTBuffer):
        """Construct the C++ backward task object. Backend-specific."""
        ...

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor) -> None:
        ...

    @abstractmethod
    def init_lora_weights(
        self,
        gate_lora_a: torch.Tensor, gate_lora_b: torch.Tensor,
        up_lora_a: torch.Tensor, up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor, down_lora_b: torch.Tensor,
        grad_gate_lora_a: torch.Tensor, grad_gate_lora_b: torch.Tensor,
        grad_up_lora_a: torch.Tensor, grad_up_lora_b: torch.Tensor,
        grad_down_lora_a: torch.Tensor, grad_down_lora_b: torch.Tensor,
    ) -> None:
        ...

    @abstractmethod
    def update_lora_weights(self) -> None:
        ...

    # ========== Buffer helpers ==========

    def _get_buffer(self, qlen: int) -> KExpertsSFTBuffer:
        return KExpertsSFTBuffer.get_buffer(
            qlen=qlen,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            lora_rank=self.lora_rank,
            dtype=torch.bfloat16,
        )

    def _validate_forward_inputs(self, hidden_states: torch.Tensor, expert_ids: torch.Tensor, weights: torch.Tensor):
        if not self._weights_loaded:
            raise RuntimeError("Weights not loaded. Call load_weights() or load_weights_from_tensors() first.")
        if not self._lora_initialized and not self._is_skip_lora:
            raise RuntimeError("LoRA weights not initialized. Call init_lora_weights() first.")
        qlen = hidden_states.shape[0]
        if qlen > self.chunked_prefill_size:
            raise ValueError(
                f"qlen ({qlen}) exceeds chunked_prefill_size ({self.chunked_prefill_size}). "
                "Increase chunked_prefill_size or reduce qlen to avoid buffer overrun."
            )
        if expert_ids.shape[0] != qlen or expert_ids.shape[1] != self.num_experts_per_tok:
            raise ValueError(
                f"expert_ids shape {tuple(expert_ids.shape)} must be ({qlen}, {self.num_experts_per_tok})."
            )
        if weights.shape[0] != qlen or weights.shape[1] != self.num_experts_per_tok:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} must be ({qlen}, {self.num_experts_per_tok})."
            )

    def _copy_inputs_to_buffer(self, buffer: KExpertsSFTBuffer, hidden_states: torch.Tensor,
                               expert_ids: torch.Tensor, weights: torch.Tensor, qlen: int) -> torch.device:
        """Copy inputs to CPU buffer, return input device."""
        input_device = hidden_states.device
        buffer.input_cpu[:qlen].copy_(hidden_states.to(torch.bfloat16), non_blocking=True)
        buffer.expert_ids_cpu[:qlen].copy_(expert_ids.to(torch.int64), non_blocking=True)
        buffer.weights_cpu[:qlen].copy_(weights.to(torch.float32), non_blocking=True)
        buffer.bsz_tensor[0] = qlen
        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)
        return input_device

    def _copy_grad_output_to_cpu(self, buffer: KExpertsSFTBuffer, grad_output: torch.Tensor, qlen: int):
        """Copy grad_output to CPU buffer."""
        input_device = grad_output.device
        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)
        buffer.grad_output_cpu[:qlen].copy_(grad_output.to(torch.bfloat16))

    def _return_output(self, buffer: KExpertsSFTBuffer, qlen: int, output_device: Optional[torch.device]):
        if output_device is not None:
            return buffer.output_cpu[:qlen].to(device=output_device, non_blocking=True)
        else:
            return buffer.output_cpu[:qlen].clone()

    def _return_grads(self, buffer: KExpertsSFTBuffer, qlen: int, output_device: Optional[torch.device]):
        if output_device is not None:
            grad_input = buffer.grad_input_cpu[:qlen].to(device=output_device, non_blocking=True)
            grad_weights = buffer.grad_weights[:qlen].to(device=output_device, non_blocking=True)
        else:
            grad_input = buffer.grad_input_cpu[:qlen].clone()
            grad_weights = buffer.grad_weights[:qlen].clone()
        return grad_input, grad_weights

    # ========== Concrete forward/backward ==========

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        save_for_backward: bool = True,
        output_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Synchronous forward pass with optional gradient caching."""
        self._validate_forward_inputs(hidden_states, expert_ids, weights)
        qlen = hidden_states.shape[0]
        buffer = self._get_buffer(qlen)
        self._copy_inputs_to_buffer(buffer, hidden_states, expert_ids, weights, qlen)

        self.cpu_infer.submit(self._make_forward_task(buffer, save_for_backward))
        self.cpu_infer.sync()

        if save_for_backward and self._cache_depth == 0:
            self._cache_depth += 1

        return self._return_output(buffer, qlen, output_device)

    def backward(
        self,
        grad_output: torch.Tensor,
        output_device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass computing grad_input and grad_weights."""
        if self._cache_depth <= 0:
            raise RuntimeError("No forward cache available. Call forward(save_for_backward=True) first.")

        qlen = grad_output.shape[0]
        buffer = self._get_buffer(qlen)
        self._copy_grad_output_to_cpu(buffer, grad_output, qlen)

        self.cpu_infer.submit(self._make_backward_task(buffer))
        self.cpu_infer.sync()

        self._cache_depth -= 1
        return self._return_grads(buffer, qlen, output_device)

    # ========== Async forward ==========

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        save_for_backward: bool = True,
    ) -> None:
        """Submit forward pass asynchronously (non-blocking). Call sync_forward() to get results."""
        self._validate_forward_inputs(hidden_states, expert_ids, weights)
        qlen = hidden_states.shape[0]
        buffer = self._get_buffer(qlen)
        self._copy_inputs_to_buffer(buffer, hidden_states, expert_ids, weights, qlen)

        self._pending_buffer = buffer
        self._pending_save_for_backward = save_for_backward
        self._pending_qlen = qlen

        self.cpu_infer.submit(self._make_forward_task(buffer, save_for_backward))

    def sync_forward(self, output_device: Optional[torch.device] = None) -> torch.Tensor:
        """Synchronize and retrieve forward results. Must be called after submit_forward()."""
        if not hasattr(self, "_pending_buffer") or self._pending_buffer is None:
            raise RuntimeError("No pending forward. Call submit_forward() first.")

        self.cpu_infer.sync()

        buffer = self._pending_buffer
        save_for_backward = self._pending_save_for_backward
        qlen = self._pending_qlen

        if save_for_backward and self._cache_depth == 0:
            self._cache_depth += 1

        self._pending_buffer = None
        self._pending_save_for_backward = None
        self._pending_qlen = None

        return self._return_output(buffer, qlen, output_device)

    # ========== Async backward ==========

    def submit_backward_async(
        self,
        grad_output: torch.Tensor,
        output_device: Optional[torch.device] = None,
    ) -> None:
        """Submit backward task without waiting. Call sync_backward() for results."""
        if self._cache_depth <= 0:
            raise RuntimeError("No forward cache available. Call forward(save_for_backward=True) first.")

        qlen = grad_output.shape[0]
        buffer = self._get_buffer(qlen)
        self._copy_grad_output_to_cpu(buffer, grad_output, qlen)

        self.cpu_infer.submit(self._make_backward_task(buffer))
        self._async_bwd_qlen = qlen
        self._async_bwd_output_device = output_device

    def sync_backward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wait for async backward and return results."""
        self.cpu_infer.sync()

        qlen = self._async_bwd_qlen
        output_device = self._async_bwd_output_device
        buffer = self._get_buffer(qlen)

        self._cache_depth -= 1
        return self._return_grads(buffer, qlen, output_device)

    # ========== Backward repack (optional, subclasses may override) ==========

    def submit_backward_repack(self):
        if not self._weights_loaded or self.moe is None:
            return
        if hasattr(self.moe, 'submit_backward_repack'):
            self.moe.submit_backward_repack()

    def wait_backward_repack(self):
        if not self._weights_loaded or self.moe is None:
            return
        if hasattr(self.moe, 'wait_backward_repack'):
            self.moe.wait_backward_repack()
