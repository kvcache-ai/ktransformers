# Base classes for SFT MoE operations
# SPDX-License-Identifier: Apache-2.0

"""
SFT (Supervised Fine-Tuning) MoE base classes and buffer management.

Provides:
- KExpertsSFTBuffer: Grow-only shared buffer for forward/backward passes
- BaseSFTMoEWrapper: Abstract base with concrete buffer management (template method pattern)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from ..experts_base import KExpertsCPUBuffer, _MoEBase


def _supports_authoritative_optimizer_grads(method: str, num_gpu_experts: int) -> bool:
    """Whether this backend can use C++-authoritative optimizer gradients."""
    return method == "AMXBF16_SFT" and int(num_gpu_experts) == 0


@dataclass(frozen=True)
class _AuthoritativeOptimizerGrad:
    """Stable Parameter-to-C++-gradient binding for one optimizer tensor."""

    name: str
    parameter: torch.nn.Parameter
    grad_view: torch.Tensor
    metadata: tuple


def _authoritative_grad_metadata(tensor: torch.Tensor) -> tuple:
    device_index = tensor.device.index if tensor.device.index is not None else -1
    return (
        tensor.dtype,
        tensor.layout,
        tensor.device.type,
        device_index,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        int(tensor.data_ptr()),
    )


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


class _SFTForwardBufferView:
    """Minimal buffer view consumed by AMXSFTMoEWrapper._make_forward_task."""

    __slots__ = ("bsz_tensor", "expert_ids_cpu", "weights_cpu", "input_cpu", "output_cpu")

    def __init__(
        self,
        bsz_tensor: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        input_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ):
        self.bsz_tensor = bsz_tensor
        self.expert_ids_cpu = expert_ids_cpu
        self.weights_cpu = weights_cpu
        self.input_cpu = input_cpu
        self.output_cpu = output_cpu


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
        full_weight_grad: bool = False,
    ):
        self.cpu_infer = self._get_cpu_infer(cpuinfer_threads, threadpool_count)

        self._validate_base_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
        )
        self._validate_sft_config(lora_rank, lora_alpha, max_cache_depth, full_weight_grad=full_weight_grad)

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
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 0.0
        self.max_cache_depth = max_cache_depth

        self._full_weight_grad = full_weight_grad

        self.gate_lora_a: Optional[torch.Tensor] = None
        self.gate_lora_b: Optional[torch.Tensor] = None
        self.up_lora_a: Optional[torch.Tensor] = None
        self.up_lora_b: Optional[torch.Tensor] = None
        self.down_lora_a: Optional[torch.Tensor] = None
        self.down_lora_b: Optional[torch.Tensor] = None

        # Base weight parameters for full fine-tuning
        self.gate_proj_buf: Optional[torch.Tensor] = None
        self.up_proj_buf: Optional[torch.Tensor] = None
        self.down_proj_buf: Optional[torch.Tensor] = None
        self.grad_gate_proj_buf: Optional[torch.Tensor] = None
        self.grad_up_proj_buf: Optional[torch.Tensor] = None
        self.grad_down_proj_buf: Optional[torch.Tensor] = None

        self._weights_loaded: bool = False
        self._lora_initialized: bool = False
        self._cache_depth: int = 0
        self._is_skip_lora: bool = False
        self._base_weights_dirty: bool = False
        # AMXSFTMoEWrapper enables this capability only for AMXBF16_SFT.
        # Keeping it false here preserves legacy INT8/INT4/SkipLoRA behavior.
        self._uses_authoritative_optimizer_grads: bool = False
        self._init_authoritative_optimizer_grads()
        self.reuse_checkpoint_forward: bool = False
        self._kt_has_cached_forward: bool = False
        self._checkpoint_output_cpu: Optional[torch.Tensor] = None
        self._checkpoint_output_qlen: int = 0
        self._backward_repack_pending: bool = False

        self.moe = None

    # ========== Authoritative optimizer-gradient lifecycle ==========

    def _init_authoritative_optimizer_grads(self) -> None:
        self._authoritative_optimizer_grads: list[_AuthoritativeOptimizerGrad] = []
        self._authoritative_grad_submission_pending: bool = False
        self._authoritative_grad_pending_accumulate: bool = False

    @property
    def authoritative_optimizer_grads(self) -> tuple[_AuthoritativeOptimizerGrad, ...]:
        """Read-only view of the persistent C++ optimizer-gradient bindings."""
        return tuple(self._authoritative_optimizer_grads)

    def register_authoritative_optimizer_grad(
        self,
        name: str,
        parameter: torch.nn.Parameter,
        grad_view: torch.Tensor,
    ) -> None:
        """Register a C++-written gradient as the Parameter's sole grad copy."""
        if not self._uses_authoritative_optimizer_grads:
            return
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"{name}: expected nn.Parameter, got {type(parameter)!r}")
        if not isinstance(grad_view, torch.Tensor):
            raise TypeError(f"{name}: expected Tensor grad view, got {type(grad_view)!r}")
        if parameter.shape != grad_view.shape:
            raise ValueError(
                f"{name}: Parameter/grad shape mismatch: {tuple(parameter.shape)} != {tuple(grad_view.shape)}"
            )
        if parameter.dtype != grad_view.dtype or parameter.device != grad_view.device:
            raise ValueError(
                f"{name}: Parameter/grad dtype or device mismatch: "
                f"{parameter.dtype}/{parameter.device} != {grad_view.dtype}/{grad_view.device}"
            )
        for entry in self._authoritative_optimizer_grads:
            if entry.parameter is parameter:
                raise RuntimeError(f"{name}: Parameter is already registered as {entry.name}")
            if entry.grad_view is grad_view:
                raise RuntimeError(f"{name}: grad view is already registered as {entry.name}")

        # Initial state is always a closed optimizer window.  The view remains
        # alive in this registry while Parameter.grad is None.
        parameter.grad = None
        self._authoritative_optimizer_grads.append(
            _AuthoritativeOptimizerGrad(
                name=name,
                parameter=parameter,
                grad_view=grad_view,
                metadata=_authoritative_grad_metadata(grad_view),
            )
        )

    def _validate_authoritative_grad_metadata(self) -> None:
        for entry in self._authoritative_optimizer_grads:
            current = _authoritative_grad_metadata(entry.grad_view)
            if current != entry.metadata:
                raise RuntimeError(
                    f"{entry.name}: authoritative grad view metadata changed; "
                    "replacing or resizing a C++ gradient buffer is unsupported"
                )
            parameter = entry.parameter
            grad_view = entry.grad_view
            if parameter.shape != grad_view.shape or parameter.dtype != grad_view.dtype:
                raise RuntimeError(f"{entry.name}: Parameter metadata no longer matches its authoritative grad view")
            if parameter.device != grad_view.device:
                raise RuntimeError(f"{entry.name}: Parameter device no longer matches its authoritative grad view")

    def validate_authoritative_optimizer_grad_state(self) -> str:
        """Validate aliases and return ``empty``, ``closed``, or ``open``."""
        if not self._uses_authoritative_optimizer_grads or not self._authoritative_optimizer_grads:
            return "empty"
        self._validate_authoritative_grad_metadata()

        none_count = 0
        alias_count = 0
        for entry in self._authoritative_optimizer_grads:
            grad = entry.parameter.grad
            if grad is None:
                none_count += 1
            elif grad is entry.grad_view:
                alias_count += 1
            else:
                raise RuntimeError(
                    f"{entry.name}: Parameter.grad was externally replaced; "
                    "expected None or the registered authoritative grad view"
                )

        total = len(self._authoritative_optimizer_grads)
        if none_count == total:
            return "closed"
        if alias_count == total:
            return "open"
        raise RuntimeError(
            "Mixed authoritative optimizer-gradient state: some KT Parameter.grad values are None "
            "while others still alias their C++ buffers"
        )

    def _prepare_authoritative_optimizer_grad_write(self, optimizer_grad_scale: float) -> bool:
        if self._authoritative_grad_submission_pending:
            raise RuntimeError("An authoritative optimizer-gradient backward submission is already pending")
        scale = float(optimizer_grad_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"optimizer_grad_scale must be finite and positive, got {optimizer_grad_scale}")
        state = self.validate_authoritative_optimizer_grad_state()
        accumulate = state == "open"
        self._authoritative_grad_submission_pending = True
        self._authoritative_grad_pending_accumulate = accumulate
        return accumulate

    def _publish_authoritative_optimizer_grads(self) -> None:
        if not self._authoritative_grad_submission_pending:
            raise RuntimeError("No authoritative optimizer-gradient backward submission is pending")
        expected_state = "open" if self._authoritative_grad_pending_accumulate else "closed"
        try:
            state = self.validate_authoritative_optimizer_grad_state()
            if state not in ("empty", expected_state):
                raise RuntimeError(
                    f"Authoritative optimizer-gradient state changed during C++ backward: "
                    f"expected {expected_state}, got {state}"
                )
            for entry in self._authoritative_optimizer_grads:
                entry.parameter.grad = entry.grad_view
        except Exception:
            self._abort_authoritative_optimizer_grad_write()
            raise
        self._authoritative_grad_submission_pending = False
        self._authoritative_grad_pending_accumulate = False

    def _abort_authoritative_optimizer_grad_write(self) -> None:
        # A failed C++ task may have partially modified its outputs.  Closing
        # the Python window forces the next task down the overwrite/full-init path.
        for entry in self._authoritative_optimizer_grads:
            entry.parameter.grad = None
        self._authoritative_grad_submission_pending = False
        self._authoritative_grad_pending_accumulate = False

    def release_authoritative_optimizer_grads(self) -> None:
        """Close the optimizer window after step without touching C++ buffers."""
        if not self._uses_authoritative_optimizer_grads:
            return
        if self._authoritative_grad_submission_pending:
            raise RuntimeError("Cannot release authoritative gradients while C++ backward is pending")
        self.validate_authoritative_optimizer_grad_state()
        for entry in self._authoritative_optimizer_grads:
            entry.parameter.grad = None

    @staticmethod
    def _validate_sft_config(
        lora_rank: int, lora_alpha: float, max_cache_depth: int, full_weight_grad: bool = False
    ) -> None:
        if not full_weight_grad and lora_rank <= 0:
            raise ValueError(
                f"lora_rank must be positive in LoRA mode, got {lora_rank}. "
                "Set kt_train_mode='full' for full fine-tuning."
            )
        if lora_rank > 0 and lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
        if max_cache_depth <= 0:
            raise ValueError(f"max_cache_depth must be positive, got {max_cache_depth}")

    # ========== Full weight grad methods ==========

    def init_full_weight_grad_buffers(
        self, gate_proj: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor
    ) -> None:
        """Initialize base weight nn.Parameter buffers and gradient buffers for full fine-tuning.

        Args:
            gate_proj: [num_experts, intermediate_size, hidden_size] BF16 CPU tensor
            up_proj: [num_experts, intermediate_size, hidden_size] BF16 CPU tensor
            down_proj: [num_experts, hidden_size, intermediate_size] BF16 CPU tensor
        """
        import torch.nn as nn

        dtype = torch.bfloat16
        E = self.num_experts
        I = self.moe_intermediate_size
        H = self.hidden_size

        # Create nn.Parameter buffers (optimizer-visible)
        self.gate_proj_buf = nn.Parameter(gate_proj.to(dtype=dtype, device="cpu").contiguous(), requires_grad=True)
        self.up_proj_buf = nn.Parameter(up_proj.to(dtype=dtype, device="cpu").contiguous(), requires_grad=True)
        self.down_proj_buf = nn.Parameter(down_proj.to(dtype=dtype, device="cpu").contiguous(), requires_grad=True)

        # C++ clears these authoritative gradient buffers before first use.
        self.grad_gate_proj_buf = torch.empty(E, I, H, dtype=dtype, device="cpu")
        self.grad_up_proj_buf = torch.empty(E, I, H, dtype=dtype, device="cpu")
        self.grad_down_proj_buf = torch.empty(E, H, I, dtype=dtype, device="cpu")

        if self._uses_authoritative_optimizer_grads:
            self.register_authoritative_optimizer_grad("base.gate_proj", self.gate_proj_buf, self.grad_gate_proj_buf)
            self.register_authoritative_optimizer_grad("base.up_proj", self.up_proj_buf, self.grad_up_proj_buf)
            self.register_authoritative_optimizer_grad("base.down_proj", self.down_proj_buf, self.grad_down_proj_buf)
        # Legacy backends leave .grad unset here and return these buffers from
        # KTMoEFunction.backward(), preserving their existing AccumulateGrad path.

    @abstractmethod
    def update_base_weights(self) -> None:
        """Sync updated base weight parameters back to C++ kernel after optimizer step."""
        ...

    # ========== Abstract methods for subclasses ==========

    @abstractmethod
    def _make_forward_task(self, buffer: KExpertsSFTBuffer, save_for_backward: bool):
        """Construct the C++ forward task object. Backend-specific."""
        ...

    @abstractmethod
    def _make_backward_task(
        self,
        buffer: KExpertsSFTBuffer,
        accumulate_optimizer_grads: bool = False,
        optimizer_grad_scale: float = 1.0,
    ):
        """Construct the C++ backward task object. Backend-specific."""
        ...

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor) -> None: ...

    @abstractmethod
    def init_lora_weights(
        self,
        gate_lora_a: torch.Tensor,
        gate_lora_b: torch.Tensor,
        up_lora_a: torch.Tensor,
        up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        grad_gate_lora_a: torch.Tensor,
        grad_gate_lora_b: torch.Tensor,
        grad_up_lora_a: torch.Tensor,
        grad_up_lora_b: torch.Tensor,
        grad_down_lora_a: torch.Tensor,
        grad_down_lora_b: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def update_lora_weights(self) -> None: ...

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
        # Hybrid mode still requires LoRA buffers even though base gradients are
        # enabled.  Only pure Full (lora_rank == 0) may legitimately skip them.
        if self.lora_rank > 0 and not self._lora_initialized and not self._is_skip_lora:
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
            raise ValueError(f"weights shape {tuple(weights.shape)} must be ({qlen}, {self.num_experts_per_tok}).")

    def _copy_inputs_to_buffer(
        self,
        buffer: KExpertsSFTBuffer,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        qlen: int,
    ) -> torch.device:
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

    def cache_checkpoint_output(self, output_cpu: torch.Tensor, qlen: int) -> None:
        if output_cpu.device.type != "cpu":
            raise ValueError("checkpoint CPU expert output must reside on CPU")
        if output_cpu.shape[0] < qlen:
            raise ValueError(f"checkpoint output is shorter than qlen: {output_cpu.shape[0]} < {qlen}")
        self._checkpoint_output_cpu = output_cpu[:qlen].contiguous()
        self._checkpoint_output_qlen = qlen
        self._kt_has_cached_forward = True

    def get_checkpoint_output(self, qlen: int, output_device: Optional[torch.device] = None) -> torch.Tensor:
        if not self._kt_has_cached_forward or self._checkpoint_output_cpu is None:
            raise RuntimeError("No cached checkpoint forward output is available.")
        if qlen != self._checkpoint_output_qlen:
            raise RuntimeError(f"Cached checkpoint qlen mismatch: cached={self._checkpoint_output_qlen}, requested={qlen}")
        output = self._checkpoint_output_cpu
        if output_device is not None:
            return output.to(device=output_device, non_blocking=True)
        return output

    def clear_checkpoint_output(self) -> None:
        self._checkpoint_output_cpu = None
        self._checkpoint_output_qlen = 0
        self._kt_has_cached_forward = False

    def _return_grads(self, buffer: KExpertsSFTBuffer, qlen: int, output_device: Optional[torch.device]):
        if output_device is not None:
            grad_input = buffer.grad_input_cpu[:qlen].to(device=output_device, non_blocking=True)
            grad_weights = buffer.grad_weights[:qlen].to(device=output_device, non_blocking=True)
        else:
            grad_input = buffer.grad_input_cpu[:qlen].clone()
            grad_weights = buffer.grad_weights[:qlen].clone()
        return grad_input, grad_weights

    # ========== Concrete forward/backward ==========

    def _wait_for_pending_backward_repack(self) -> None:
        if getattr(self, "_backward_repack_pending", False):
            self.wait_backward_repack()

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

        self._wait_for_pending_backward_repack()
        self.cpu_infer.submit(self._make_forward_task(buffer, save_for_backward))
        self.cpu_infer.sync()

        if save_for_backward and self._cache_depth == 0:
            self._cache_depth += 1

        return self._return_output(buffer, qlen, output_device)

    def backward(
        self,
        grad_output: torch.Tensor,
        output_device: Optional[torch.device] = None,
        optimizer_grad_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass computing grad_input and grad_weights."""
        if self._cache_depth <= 0:
            raise RuntimeError("No forward cache available. Call forward(save_for_backward=True) first.")
        if self._uses_authoritative_optimizer_grads and self._authoritative_grad_submission_pending:
            raise RuntimeError("An authoritative optimizer-gradient backward submission is already pending")
        optimizer_grad_scale = float(optimizer_grad_scale)
        if not math.isfinite(optimizer_grad_scale) or optimizer_grad_scale <= 0.0:
            raise ValueError(f"optimizer_grad_scale must be finite and positive, got {optimizer_grad_scale}")

        qlen = grad_output.shape[0]
        buffer = self._get_buffer(qlen)
        self._copy_grad_output_to_cpu(buffer, grad_output, qlen)

        use_authoritative = self._uses_authoritative_optimizer_grads
        accumulate_optimizer_grads = False
        if use_authoritative:
            accumulate_optimizer_grads = self._prepare_authoritative_optimizer_grad_write(optimizer_grad_scale)
        try:
            if use_authoritative:
                backward_task = self._make_backward_task(
                    buffer,
                    accumulate_optimizer_grads=accumulate_optimizer_grads,
                    optimizer_grad_scale=optimizer_grad_scale,
                )
            elif optimizer_grad_scale != 1.0:
                # Legacy backends still let PyTorch AccumulateGrad own GAS
                # accumulation, but their C++ dWeight producer must apply
                # distributed world-size normalization before grad clipping.
                backward_task = self._make_backward_task(
                    buffer,
                    accumulate_optimizer_grads=False,
                    optimizer_grad_scale=optimizer_grad_scale,
                )
            else:
                # Preserve the historical task signature for single-rank
                # legacy backends and older compatible extension builds.
                backward_task = self._make_backward_task(buffer)
            self._wait_for_pending_backward_repack()
            self.cpu_infer.submit(backward_task)
            self.cpu_infer.sync()
            result = self._return_grads(buffer, qlen, output_device)
            if use_authoritative:
                self._publish_authoritative_optimizer_grads()
        except Exception:
            if use_authoritative:
                self._abort_authoritative_optimizer_grad_write()
                # The C++ forward cache may already have been consumed by a
                # partially executed backward. Require a fresh forward before
                # retrying instead of reusing an indeterminate cache entry.
                self._cache_depth = max(0, self._cache_depth - 1)
            raise

        self._cache_depth -= 1
        return result

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

        self._wait_for_pending_backward_repack()
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

    # ========== Inference-only async forward ==========

    def submit_forward_inference(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        cuda_stream,
    ) -> None:
        """
        Submit an SFT MoE forward pass for serving.

        This path mirrors the normal KT inference wrapper: inputs are copied to
        pinned CPU staging buffers, the CPUInfer task is enqueued with the
        caller CUDA stream, and sync_forward_inference() returns a persistent
        GPU output buffer. It deliberately avoids the training-oriented
        torch.cuda.synchronize() in _copy_inputs_to_buffer().
        """
        if not hasattr(self.cpu_infer, "submit_with_cuda_stream"):
            self.submit_forward(hidden_states, expert_ids, weights, save_for_backward=False)
            self._pending_inference_fallback = True
            self._pending_inference_fallback_device = hidden_states.device
            return

        self._validate_forward_inputs(hidden_states, expert_ids, weights)
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        (
            input_tensor_cpu,
            expert_ids_cpu,
            _deferred_expert_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        torch_stream = (
            cuda_stream
            if isinstance(cuda_stream, torch.cuda.Stream)
            else torch.cuda.ExternalStream(cuda_stream, device=flat_hidden_states.device)
        )
        with torch.cuda.stream(torch_stream):
            input_tensor_cpu[current_slot].copy_(flat_hidden_states.to(torch.bfloat16), non_blocking=True)
            expert_ids_cpu[current_slot].copy_(expert_ids.to(torch.int64), non_blocking=True)
            weights_cpu[current_slot].copy_(weights.to(torch.float32), non_blocking=True)

        buffer_view = _SFTForwardBufferView(
            bsz_tensor=bsz_slot_tensor,
            expert_ids_cpu=expert_ids_cpu[current_slot],
            weights_cpu=weights_cpu[current_slot],
            input_cpu=input_tensor_cpu[current_slot],
            output_cpu=output_cpu[current_slot],
        )

        self._pending_inference_fallback = False
        self._pending_inference_output_cpu = output_cpu[current_slot]
        self._pending_inference_output_gpu = output_gpu[current_slot]

        self._wait_for_pending_backward_repack()
        self.cpu_infer.submit_with_cuda_stream(
            cuda_stream,
            self._make_forward_task(buffer_view, save_for_backward=False),
        )

    def sync_forward_inference(self, cuda_stream) -> torch.Tensor:
        """
        Synchronize a serving forward submitted by submit_forward_inference().

        Returns a persistent GPU buffer matching the input batch shape. Consumers
        on the same CUDA stream will naturally wait for the non-blocking D2H/H2D
        staging work ordered through CPUInfer's stream synchronization.
        """
        if getattr(self, "_pending_inference_fallback", False):
            self._pending_inference_fallback = False
            output_device = getattr(self, "_pending_inference_fallback_device", None)
            self._pending_inference_fallback_device = None
            return self.sync_forward(output_device=output_device)

        if not hasattr(self, "_pending_inference_output_cpu"):
            raise RuntimeError("No pending inference forward. Call submit_forward_inference() first.")

        torch_stream = (
            cuda_stream
            if isinstance(cuda_stream, torch.cuda.Stream)
            else torch.cuda.ExternalStream(cuda_stream, device=self._pending_inference_output_gpu.device)
        )
        self.cpu_infer.sync_with_cuda_stream(cuda_stream)
        with torch.cuda.stream(torch_stream):
            self._pending_inference_output_gpu.copy_(self._pending_inference_output_cpu, non_blocking=True)
        output = self._pending_inference_output_gpu

        del self._pending_inference_output_cpu
        del self._pending_inference_output_gpu
        return output

    # ========== Async backward ==========

    def submit_backward_async(
        self,
        grad_output: torch.Tensor,
        output_device: Optional[torch.device] = None,
        optimizer_grad_scale: float = 1.0,
    ) -> None:
        """Submit backward task without waiting. Call sync_backward() for results."""
        if self._cache_depth <= 0:
            raise RuntimeError("No forward cache available. Call forward(save_for_backward=True) first.")
        if self._uses_authoritative_optimizer_grads and self._authoritative_grad_submission_pending:
            raise RuntimeError("An authoritative optimizer-gradient backward submission is already pending")
        optimizer_grad_scale = float(optimizer_grad_scale)
        if not math.isfinite(optimizer_grad_scale) or optimizer_grad_scale <= 0.0:
            raise ValueError(f"optimizer_grad_scale must be finite and positive, got {optimizer_grad_scale}")

        qlen = grad_output.shape[0]
        buffer = self._get_buffer(qlen)
        self._copy_grad_output_to_cpu(buffer, grad_output, qlen)

        use_authoritative = self._uses_authoritative_optimizer_grads
        accumulate_optimizer_grads = False
        if use_authoritative:
            accumulate_optimizer_grads = self._prepare_authoritative_optimizer_grad_write(optimizer_grad_scale)
        try:
            if use_authoritative:
                backward_task = self._make_backward_task(
                    buffer,
                    accumulate_optimizer_grads=accumulate_optimizer_grads,
                    optimizer_grad_scale=optimizer_grad_scale,
                )
            elif optimizer_grad_scale != 1.0:
                backward_task = self._make_backward_task(
                    buffer,
                    accumulate_optimizer_grads=False,
                    optimizer_grad_scale=optimizer_grad_scale,
                )
            else:
                backward_task = self._make_backward_task(buffer)
            self._wait_for_pending_backward_repack()
            self.cpu_infer.submit(backward_task)
        except Exception:
            if use_authoritative:
                # submit() is expected to be atomic, but drain defensively in
                # case a backend queued work before reporting an error.
                try:
                    self.cpu_infer.sync()
                except Exception:
                    pass
                self._abort_authoritative_optimizer_grad_write()
                self._cache_depth = max(0, self._cache_depth - 1)
            self._async_bwd_qlen = None
            self._async_bwd_output_device = None
            self._async_bwd_uses_authoritative = False
            raise
        self._async_bwd_qlen = qlen
        self._async_bwd_output_device = output_device
        self._async_bwd_uses_authoritative = use_authoritative

    def sync_backward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wait for async backward and return results."""
        if not hasattr(self, "_async_bwd_qlen") or self._async_bwd_qlen is None:
            raise RuntimeError("No pending backward. Call submit_backward_async() first.")

        use_authoritative = getattr(self, "_async_bwd_uses_authoritative", False)
        try:
            self.cpu_infer.sync()
            qlen = self._async_bwd_qlen
            output_device = self._async_bwd_output_device
            buffer = self._get_buffer(qlen)
            result = self._return_grads(buffer, qlen, output_device)
            if use_authoritative:
                self._publish_authoritative_optimizer_grads()
        except Exception:
            if use_authoritative:
                self._abort_authoritative_optimizer_grad_write()
                self._cache_depth = max(0, self._cache_depth - 1)
            self._async_bwd_qlen = None
            self._async_bwd_output_device = None
            self._async_bwd_uses_authoritative = False
            raise

        self._cache_depth -= 1
        self._async_bwd_qlen = None
        self._async_bwd_output_device = None
        self._async_bwd_uses_authoritative = False
        return result

    # ========== Backward repack (optional, subclasses may override) ==========

    def submit_backward_repack(self):
        if not self._weights_loaded or self.moe is None:
            return
        if hasattr(self.moe, "submit_backward_repack"):
            self.moe.submit_backward_repack()
            self._backward_repack_pending = True

    def wait_backward_repack(self):
        if not self._weights_loaded or self.moe is None:
            self._backward_repack_pending = False
            return
        if hasattr(self.moe, "wait_backward_repack"):
            self.moe.wait_backward_repack()
        self._backward_repack_pending = False
