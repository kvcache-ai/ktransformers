# SFT MoE Wrapper classes for CPU-based fine-tuning operations
# SPDX-License-Identifier: Apache-2.0

"""
SFT (Supervised Fine-Tuning) MoE Wrapper classes and buffer management.

This module provides:
- KExpertsSFTBuffer: Buffer management for SFT forward/backward passes
- BaseSFTMoEWrapper: Abstract base class for SFT MoE wrappers

Key differences from inference wrappers:
- Supports forward_sft() with gradient caching for backward pass
- Supports backward() for computing LoRA gradients
- Uses synchronous execution (no double buffering)
- Independent from inference forward() logic to ensure gradient correctness
"""

from __future__ import annotations

import torch
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .experts_base import _MoEBase


class KExpertsSFTBuffer:
    """
    CPU buffer management for SFT expert computation.

    Unlike inference KExpertsCPUBuffer:
    - No double buffering (SFT requires synchronous execution)
    - Includes gradient buffers for backward pass
    - Includes 6 LoRA gradient buffers

    Buffer contents:
    - Forward: input_cpu, expert_ids_cpu, weights_cpu, output_cpu
    - Backward: grad_output_cpu, grad_input_cpu
    - LoRA gradients: grad_gate_lora_a/b, grad_up_lora_a/b, grad_down_lora_a/b
    """

    capture_buffers: Dict[tuple, "KExpertsSFTBuffer"] = {}

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
        """
        Initialize SFT buffer.

        Args:
            qlen: Sequence length (batch size)
            hidden_size: Hidden dimension
            moe_intermediate_size: MoE intermediate dimension
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token
            lora_rank: LoRA rank
            dtype: Data type for buffers
        """
        self.qlen = qlen
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.lora_rank = lora_rank
        self.dtype = dtype

        pin_memory = False

        # ========== Forward buffers ==========
        self.input_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.expert_ids_cpu = torch.empty(
            (qlen, num_experts_per_tok), dtype=torch.int64, device="cpu", pin_memory=pin_memory
        )
        self.weights_cpu = torch.empty(
            (qlen, num_experts_per_tok), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.output_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)

        # ========== Backward buffers ==========
        self.grad_output_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.grad_input_cpu = torch.empty((qlen, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory)

        # Routing weights gradient [qlen, num_experts_per_tok] (FP32)
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
        """
        Get or create SFT buffer (with caching).

        Uses parameter combination as cache key to reuse buffers.

        Args:
            qlen: Sequence length
            hidden_size: Hidden dimension
            moe_intermediate_size: MoE intermediate dimension
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token
            lora_rank: LoRA rank
            dtype: Data type

        Returns:
            KExpertsSFTBuffer instance
        """
        key = (qlen, hidden_size, moe_intermediate_size, num_experts, num_experts_per_tok, lora_rank, dtype)

        if key not in cls.capture_buffers:
            cls.capture_buffers[key] = cls(
                qlen=qlen,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                lora_rank=lora_rank,
                dtype=dtype,
            )

        return cls.capture_buffers[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached buffers."""
        cls.capture_buffers.clear()


class BaseSFTMoEWrapper(_MoEBase, ABC):
    """
    Base class for SFT MoE CPU operations.

    Provides LoRA fine-tuning functionality including:
    - forward_sft(): Forward pass with gradient caching
    - backward(): Backward pass computing LoRA gradients
    - update_lora_weights(): Sync LoRA weights to C++ backend

    Key differences from BaseMoEWrapper (inference):
    - Uses synchronous execution (no double buffering)
    - Maintains forward cache for backward pass
    - Independent forward_sft() implementation (not sharing inference forward())

    Design Decision (forward_sft vs forward relationship):
    forward_sft() is implemented independently from forward() because:
    1. Different requirements: inference optimizes for latency, SFT requires gradient correctness
    2. Safety: inference optimizations (deferred experts, async execution) would break SFT gradients
    3. Most reusable optimizations are already in C++ layer (via inheritance)
    4. Manual copying of useful optimizations is safer and more maintainable

    Attributes:
        lora_rank: LoRA low-rank matrix rank
        lora_alpha: LoRA scaling factor
        lora_scaling: Actual scaling value (lora_alpha / lora_rank)
        max_cache_depth: Maximum forward cache depth for gradient checkpointing
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
        # SFT-specific parameters
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        max_cache_depth: int = 1,
    ):
        """
        Initialize SFT MoE Wrapper.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            num_gpu_experts: Number of experts on GPU (usually 0 for SFT)
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            lora_rank: LoRA rank (r)
            lora_alpha: LoRA scaling factor (alpha)
            max_cache_depth: Maximum forward cache depth
        """
        # Get shared CPUInfer instance
        self.cpu_infer = self._get_cpu_infer(cpuinfer_threads, threadpool_count)

        # Validate basic configuration
        self._validate_base_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
        )

        # Validate SFT-specific parameters
        self._validate_sft_config(lora_rank, lora_alpha, max_cache_depth)

        # Save configuration
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_gpu_experts = num_gpu_experts
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.threadpool_count = threadpool_count

        # SFT-specific configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank
        self.max_cache_depth = max_cache_depth

        # LoRA weight placeholders (set via init_lora_weights)
        self.gate_lora_a: Optional[torch.Tensor] = None
        self.gate_lora_b: Optional[torch.Tensor] = None
        self.up_lora_a: Optional[torch.Tensor] = None
        self.up_lora_b: Optional[torch.Tensor] = None
        self.down_lora_a: Optional[torch.Tensor] = None
        self.down_lora_b: Optional[torch.Tensor] = None

        # State tracking
        self._weights_loaded: bool = False
        self._lora_initialized: bool = False
        self._cache_depth: int = 0

        # Backend-specific initialization happens in subclasses
        self.moe = None

    @staticmethod
    def _validate_sft_config(lora_rank: int, lora_alpha: float, max_cache_depth: int) -> None:
        """
        Validate SFT-specific parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {lora_rank}")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
        if max_cache_depth <= 0:
            raise ValueError(f"max_cache_depth must be positive, got {max_cache_depth}")

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor) -> None:
        """
        Load base weights for this layer.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

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
    ) -> None:
        """
        Initialize LoRA weights.

        LoRA output formula:
            lora_output = (input @ A.T @ B.T) * (lora_alpha / lora_rank)
            output = base_output + lora_output

        Args:
            gate_lora_a: Gate LoRA A matrix [num_experts, lora_rank, hidden_size]
            gate_lora_b: Gate LoRA B matrix [num_experts, intermediate_size, lora_rank]
            up_lora_a: Up LoRA A matrix [num_experts, lora_rank, hidden_size]
            up_lora_b: Up LoRA B matrix [num_experts, intermediate_size, lora_rank]
            down_lora_a: Down LoRA A matrix [num_experts, lora_rank, intermediate_size]
            down_lora_b: Down LoRA B matrix [num_experts, hidden_size, lora_rank]
        """
        pass

    @abstractmethod
    def forward_sft(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        save_for_backward: bool = True,
        output_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        SFT forward pass with optional gradient caching.

        Optimized for minimal data copying:
        - Accepts GPU tensors directly, copies to pinned buffer in one step
        - Returns directly to output_device without intermediate clone

        Args:
            hidden_states: Input hidden states [qlen, hidden_size] (any device)
            expert_ids: Expert IDs [qlen, num_experts_per_tok] (any device)
            weights: Expert weights [qlen, num_experts_per_tok] (any device)
            save_for_backward: Whether to save activations for backward pass
            output_device: Target device for output (None = clone CPU tensor)

        Returns:
            Output hidden states [qlen, hidden_size]
        """
        pass

    @abstractmethod
    def backward(
        self,
        grad_output: torch.Tensor,
        output_device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass computing gradients.

        Must be called after forward_sft(save_for_backward=True).

        Optimized for minimal data copying:
        - Accepts GPU tensors directly
        - Returns grad_input directly to output_device without intermediate clone
        - LoRA gradients are returned in grad_loras dict (no clone needed)

        Args:
            grad_output: Gradient from upstream [qlen, hidden_size] (any device)
            lora_params: Optional dict of LoRA parameters (kept for compatibility).
                         If provided, gradients are still returned in grad_loras.
                         Keys: gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b
            output_device: Target device for grad_input (None = clone CPU tensor)

        Returns:
            grad_input: Input gradient [qlen, hidden_size]
            grad_loras: LoRA gradients dict (e.g., grad_gate_lora_a, grad_gate_lora_b, ...)
            grad_weights: Routing weights gradient [qlen, num_experts_per_tok]
        """
        pass

    @abstractmethod
    def update_lora_weights(self) -> None:
        """
        Sync LoRA weights to C++ backend.

        Call this after using an external optimizer to update LoRA weights.
        This is a zero-copy operation that passes Python tensor pointers.

        Typical usage:
            # 1. Forward + backward
            output = wrapper.forward_sft(input, expert_ids, weights)
            grad_input, grad_loras = wrapper.backward(grad_output)

            # 2. Update LoRA weights with optimizer
            optimizer.step()

            # 3. Sync to C++
            wrapper.update_lora_weights()
        """
        pass

    @abstractmethod
    def submit_forward_sft(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        save_for_backward: bool = True,
    ) -> None:
        """
        Submit SFT forward pass asynchronously (non-blocking).

        This method submits the CPU MoE computation without waiting for completion,
        allowing GPU computation (shared_experts, lora_experts) to proceed in parallel.

        Must be followed by sync_forward_sft() to retrieve results.

        Args:
            hidden_states: Input hidden states [qlen, hidden_size]
            expert_ids: Expert IDs [qlen, num_experts_per_tok]
            weights: Expert weights [qlen, num_experts_per_tok]
            save_for_backward: Whether to save activations for backward pass
        """
        pass

    @abstractmethod
    def sync_forward_sft(self, output_device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Synchronize and retrieve SFT forward results.

        Must be called after submit_forward_sft().

        Args:
            output_device: Target device for output (None = clone CPU tensor)

        Returns:
            Output hidden states [qlen, hidden_size]
        """
        pass

    # ========== Inference methods (not available in SFT mode) ==========

    def forward(self, *args, **kwargs):
        """Inference forward is not available in SFT mode."""
        raise RuntimeError("forward() is not available in SFT mode. " "Use forward_sft() instead.")

    def submit_forward(self, *args, **kwargs):
        """Async submit is not available in SFT mode."""
        raise RuntimeError("submit_forward() is not available in SFT mode. " "Use submit_forward_sft() instead.")

    def sync_forward(self, *args, **kwargs):
        """Async sync is not available in SFT mode."""
        raise RuntimeError("sync_forward() is not available in SFT mode. " "Use sync_forward_sft() instead.")

    def select_deferred_experts(self, *args, **kwargs):
        """Deferred experts is not available in SFT mode."""
        raise RuntimeError(
            "select_deferred_experts() is not available in SFT mode. "
            "SFT requires all experts for gradient computation."
        )
