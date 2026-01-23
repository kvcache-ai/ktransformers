# AMX SFT MoE Wrapper classes for CPU-based fine-tuning operations
# SPDX-License-Identifier: Apache-2.0

"""
AMX-based SFT MoE Wrapper implementation.

Supports quantization methods:
- AMXBF16_SFT: BF16 precision training
- AMXINT8_SFT: INT8 quantization training
- AMXINT4_SFT: INT4 quantization training
- AMXINT4_KGroup_SFT: INT4 K-Group quantization training (AWQ/K2)
"""

import torch
from typing import Dict, Tuple, Optional, List

from kt_kernel_ext.moe import MOESFTConfig

from .loader import BF16SafeTensorLoader, SafeTensorLoader

try:
    from kt_kernel_ext.moe import (
        AMXBF16_SFT_MOE,
        AMXInt8_SFT_MOE,
        AMXInt4_SFT_MOE,
        AMXInt4_1_SFT_MOE,
        AMXInt4_1KGroup_SFT_MOE,
        AMXInt4_KGroup_SFT_MOE,
        # SkipLoRA variants (skip all LoRA computation in backward)
        AMXBF16_SFT_MOE_SkipLoRA,
        AMXInt8_SFT_MOE_SkipLoRA,
        AMXInt4_SFT_MOE_SkipLoRA,
        AMXInt4_1_SFT_MOE_SkipLoRA,
        AMXInt4_1KGroup_SFT_MOE_SkipLoRA,
        AMXInt4_KGroup_SFT_MOE_SkipLoRA,
    )

    _HAS_AMX_SFT_SUPPORT = True
except (ImportError, AttributeError):
    _HAS_AMX_SFT_SUPPORT = False
    AMXBF16_SFT_MOE = None
    AMXInt8_SFT_MOE = None
    AMXInt4_SFT_MOE = None
    AMXInt4_1_SFT_MOE = None
    AMXInt4_1KGroup_SFT_MOE = None
    AMXInt4_KGroup_SFT_MOE = None
    # SkipLoRA variants
    AMXBF16_SFT_MOE_SkipLoRA = None
    AMXInt8_SFT_MOE_SkipLoRA = None
    AMXInt4_SFT_MOE_SkipLoRA = None
    AMXInt4_1_SFT_MOE_SkipLoRA = None
    AMXInt4_1KGroup_SFT_MOE_SkipLoRA = None
    AMXInt4_KGroup_SFT_MOE_SkipLoRA = None

from ..experts_sft import BaseSFTMoEWrapper, KExpertsSFTBuffer


# Mapping from method string to C++ SFT MOE class
_SFT_METHOD_TO_CLASS = {
    "AMXBF16_SFT": AMXBF16_SFT_MOE,
    "AMXINT8_SFT": AMXInt8_SFT_MOE,
    "AMXINT4_SFT": AMXInt4_SFT_MOE,
    "AMXINT4_1_SFT": AMXInt4_1_SFT_MOE,
    "AMXINT4_KGroup_SFT": AMXInt4_KGroup_SFT_MOE,
    "AMXINT4_1KGroup_SFT": AMXInt4_1KGroup_SFT_MOE,
    # SkipLoRA variants (skip all LoRA computation in backward, only compute base weight grad_input)
    "AMXBF16_SFT_SkipLoRA": AMXBF16_SFT_MOE_SkipLoRA,
    "AMXINT8_SFT_SkipLoRA": AMXInt8_SFT_MOE_SkipLoRA,
    "AMXINT4_SFT_SkipLoRA": AMXInt4_SFT_MOE_SkipLoRA,
    "AMXINT4_1_SFT_SkipLoRA": AMXInt4_1_SFT_MOE_SkipLoRA,
    "AMXINT4_KGroup_SFT_SkipLoRA": AMXInt4_KGroup_SFT_MOE_SkipLoRA,
    "AMXINT4_1KGroup_SFT_SkipLoRA": AMXInt4_1KGroup_SFT_MOE_SkipLoRA,
}


class AMXSFTMoEWrapper(BaseSFTMoEWrapper):
    """
    AMX-based SFT MoE wrapper implementation.

    Supports BF16, INT8, INT4, and INT4 K-Group quantization methods
    for supervised fine-tuning with LoRA adapters.

    Design Note (forward_sft vs forward):
        forward_sft() is implemented independently from inference forward() because:
        1. Different requirements: inference optimizes for latency, SFT requires gradient correctness
        2. Safety: inference optimizations (deferred experts, async execution) would break SFT gradients
        3. Most reusable optimizations are already in C++ layer (via inheritance)
        4. Manual copying of useful optimizations is safer and more maintainable
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
        method: str = "AMXBF16_SFT",
        # Quantization config (for K-Group methods)
        group_size: int = 128,
        zero_point: bool = True,
    ):
        """
        Initialize AMX SFT MoE Wrapper.

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
            method: AMX quantization method for SFT
            group_size: Quantization group size (for K-Group methods)
            zero_point: Whether to use zero point quantization (for K-Group methods)
        """
        if not _HAS_AMX_SFT_SUPPORT:
            raise RuntimeError(
                "AMX SFT backend not available. kt_kernel_ext was not compiled with AMX SFT support.\n"
                "Please recompile with AMX SFT enabled."
            )

        # Initialize base class
        super().__init__(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_gpu_experts=num_gpu_experts,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_cache_depth=max_cache_depth,
        )

        # Store method and quantization config
        self.method = method
        self.group_size = group_size
        self.zero_point = zero_point

        # Validate method
        if method not in _SFT_METHOD_TO_CLASS:
            raise ValueError(
                f"Unknown SFT method: {method}. " f"Supported methods: {list(_SFT_METHOD_TO_CLASS.keys())}"
            )

        # Get the C++ class for this method
        moe_class = _SFT_METHOD_TO_CLASS[method]
        if moe_class is None:
            raise RuntimeError(f"AMX SFT method '{method}' not available in current build.")

        # Base weight storage (set via load_weights_from_tensors or loaded from file)
        self.gate_proj: Optional[torch.Tensor] = None
        self.up_proj: Optional[torch.Tensor] = None
        self.down_proj: Optional[torch.Tensor] = None

        # MoE instance will be created during load_weights
        self._moe_class = moe_class

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor) -> None:
        """
        Load base weights for this layer.

        Supports two loading modes:
        1. From tensors: Call load_weights_from_tensors() first, then load_weights()
        2. From files: Automatically load from weight_path if base weights not set
           - AMXBF16_SFT: Uses BF16SafeTensorLoader (HuggingFace format)
           - AMXINT8_SFT/AMXINT4_SFT: Uses SafeTensorLoader (pre-quantized format)

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        if self._weights_loaded:
            return

        # If base weights not set, try to load from file
        if self.gate_proj is None or self.up_proj is None or self.down_proj is None:
            self._load_base_weights_from_file()

        # Create MOE SFT config
        config = MOESFTConfig()
        config.expert_num = self.num_experts
        config.num_experts_per_tok = self.num_experts_per_tok
        config.hidden_size = self.hidden_size
        config.intermediate_size = self.moe_intermediate_size
        config.lora_rank = self.lora_rank
        config.lora_alpha = self.lora_alpha
        config.max_cache_depth = self.max_cache_depth
        config.max_len = self.chunked_prefill_size
        config.layer_idx = self.layer_idx

        # Set base weight pointers
        config.gate_proj = self.gate_proj.data_ptr()
        config.up_proj = self.up_proj.data_ptr()
        config.down_proj = self.down_proj.data_ptr()

        # Set LoRA weight pointers (if initialized)
        if self._lora_initialized:
            config.gate_lora_a = self.gate_lora_a.data_ptr()
            config.gate_lora_b = self.gate_lora_b.data_ptr()
            config.up_lora_a = self.up_lora_a.data_ptr()
            config.up_lora_b = self.up_lora_b.data_ptr()
            config.down_lora_a = self.down_lora_a.data_ptr()
            config.down_lora_b = self.down_lora_b.data_ptr()

        # Set thread pool
        config.pool = self.cpu_infer.backend_

        # Set quantization config for K-Group methods
        if self.method in ("AMXINT4_KGroup_SFT", "AMXINT4_1KGroup_SFT"):
            config.quant_config.group_size = self.group_size
            config.quant_config.zero_point = self.zero_point

        # Create MoE instance
        self.moe = self._moe_class(config)

        # Load weights
        self.cpu_infer.submit(self.moe.load_weights_task())
        self.cpu_infer.sync()

        # Warm up
        self.cpu_infer.submit(self.moe.warm_up_task())
        self.cpu_infer.sync()

        self._weights_loaded = True

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ) -> None:
        """
        Load weights from BF16/FP16 tensors.

        This is the recommended way to load weights for SFT, as it supports
        online quantization from full-precision weights.

        Args:
            gate_proj: Gate projection weights [num_experts, intermediate_size, hidden_size]
            up_proj: Up projection weights [num_experts, intermediate_size, hidden_size]
            down_proj: Down projection weights [num_experts, hidden_size, intermediate_size]
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        # Store tensors as instance variables to keep them alive
        self.gate_proj = gate_proj.contiguous()
        self.up_proj = up_proj.contiguous()
        self.down_proj = down_proj.contiguous()

        # Now load weights
        self.load_weights(physical_to_logical_map_cpu)

    def _load_base_weights_from_file(self) -> None:
        """
        Load base MoE weights from file based on the SFT method.

        Loading strategy:
        - AMXBF16_SFT: Use BF16SafeTensorLoader (HuggingFace format, no scales)
        - AMXINT8_SFT/AMXINT4_SFT: Use SafeTensorLoader (pre-quantized format with scales)
        """
        if not hasattr(self, "weight_path") or self.weight_path is None:
            raise RuntimeError(
                "weight_path not set. Cannot load weights from file. "
                "Either set weight_path or call load_weights_from_tensors() instead."
            )

        print(
            f"[AMXSFTMoEWrapper] Loading base weights for layer {self.layer_idx} "
            f"from {self.weight_path} using method {self.method}"
        )

        # Determine loader and base key format based on method
        if self.method == "AMXBF16_SFT":
            # BF16 mode: Load from HuggingFace model path
            loader = BF16SafeTensorLoader(self.weight_path)
            base_key = f"model.layers.{self.layer_idx}"
        else:
            # INT8/INT4 mode: Load from pre-quantized path
            # Note: SafeTensorLoader expects GGUF-style naming (blk.X)
            loader = SafeTensorLoader(self.weight_path)
            base_key = f"blk.{self.layer_idx}"

        # Load expert weights
        experts_data = loader.load_experts(base_key, device="cpu")

        # Extract weights (list of tensors per expert -> stacked tensor)
        gate_weights: List[torch.Tensor] = experts_data["gate"]
        up_weights: List[torch.Tensor] = experts_data["up"]
        down_weights: List[torch.Tensor] = experts_data["down"]

        # Stack expert weights: [num_experts, ...]
        # For BF16: weights are already tensors
        # For SafeTensorLoader: weights might be numpy arrays in nested lists
        if self.method == "AMXBF16_SFT":
            # BF16SafeTensorLoader returns list of tensors
            self.gate_proj = torch.stack(gate_weights, dim=0).contiguous()
            self.up_proj = torch.stack(up_weights, dim=0).contiguous()
            self.down_proj = torch.stack(down_weights, dim=0).contiguous()
        else:
            # SafeTensorLoader returns nested lists [numa_id][expert_id] -> numpy array
            # For SFT, we typically use numa_id=0
            import numpy as np

            numa_id = 0
            self.gate_proj = torch.from_numpy(np.stack(gate_weights[numa_id], axis=0)).contiguous()
            self.up_proj = torch.from_numpy(np.stack(up_weights[numa_id], axis=0)).contiguous()
            self.down_proj = torch.from_numpy(np.stack(down_weights[numa_id], axis=0)).contiguous()

            # Also store scales for INT8/INT4 methods
            self.gate_scale = torch.from_numpy(np.stack(experts_data["gate_scale"][numa_id], axis=0)).contiguous()
            self.up_scale = torch.from_numpy(np.stack(experts_data["up_scale"][numa_id], axis=0)).contiguous()
            self.down_scale = torch.from_numpy(np.stack(experts_data["down_scale"][numa_id], axis=0)).contiguous()

        # Close loader handles
        loader.close_all_handles()

        print(
            f"[AMXSFTMoEWrapper] Loaded weights: gate_proj={self.gate_proj.shape}, "
            f"up_proj={self.up_proj.shape}, down_proj={self.down_proj.shape}"
        )

    def init_lora_weights(
        self,
        gate_lora_a: torch.Tensor,
        gate_lora_b: torch.Tensor,
        up_lora_a: torch.Tensor,
        up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
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
        # Validate shapes
        expected_shapes = {
            "gate_lora_a": (self.num_experts, self.lora_rank, self.hidden_size),
            "gate_lora_b": (self.num_experts, self.moe_intermediate_size, self.lora_rank),
            "up_lora_a": (self.num_experts, self.lora_rank, self.hidden_size),
            "up_lora_b": (self.num_experts, self.moe_intermediate_size, self.lora_rank),
            "down_lora_a": (self.num_experts, self.lora_rank, self.moe_intermediate_size),
            "down_lora_b": (self.num_experts, self.hidden_size, self.lora_rank),
        }

        provided_tensors = {
            "gate_lora_a": gate_lora_a,
            "gate_lora_b": gate_lora_b,
            "up_lora_a": up_lora_a,
            "up_lora_b": up_lora_b,
            "down_lora_a": down_lora_a,
            "down_lora_b": down_lora_b,
        }

        for name, tensor in provided_tensors.items():
            expected = expected_shapes[name]
            if tensor.shape != expected:
                raise ValueError(f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}")

        # Store LoRA weights (contiguous for C++ access)
        self.gate_lora_a = gate_lora_a.contiguous()
        self.gate_lora_b = gate_lora_b.contiguous()
        self.up_lora_a = up_lora_a.contiguous()
        self.up_lora_b = up_lora_b.contiguous()
        self.down_lora_a = down_lora_a.contiguous()
        self.down_lora_b = down_lora_b.contiguous()

        self._lora_initialized = True

        # If weights already loaded, update LoRA pointers in C++
        if self._weights_loaded and self.moe is not None:
            self.update_lora_weights()

    def forward_sft(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        save_for_backward: bool = True,
    ) -> torch.Tensor:
        """
        SFT forward pass with optional gradient caching.

        Args:
            hidden_states: Input hidden states [qlen, hidden_size]
            expert_ids: Expert IDs [qlen, num_experts_per_tok]
            weights: Expert weights [qlen, num_experts_per_tok]
            save_for_backward: Whether to save activations for backward pass

        Returns:
            Output hidden states [qlen, hidden_size]
        """
        if not self._weights_loaded:
            raise RuntimeError("Weights not loaded. Call load_weights() or load_weights_from_tensors() first.")

        if not self._lora_initialized:
            raise RuntimeError("LoRA weights not initialized. Call init_lora_weights() first.")

        qlen = hidden_states.shape[0]
        if qlen > self.chunked_prefill_size:
            raise ValueError(
                f"qlen ({qlen}) exceeds chunked_prefill_size ({self.chunked_prefill_size}). "
                "Increase chunked_prefill_size or reduce qlen to avoid buffer overrun."
            )
        if expert_ids.shape[0] != qlen or expert_ids.shape[1] != self.num_experts_per_tok:
            raise ValueError(
                f"expert_ids shape {tuple(expert_ids.shape)} must be " f"({qlen}, {self.num_experts_per_tok})."
            )
        if weights.shape[0] != qlen or weights.shape[1] != self.num_experts_per_tok:
            raise ValueError(f"weights shape {tuple(weights.shape)} must be " f"({qlen}, {self.num_experts_per_tok}).")

        # Get or create buffer
        buffer = KExpertsSFTBuffer.get_buffer(
            qlen=qlen,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            lora_rank=self.lora_rank,
            dtype=hidden_states.dtype,
        )

        # Copy input data to CPU buffers
        buffer.input_cpu.copy_(hidden_states, non_blocking=True)
        buffer.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
        buffer.weights_cpu.copy_(weights, non_blocking=True)
        buffer.bsz_tensor[0] = qlen

        # Submit forward task
        self.cpu_infer.submit(
            self.moe.forward_sft_task(
                buffer.bsz_tensor.data_ptr(),
                self.num_experts_per_tok,
                buffer.expert_ids_cpu.data_ptr(),
                buffer.weights_cpu.data_ptr(),
                buffer.input_cpu.data_ptr(),
                buffer.output_cpu.data_ptr(),
                save_for_backward,
            )
        )
        self.cpu_infer.sync()

        # Track cache depth
        if save_for_backward:
            self._cache_depth += 1
            if self._cache_depth > self.max_cache_depth:
                raise RuntimeError(
                    f"Forward cache full (depth={self._cache_depth}, max={self.max_cache_depth}). "
                    "Call backward() to release cache entries."
                )

        return buffer.output_cpu.clone()

    def backward(
        self,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Backward pass computing gradients.

        Must be called after forward_sft(save_for_backward=True).

        Args:
            grad_output: Gradient from upstream [qlen, hidden_size]

        Returns:
            grad_input: Input gradient [qlen, hidden_size] (BF16)
            grad_loras: Dictionary of LoRA gradients containing:
                - grad_gate_lora_a: [num_experts, lora_rank, hidden_size] (BF16)
                - grad_gate_lora_b: [num_experts, intermediate_size, lora_rank] (BF16)
                - grad_up_lora_a: [num_experts, lora_rank, hidden_size] (BF16)
                - grad_up_lora_b: [num_experts, intermediate_size, lora_rank] (BF16)
                - grad_down_lora_a: [num_experts, lora_rank, intermediate_size] (BF16)
                - grad_down_lora_b: [num_experts, hidden_size, lora_rank] (BF16)
            grad_weights: Routing weights gradient [qlen, num_experts_per_tok] (FP32)
        """
        if self._cache_depth <= 0:
            raise RuntimeError("No forward cache available. Call forward_sft(save_for_backward=True) first.")

        qlen = grad_output.shape[0]

        # Get buffer (should exist from forward pass)
        buffer = KExpertsSFTBuffer.get_buffer(
            qlen=qlen,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            lora_rank=self.lora_rank,
            dtype=grad_output.dtype,
        )

        # Copy gradient to CPU buffer
        buffer.grad_output_cpu.copy_(grad_output, non_blocking=True)

        # Zero out gradient buffers
        buffer.grad_input_cpu.zero_()
        buffer.grad_gate_lora_a.zero_()
        buffer.grad_gate_lora_b.zero_()
        buffer.grad_up_lora_a.zero_()
        buffer.grad_up_lora_b.zero_()
        buffer.grad_down_lora_a.zero_()
        buffer.grad_down_lora_b.zero_()
        buffer.grad_weights.zero_()

        # Submit backward task
        self.cpu_infer.submit(
            self.moe.backward_task(
                buffer.grad_output_cpu.data_ptr(),
                buffer.grad_input_cpu.data_ptr(),
                buffer.grad_gate_lora_a.data_ptr(),
                buffer.grad_gate_lora_b.data_ptr(),
                buffer.grad_up_lora_a.data_ptr(),
                buffer.grad_up_lora_b.data_ptr(),
                buffer.grad_down_lora_a.data_ptr(),
                buffer.grad_down_lora_b.data_ptr(),
                buffer.grad_weights.data_ptr(),
            )
        )
        self.cpu_infer.sync()

        # Decrease cache depth
        self._cache_depth -= 1

        # Return cloned gradients
        grad_input = buffer.grad_input_cpu.clone()
        grad_loras = {
            "grad_gate_lora_a": buffer.grad_gate_lora_a.clone(),
            "grad_gate_lora_b": buffer.grad_gate_lora_b.clone(),
            "grad_up_lora_a": buffer.grad_up_lora_a.clone(),
            "grad_up_lora_b": buffer.grad_up_lora_b.clone(),
            "grad_down_lora_a": buffer.grad_down_lora_a.clone(),
            "grad_down_lora_b": buffer.grad_down_lora_b.clone(),
        }
        grad_weights = buffer.grad_weights.clone()

        return grad_input, grad_loras, grad_weights

    def update_lora_weights(self) -> None:
        """
        Sync LoRA weights to C++ backend.

        Call this after using an external optimizer to update LoRA weights.
        This is needed because TP mode partitions weights internally.

        Typical usage:
            # 1. Forward + backward
            output = wrapper.forward_sft(input, expert_ids, weights)
            grad_input, grad_loras = wrapper.backward(grad_output)

            # 2. Update LoRA weights with optimizer
            optimizer.step()

            # 3. Sync to C++
            wrapper.update_lora_weights()
        """
        if not self._weights_loaded:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        if not self._lora_initialized:
            raise RuntimeError("LoRA weights not initialized. Call init_lora_weights() first.")

        # Submit update task
        self.cpu_infer.submit(
            self.moe.update_lora_weights_task(
                self.gate_lora_a.data_ptr(),
                self.gate_lora_b.data_ptr(),
                self.up_lora_a.data_ptr(),
                self.up_lora_b.data_ptr(),
                self.down_lora_a.data_ptr(),
                self.down_lora_b.data_ptr(),
            )
        )
        self.cpu_infer.sync()
