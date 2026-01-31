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

import ctypes
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
        if self.gate_proj is None and not getattr(self, "_use_projs_path", False):
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
        if getattr(self, "_use_projs_path", False):
            # Pre-quantized per-NUMA per-expert path (INT8/INT4)
            config.gate_projs = self._gate_projs_ptrs
            config.up_projs = self._up_projs_ptrs
            config.down_projs = self._down_projs_ptrs
            config.gate_scales = self._gate_scale_ptrs
            config.up_scales = self._up_scale_ptrs
            config.down_scales = self._down_scale_ptrs
        else:
            # Flat BF16 buffer path
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
            # Keep per-NUMA per-expert arrays for gate_projs/gate_scales path
            import numpy as np

            num_numa = len(gate_weights)

            # Store raw per-NUMA per-expert numpy arrays (keep references alive)
            self._gate_weights_per_numa = gate_weights  # [numa_id][expert_id] -> np array
            self._up_weights_per_numa = up_weights
            self._down_weights_per_numa = down_weights
            self._gate_scales_per_numa = experts_data["gate_scale"]
            self._up_scales_per_numa = experts_data["up_scale"]
            self._down_scales_per_numa = experts_data["down_scale"]

            # Build pointer arrays: [[ptr_expert_0, ptr_expert_1, ...], ...] per NUMA
            def _make_ptrs(arrays_per_numa):
                return [
                    [
                        ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                        for et in numa_array
                    ]
                    for numa_array in arrays_per_numa
                ]

            self._gate_projs_ptrs = _make_ptrs(gate_weights)
            self._up_projs_ptrs = _make_ptrs(up_weights)
            self._down_projs_ptrs = _make_ptrs(down_weights)
            self._gate_scale_ptrs = _make_ptrs(experts_data["gate_scale"])
            self._up_scale_ptrs = _make_ptrs(experts_data["up_scale"])
            self._down_scale_ptrs = _make_ptrs(experts_data["down_scale"])

            # Set gate_proj to None so load_weights() uses gate_projs path
            self.gate_proj = None
            self.up_proj = None
            self.down_proj = None
            self._use_projs_path = True

        # Close loader handles
        loader.close_all_handles()

        if getattr(self, "_use_projs_path", False):
            num_numa = len(self._gate_weights_per_numa)
            num_experts = len(self._gate_weights_per_numa[0])
            print(
                f"[AMXSFTMoEWrapper] Loaded pre-quantized weights: "
                f"{num_numa} NUMA nodes, {num_experts} experts per NUMA"
            )
        else:
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

        self.grad_gate_lora_a = (
            torch.empty((self.num_experts, self.lora_rank, self.hidden_size), dtype=torch.bfloat16, device="cpu")
            .zero_()
            .contiguous()
        )
        self.grad_gate_lora_b = (
            torch.empty(
                (self.num_experts, self.moe_intermediate_size, self.lora_rank), dtype=torch.bfloat16, device="cpu"
            )
            .zero_()
            .contiguous()
        )

        self.grad_up_lora_a = (
            torch.empty((self.num_experts, self.lora_rank, self.hidden_size), dtype=torch.bfloat16, device="cpu")
            .zero_()
            .contiguous()
        )
        self.grad_up_lora_b = (
            torch.empty(
                (self.num_experts, self.moe_intermediate_size, self.lora_rank), dtype=torch.bfloat16, device="cpu"
            )
            .zero_()
            .contiguous()
        )

        self.grad_down_lora_a = (
            torch.empty(
                (self.num_experts, self.lora_rank, self.moe_intermediate_size), dtype=torch.bfloat16, device="cpu"
            )
            .zero_()
            .contiguous()
        )
        self.grad_down_lora_b = (
            torch.empty((self.num_experts, self.hidden_size, self.lora_rank), dtype=torch.bfloat16, device="cpu")
            .zero_()
            .contiguous()
        )

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
        output_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        SFT forward pass with optional gradient caching.

        Optimized for minimal data copying:
        - Accepts GPU tensors directly, copies to pinned buffer in one step
        - Returns directly to output_device without intermediate clone

        Args:
            hidden_states: Input hidden states [qlen, hidden_size] (any device, will be converted to bf16)
            expert_ids: Expert IDs [qlen, num_experts_per_tok] (any device, will be converted to int64)
            weights: Expert weights [qlen, num_experts_per_tok] (any device, will be converted to float32)
            save_for_backward: Whether to save activations for backward pass
            output_device: Target device for output (None = return CPU tensor without clone, caller must copy immediately)

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

        # Get or create buffer (always bf16 for computation)
        buffer = KExpertsSFTBuffer.get_buffer(
            qlen=qlen,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            lora_rank=self.lora_rank,
            dtype=torch.bfloat16,
        )

        # Copy input data directly to pinned CPU buffers (works for both CPU and GPU tensors)
        # For GPU tensors: this is a single GPU->pinned copy (faster than GPU->CPU->pinned)
        # For CPU tensors: this is a CPU->pinned copy
        input_device = hidden_states.device
        buffer.input_cpu.copy_(hidden_states.to(torch.bfloat16), non_blocking=True)
        buffer.expert_ids_cpu.copy_(expert_ids.to(torch.int64), non_blocking=True)
        buffer.weights_cpu.copy_(weights.to(torch.float32), non_blocking=True)
        buffer.bsz_tensor[0] = qlen

        # Synchronize CUDA stream if input was on GPU to ensure data has arrived
        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)

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

        # Return output: if output_device specified, copy directly to that device
        # This avoids clone() when transferring to GPU (pinned->GPU is fast)
        if output_device is not None:
            return buffer.output_cpu.to(device=output_device, non_blocking=True)
        else:
            # No output device specified: clone for safety (legacy behavior)
            return buffer.output_cpu.clone()

    def backward(
        self,
        grad_output: torch.Tensor,
        lora_params: Optional[Dict[str, torch.nn.Parameter]] = None,
        output_device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Backward pass computing gradients.

        Must be called after forward_sft(save_for_backward=True).

        Optimized for minimal data copying:
        - Accepts GPU tensors directly
        - Returns directly to output_device without intermediate clone
        - LoRA gradients are returned in grad_loras dict (no clone needed)

        Args:
            grad_output: Gradient from upstream [qlen, hidden_size] (any device, will be converted to bf16)
            lora_params: Optional dict of LoRA parameters (kept for compatibility).
                         If provided, gradients are still returned in grad_loras.
                         Keys: gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b
            output_device: Target device for grad_input output (None = clone CPU tensors for safety)

        Returns:
            grad_input: Input gradient [qlen, hidden_size]
            grad_loras: LoRA gradients dict (e.g., grad_gate_lora_a, grad_gate_lora_b, ...)
            grad_weights: Routing weights gradient [qlen, num_experts_per_tok]
        """
        if self._cache_depth <= 0:
            raise RuntimeError("No forward cache available. Call forward_sft(save_for_backward=True) first.")

        qlen = grad_output.shape[0]

        # Get buffer (should exist from forward pass, always bf16)
        buffer = KExpertsSFTBuffer.get_buffer(
            qlen=qlen,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            lora_rank=self.lora_rank,
            dtype=torch.bfloat16,
        )

        # Copy gradient directly to pinned CPU buffer (works for both CPU and GPU tensors)
        input_device = grad_output.device
        buffer.grad_output_cpu.copy_(grad_output.to(torch.bfloat16), non_blocking=True)

        # Zero out gradient buffers
        buffer.grad_input_cpu.zero_()
        buffer.grad_weights.zero_()

        # Synchronize CUDA stream if input was on GPU to ensure data has arrived
        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)

        # Submit backward task
        self.cpu_infer.submit(
            self.moe.backward_task(
                buffer.grad_output_cpu.data_ptr(),
                buffer.grad_input_cpu.data_ptr(),
                self.grad_gate_lora_a.data_ptr(),
                self.grad_gate_lora_b.data_ptr(),
                self.grad_up_lora_a.data_ptr(),
                self.grad_up_lora_b.data_ptr(),
                self.grad_down_lora_a.data_ptr(),
                self.grad_down_lora_b.data_ptr(),
                buffer.grad_weights.data_ptr(),
            )
        )
        self.cpu_infer.sync()

        # # Debug: print LoRA weights and computed gradients
        # print(f"\033[33m[AMX_SFT DEBUG] layer={self.layer_idx} backward "
        #       f"lora_a weights: gate={self.gate_lora_a.float().norm().item():.6f} "
        #       f"up={self.up_lora_a.float().norm().item():.6f} "
        #       f"down={self.down_lora_a.float().norm().item():.6f} | "
        #       f"lora_b weights: gate={self.gate_lora_b.float().norm().item():.6f} "
        #       f"up={self.up_lora_b.float().norm().item():.6f} "
        #       f"down={self.down_lora_b.float().norm().item():.6f} | "
        #       f"grad_a: gate={self.grad_gate_lora_a.float().norm().item():.6f} "
        #       f"up={self.grad_up_lora_a.float().norm().item():.6f} "
        #       f"down={self.grad_down_lora_a.float().norm().item():.6f} | "
        #       f"grad_b: gate={self.grad_gate_lora_b.float().norm().item():.6f} "
        #       f"up={self.grad_up_lora_b.float().norm().item():.6f} "
        #       f"down={self.grad_down_lora_b.float().norm().item():.6f}"
        #       f"\033[0m", flush=True)

        # Decrease cache depth
        self._cache_depth -= 1

        # Return gradients: if output_device specified, transfer grad_input directly
        if output_device is not None:
            grad_input = buffer.grad_input_cpu.to(device=output_device, non_blocking=True)
            grad_weights = buffer.grad_weights.to(device=output_device, non_blocking=True)
        else:
            # No output device: clone for safety (legacy behavior)
            grad_input = buffer.grad_input_cpu.clone()
            grad_weights = buffer.grad_weights.clone()

        grad_loras = {
            "grad_gate_lora_a": self.grad_gate_lora_a,
            "grad_gate_lora_b": self.grad_gate_lora_b,
            "grad_up_lora_a": self.grad_up_lora_a,
            "grad_up_lora_b": self.grad_up_lora_b,
            "grad_down_lora_a": self.grad_down_lora_a,
            "grad_down_lora_b": self.grad_down_lora_b,
        }

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
        # print(f"\033[36m[AMX_SFT DEBUG] layer={self.layer_idx} update_lora_weights "
        #       f"gate_lora_a: ptr={self.gate_lora_a.data_ptr()} norm={self.gate_lora_a.float().norm().item():.6f} "
        #       f"gate_lora_b: ptr={self.gate_lora_b.data_ptr()} norm={self.gate_lora_b.float().norm().item():.6f} "
        #       f"up_lora_a: ptr={self.up_lora_a.data_ptr()} norm={self.up_lora_a.float().norm().item():.6f} "
        #       f"up_lora_b: ptr={self.up_lora_b.data_ptr()} norm={self.up_lora_b.float().norm().item():.6f} "
        #       f"down_lora_a: ptr={self.down_lora_a.data_ptr()} norm={self.down_lora_a.float().norm().item():.6f} "
        #       f"down_lora_b: ptr={self.down_lora_b.data_ptr()} norm={self.down_lora_b.float().norm().item():.6f}"
        #       f"\033[0m", flush=True)
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

        Optimized: accepts GPU tensors directly, copies to pinned buffer in one step.

        Args:
            hidden_states: Input hidden states [qlen, hidden_size] (any device, will be converted to bf16)
            expert_ids: Expert IDs [qlen, num_experts_per_tok] (any device, will be converted to int64)
            weights: Expert weights [qlen, num_experts_per_tok] (any device, will be converted to float32)
            save_for_backward: Whether to save activations for backward pass
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

        # Get or create buffer (always bf16)
        buffer = KExpertsSFTBuffer.get_buffer(
            qlen=qlen,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            lora_rank=self.lora_rank,
            dtype=torch.bfloat16,
        )

        # Copy input data directly to pinned CPU buffers (works for both CPU and GPU tensors)
        input_device = hidden_states.device
        buffer.input_cpu.copy_(hidden_states.to(torch.bfloat16), non_blocking=True)
        buffer.expert_ids_cpu.copy_(expert_ids.to(torch.int64), non_blocking=True)
        buffer.weights_cpu.copy_(weights.to(torch.float32), non_blocking=True)
        buffer.bsz_tensor[0] = qlen

        # Synchronize CUDA stream if input was on GPU to ensure data has arrived
        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)

        # Store buffer reference and save_for_backward flag for sync_forward_sft
        self._pending_buffer = buffer
        self._pending_save_for_backward = save_for_backward
        self._pending_qlen = qlen

        # Submit forward task (non-blocking)
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

    def sync_forward_sft(self, output_device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Synchronize and retrieve SFT forward results.

        Must be called after submit_forward_sft().

        Args:
            output_device: Target device for output (None = clone CPU tensor for safety)

        Returns:
            Output hidden states [qlen, hidden_size]
        """
        if not hasattr(self, "_pending_buffer") or self._pending_buffer is None:
            raise RuntimeError("No pending forward. Call submit_forward_sft() first.")

        # Wait for completion
        self.cpu_infer.sync()

        buffer = self._pending_buffer
        save_for_backward = self._pending_save_for_backward

        # Track cache depth
        if save_for_backward:
            self._cache_depth += 1
            if self._cache_depth > self.max_cache_depth:
                raise RuntimeError(
                    f"Forward cache full (depth={self._cache_depth}, max={self.max_cache_depth}). "
                    "Call backward() to release cache entries."
                )

        # Clear pending state
        self._pending_buffer = None
        self._pending_save_for_backward = None
        self._pending_qlen = None

        # Return output: if output_device specified, transfer directly
        if output_device is not None:
            return buffer.output_cpu.to(device=output_device, non_blocking=True)
        else:
            return buffer.output_cpu.clone()
