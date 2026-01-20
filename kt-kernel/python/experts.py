# Wrapper for MoE CPU inference operations
# This module encapsulates CPU inference engine, weight loading, and buffer management
# SPDX-License-Identifier: Apache-2.0

"""
Expert wrappers for CPU-based MoE operations (inference and SFT).

This module provides the main factory interface (KTMoEWrapper) that automatically
selects the appropriate backend implementation based on the method and mode parameters.

Usage:
    # Inference mode (default)
    wrapper = KTMoEWrapper(..., mode="inference", method="AMXINT4")

    # SFT mode
    wrapper = KTMoEWrapper(..., mode="sft", method="AMXBF16_SFT", lora_rank=16)
"""

from __future__ import annotations

from typing import List, Optional, Union

# Import base infrastructure for inference
from .experts_base import BaseMoEWrapper, KExpertsCPUBuffer

# Import base infrastructure for SFT
from .experts_sft import BaseSFTMoEWrapper, KExpertsSFTBuffer

# Import inference backend implementations
from .utils.amx import AMXMoEWrapper, NativeMoEWrapper
from .utils.llamafile import LlamafileMoEWrapper
from .utils.moe_kernel import GeneralMoEWrapper

# Import SFT backend implementations
from .utils.amx_sft import AMXSFTMoEWrapper


# Valid methods for each mode
INFERENCE_METHODS = frozenset(
    [
        "AMXINT4",
        "AMXINT8",  # AMX quantization
        "RAWINT4",
        "FP8",  # Native quantization
        "LLAMAFILE",  # GGUF format
        "MOE_INT4",
        "MOE_INT8",  # General kernel
    ]
)

SFT_METHODS = frozenset(
    [
        "AMXBF16_SFT",  # AMX BF16 training
        "AMXINT8_SFT",  # AMX INT8 training
        "AMXINT4_SFT",  # AMX INT4 training
        "AMXINT4_1_SFT",  # AMX INT4_1 training
        "AMXINT4_KGroup_SFT",  # AMX INT4 K-Group training
        "AMXINT4_1KGroup_SFT",  # AMX INT4_1 K-Group training
        # SkipLoRA variants (skip all LoRA computation in backward, only compute base weight grad_input)
        "AMXBF16_SFT_SkipLoRA",
        "AMXINT8_SFT_SkipLoRA",
        "AMXINT4_SFT_SkipLoRA",
        "AMXINT4_1_SFT_SkipLoRA",
        "AMXINT4_KGroup_SFT_SkipLoRA",
        "AMXINT4_1KGroup_SFT_SkipLoRA",
    ]
)


class KTMoEWrapper:
    """
    Factory interface for MoE CPU operations (inference and SFT).

    This class serves as the main entry point for external code. It automatically
    selects the appropriate backend implementation based on the `mode` and `method` parameters.

    Supported modes:
        - "inference": Optimized for low-latency inference
        - "sft": Supervised fine-tuning with LoRA adapters

    Usage (Inference):
        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=256,
            num_experts_per_tok=8,
            hidden_size=7168,
            moe_intermediate_size=2048,
            num_gpu_experts=0,
            cpuinfer_threads=60,
            threadpool_count=4,
            weight_path="/path/to/weights",
            chunked_prefill_size=25600,
            method="AMXINT4",  # or "AMXINT8", "LLAMAFILE"
            mode="inference",  # default
        )

    Usage (SFT):
        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=256,
            num_experts_per_tok=8,
            hidden_size=7168,
            moe_intermediate_size=2048,
            num_gpu_experts=0,
            cpuinfer_threads=60,
            threadpool_count=4,
            weight_path="/path/to/weights",
            chunked_prefill_size=25600,
            method="AMXBF16_SFT",  # or "AMXINT8_SFT", "AMXINT4_SFT"
            mode="sft",
            lora_rank=16,
            lora_alpha=32.0,
        )
    """

    def __new__(
        cls,
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
        # Inference-specific parameters
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        # Mode and method selection
        method: str = "AMXINT4",
        mode: str = "inference",
        # SFT-specific parameters (only used when mode="sft")
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        max_cache_depth: int = 1,
        # Quantization config (for K-Group SFT methods)
        group_size: int = 128,
        zero_point: bool = True,
    ) -> Union[BaseMoEWrapper, BaseSFTMoEWrapper]:
        """
        Factory method to create the appropriate backend implementation.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            num_gpu_experts: Number of experts to run on GPU (usually 0 for SFT)
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory (inference only)
            max_deferred_experts_per_token: Experts per token to defer (inference only)
            method: Backend method (see INFERENCE_METHODS and SFT_METHODS)
            mode: Operation mode ("inference" or "sft")
            lora_rank: LoRA rank (SFT only)
            lora_alpha: LoRA scaling factor (SFT only)
            max_cache_depth: Maximum forward cache depth (SFT only)
            group_size: Quantization group size (SFT K-Group methods only)
            zero_point: Use zero point quantization (SFT K-Group methods only)

        Returns:
            BaseMoEWrapper for inference mode, BaseSFTMoEWrapper for SFT mode

        Raises:
            ValueError: If mode is invalid or method doesn't match mode
        """
        # Validate mode
        if mode not in ("inference", "sft"):
            raise ValueError(f"Unknown mode: '{mode}'. Supported modes: 'inference', 'sft'")

        # Validate method matches mode
        if mode == "inference":
            if method not in INFERENCE_METHODS:
                raise ValueError(
                    f"Method '{method}' not supported for inference mode. "
                    f"Supported methods: {sorted(INFERENCE_METHODS)}"
                )
        else:  # mode == "sft"
            if method not in SFT_METHODS:
                raise ValueError(
                    f"Method '{method}' not supported for SFT mode. " f"Supported methods: {sorted(SFT_METHODS)}"
                )

        # Create appropriate backend
        if mode == "inference":
            return _create_inference_wrapper(
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
                cpu_save=cpu_save,
                max_deferred_experts_per_token=max_deferred_experts_per_token,
                method=method,
            )
        else:  # mode == "sft"
            return _create_sft_wrapper(
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
                method=method,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                max_cache_depth=max_cache_depth,
                group_size=group_size,
                zero_point=zero_point,
            )

    # Forward static methods to the base class
    @staticmethod
    def set_capture_batch_sizes(capture_bs: List[int]):
        """
        Set batch sizes to capture and cache buffers for.

        This allows pre-allocation of CPU buffers for specific batch sizes,
        improving performance by avoiding buffer re-allocation during inference.

        Args:
            capture_bs: List of batch sizes to capture (e.g., [1, 2, 4, 8, 16])
        """
        BaseMoEWrapper.set_capture_batch_sizes(capture_bs)

    @staticmethod
    def get_capture_batch_sizes() -> List[int]:
        """
        Get currently configured capture batch sizes.

        Returns:
            List of batch sizes that are being captured
        """
        return BaseMoEWrapper.get_capture_batch_sizes()

    @staticmethod
    def clear_buffer_cache():
        """
        Clear all cached buffers.

        This frees up memory by clearing the buffer cache. Useful when you want
        to reset the buffer state or free memory.
        """
        BaseMoEWrapper.clear_buffer_cache()

    @staticmethod
    def clear_sft_buffer_cache():
        """
        Clear all cached SFT buffers.

        This frees up memory by clearing the SFT buffer cache. Useful when you want
        to reset the buffer state or free memory during SFT.
        """
        KExpertsSFTBuffer.clear_cache()


# =============================================================================
# Private helper functions for creating wrapper instances
# =============================================================================


def _create_inference_wrapper(
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
    cpu_save: bool,
    max_deferred_experts_per_token: Optional[int],
    method: str,
) -> BaseMoEWrapper:
    """
    Create an inference wrapper based on the method.

    Args:
        See KTMoEWrapper.__new__ for parameter descriptions.

    Returns:
        BaseMoEWrapper instance
    """
    # Select backend based on method
    if method in ["AMXINT4", "AMXINT8"]:
        backend_cls = AMXMoEWrapper
    elif method in ["RAWINT4", "FP8"]:
        backend_cls = NativeMoEWrapper
    elif method == "LLAMAFILE":
        backend_cls = LlamafileMoEWrapper
    elif method in ["MOE_INT4", "MOE_INT8"]:
        backend_cls = GeneralMoEWrapper
    else:
        # This shouldn't happen due to validation in __new__
        raise NotImplementedError(f"Unsupported inference method: {method}")

    # Create and return backend instance
    return backend_cls(
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
        cpu_save=cpu_save,
        max_deferred_experts_per_token=max_deferred_experts_per_token,
        method=method,
    )


def _create_sft_wrapper(
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
    method: str,
    lora_rank: int,
    lora_alpha: float,
    max_cache_depth: int,
    group_size: int,
    zero_point: bool,
) -> BaseSFTMoEWrapper:
    """
    Create an SFT wrapper based on the method.

    Args:
        See KTMoEWrapper.__new__ for parameter descriptions.

    Returns:
        BaseSFTMoEWrapper instance
    """
    # Currently only AMX SFT methods are supported
    # All SFT methods use AMXSFTMoEWrapper with different quantization
    return AMXSFTMoEWrapper(
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
        method=method,
        group_size=group_size,
        zero_point=zero_point,
    )
