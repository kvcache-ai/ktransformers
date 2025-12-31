# Wrapper for MoE CPU inference operations
# This module encapsulates CPU inference engine, weight loading, and buffer management
# SPDX-License-Identifier: Apache-2.0

"""
Expert wrappers for CPU-based MoE inference.

This module provides the main factory interface (KTMoEWrapper) that automatically
selects the appropriate backend implementation based on the method parameter.
"""

from __future__ import annotations

import torch
from typing import List, Optional

# Import base infrastructure
from .experts_base import BaseMoEWrapper, KExpertsCPUBuffer

# Import backend implementations
from .utils.amx import AMXMoEWrapper, NativeMoEWrapper
from .utils.llamafile import LlamafileMoEWrapper
from .utils.moe_kernel import GeneralMoEWrapper


class KTMoEWrapper:
    """
    Factory interface for MoE CPU inference operations.

    This class serves as the main entry point for external code. It automatically
    selects the appropriate backend implementation based on the `method` parameter.

    Usage:
        # Create a mask where experts 0, 2, 5 are on GPU
        gpu_mask = torch.zeros(8, dtype=torch.bool)
        gpu_mask[[0, 2, 5]] = True

        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=4096,
            moe_intermediate_size=14336,
            gpu_experts_mask=gpu_mask,  # or None for all experts on CPU
            cpuinfer_threads=32,
            threadpool_count=2,
            weight_path="/path/to/weights",
            chunked_prefill_size=512,
            method="AMXINT4"  # or "AMXINT8", "LLAMAFILE"
        )
    """

    def __new__(
        cls,
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
    ):
        """
        Factory method to create the appropriate backend implementation.

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
            max_deferred_experts_per_token: Number of experts per token to defer. Defaults to 0.
            method: Backend method ("AMXINT4", "AMXINT8", "RAWINT4", "FP8", "LLAMAFILE", "MOE_INT4", "MOE_INT8")

        Returns:
            An instance of the appropriate backend implementation (e.g., AMXMoEWrapper)
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
            raise NotImplementedError(f"Unsupported method: {method}")

        # Create and return backend instance
        return backend_cls(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
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
