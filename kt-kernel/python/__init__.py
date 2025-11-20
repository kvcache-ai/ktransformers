# KT-Kernel: High-performance kernel operations for KTransformers
# SPDX-License-Identifier: Apache-2.0

"""
KT-Kernel provides high-performance kernel operations for KTransformers,
including CPU-optimized MoE inference with AMX, AVX, and KML support.

Example usage:
    >>> from kt_kernel import KTMoEWrapper
    >>> wrapper = KTMoEWrapper(
    ...     layer_idx=0,
    ...     num_experts=8,
    ...     num_experts_per_tok=2,
    ...     hidden_size=4096,
    ...     moe_intermediate_size=14336,
    ...     num_gpu_experts=2,
    ...     cpuinfer_threads=32,
    ...     threadpool_count=2,
    ...     weight_path="/path/to/weights",
    ...     chunked_prefill_size=512,
    ...     method="AMXINT4"
    ... )
"""

from __future__ import annotations

from .experts import KTMoEWrapper

__version__ = "0.1.0"
__all__ = ["KTMoEWrapper"]
