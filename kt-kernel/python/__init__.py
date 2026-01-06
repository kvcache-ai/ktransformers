# KT-Kernel: High-performance kernel operations for KTransformers
# SPDX-License-Identifier: Apache-2.0

"""
KT-Kernel provides high-performance kernel operations for KTransformers,
including CPU-optimized MoE inference with AMX, AVX, and KML support.

The package automatically detects your CPU capabilities and loads the optimal
kernel variant (AMX, AVX512, or AVX2) at runtime.

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

    Check which CPU variant is loaded:
    >>> import kt_kernel
    >>> print(kt_kernel.__cpu_variant__)  # 'amx', 'avx512', or 'avx2'

Environment Variables:
    KT_KERNEL_CPU_VARIANT: Override automatic detection ('amx', 'avx512', 'avx2')
    KT_KERNEL_DEBUG: Enable debug output ('1' to enable)
"""

from __future__ import annotations

# Detect CPU and load optimal extension variant
_kt_kernel_ext = None
__cpu_variant__ = None
_extension_load_error = None

try:
    from ._cpu_detect import initialize as _initialize_cpu

    _kt_kernel_ext, __cpu_variant__ = _initialize_cpu()

    # Make the extension module available to other modules in this package
    import sys

    sys.modules["kt_kernel_ext"] = _kt_kernel_ext

    # Also expose kt_kernel_ext as an attribute for backward compatibility
    kt_kernel_ext = _kt_kernel_ext
except Exception as e:
    # Extension loading failed - this is expected in some environments
    # (e.g., dev environments with version mismatches, or CLI-only usage)
    # Store the error for diagnostic purposes
    _extension_load_error = e
    kt_kernel_ext = None

# Import main API only if extension loaded successfully
if _kt_kernel_ext is not None:
    try:
        from .experts import KTMoEWrapper
    except Exception:
        # If experts import fails, define a placeholder that will error helpfully
        class KTMoEWrapper:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    f"KTMoEWrapper unavailable: extension failed to load.\n"
                    f"Error: {_extension_load_error}\n"
                    f"The CLI tools are still available. For inference, please reinstall kt-kernel."
                )

else:
    # Extension not loaded - define placeholder
    class KTMoEWrapper:
        def __init__(self, *args, **kwargs):
            if _extension_load_error:
                raise RuntimeError(
                    f"KTMoEWrapper unavailable: extension failed to load.\n"
                    f"Error: {_extension_load_error}\n"
                    f"The CLI tools are still available. For inference, please reinstall kt-kernel."
                )
            else:
                raise RuntimeError("KTMoEWrapper unavailable: extension not loaded.")


# Read version from package metadata (preferred) or fallback to project root
try:
    # Try to get version from installed package metadata (works in installed environment)
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("kt-kernel")
    except PackageNotFoundError:
        # Package not installed, try to read from source tree version.py
        import os

        _root_version_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "version.py")
        if os.path.exists(_root_version_file):
            _version_ns = {}
            with open(_root_version_file, "r", encoding="utf-8") as f:
                exec(f.read(), _version_ns)
            __version__ = _version_ns.get("__version__", "0.4.3")
        else:
            __version__ = "0.4.3"
except ImportError:
    # Python < 3.8, fallback to pkg_resources or hardcoded version
    try:
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            __version__ = get_distribution("kt-kernel").version
        except DistributionNotFound:
            __version__ = "0.4.3"
    except ImportError:
        __version__ = "0.4.3"

__all__ = ["KTMoEWrapper", "kt_kernel_ext", "__cpu_variant__", "__version__"]
