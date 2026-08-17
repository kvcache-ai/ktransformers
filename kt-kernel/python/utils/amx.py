import gc
import glob
import logging
import os
import time
import torch
import ctypes
from typing import List, Optional

logger = logging.getLogger(__name__)


def _rss_gb() -> float:
    """Current process RSS in GiB (reads /proc/self/statm)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    except (OSError, ValueError, IndexError):
        return -1.0


def _is_gguf_path(path: str) -> bool:
    """True when path is a .gguf file or a directory containing .gguf files."""
    if os.path.isfile(path):
        return path.endswith(".gguf")
    if os.path.isdir(path):
        try:
            return any(f.endswith(".gguf") for f in os.listdir(path))
        except OSError:
            return False
    return False


# Use relative imports for package structure
from ..experts_base import BaseMoEWrapper
from .loader import (
    SafeTensorLoader,
    CompressedSafeTensorLoader,
    FP8SafeTensorLoader,
    BF16SafeTensorLoader,
    GPTQSafeTensorLoader,
    MXFP4SafeTensorLoader,
    MXFP8SafeTensorLoader,
    GGUFLoader,
)
from .gguf_cache import GGUFCacheManager
from kt_kernel_ext.moe import MOEConfig
import kt_kernel_ext
import kt_kernel_ext.moe as _moe_mod

AMXInt4_MOE = getattr(_moe_mod, "AMXInt4_MOE", None)
AMXInt8_MOE = getattr(_moe_mod, "AMXInt8_MOE", None)
AMXInt4_KGroup_MOE = getattr(_moe_mod, "AMXInt4_KGroup_MOE", None)
AMXInt4_KGroupBlocked_MOE = getattr(_moe_mod, "AMXInt4_KGroupBlocked_MOE", None)
AMXFP4_KGroup_MOE = getattr(_moe_mod, "AMXFP4_KGroup_MOE", None)
AMXMXFP8_KGroup_MOE = getattr(_moe_mod, "AMXMXFP8_KGroup_MOE", None)
AMXFP8_MOE = getattr(_moe_mod, "AMXFP8_MOE", None)
AMXBF16_MOE = getattr(_moe_mod, "AMXBF16_MOE", None)
AMXFP8PerChannel_MOE = getattr(_moe_mod, "AMXFP8PerChannel_MOE", None)
AVX2BF16_MOE = getattr(_moe_mod, "AVX2BF16_MOE", None)
AVX2FP8_MOE = getattr(_moe_mod, "AVX2FP8_MOE", None)
AVX2GPTQInt4_MOE = getattr(_moe_mod, "AVX2GPTQInt4_MOE", None)
AVX2RawInt4_MOE = getattr(_moe_mod, "AVX2RawInt4_MOE", None)
AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)
AVX2MXFP8_MOE = getattr(_moe_mod, "AVX2MXFP8_MOE", None)
AVXVNNI256GPTQInt4_MOE = getattr(_moe_mod, "AVXVNNI256GPTQInt4_MOE", None)
AVXVNNI256RawInt4_MOE = getattr(_moe_mod, "AVXVNNI256RawInt4_MOE", None)
SYCLGPTQInt4_MOE = getattr(_moe_mod, "SYCLGPTQInt4_MOE", None)

_HAS_AMXINT4_SUPPORT = AMXInt4_MOE is not None
_HAS_AMXINT8_SUPPORT = AMXInt8_MOE is not None
_HAS_AMXINT4_KGROUP_SUPPORT = AMXInt4_KGroup_MOE is not None
# Task-specific fused two-stage MOEs (stage pair -> per-attribute mix)
_FUSED_4x8_MOE = getattr(_moe_mod, "AMXFused4x8_MOE", None)  # int4 x int8 (either orientation)
_FUSED_8x16_MOE = getattr(_moe_mod, "AMXFused8x16_MOE", None)  # int8 x bf16 (either orientation)
_FUSED_4x16_MOE = getattr(_moe_mod, "AMXFused4x16_MOE", None)  # int4 x bf16 (either orientation)
# K2/RAWINT4 K-group quant: number of weight columns sharing one int8 scale
# along k. 32 ≈ Q4_K's sub-block granularity; 64/128 trade accuracy for size.
_AMXINT4_KGROUP_SIZE = int(os.environ.get("KT_AMXINT4_KGROUP_SIZE", "32"))
_HAS_RAWINT4_SUPPORT = AMXInt4_KGroup_MOE is not None
_HAS_RAWINT4_BLOCKED_SUPPORT = AMXInt4_KGroupBlocked_MOE is not None
_HAS_MXFP4_SUPPORT = AMXFP4_KGroup_MOE is not None
_HAS_MXFP8_SUPPORT = AMXMXFP8_KGroup_MOE is not None
_HAS_FP8_SUPPORT = AMXFP8_MOE is not None
_HAS_BF16_SUPPORT = AMXBF16_MOE is not None
_HAS_FP8_PERCHANNEL_SUPPORT = AMXFP8PerChannel_MOE is not None
_HAS_AVX2_BF16_SUPPORT = AVX2BF16_MOE is not None
_HAS_AVX2_FP8_SUPPORT = AVX2FP8_MOE is not None
_HAS_AVX2_GPTQ_INT4_SUPPORT = AVX2GPTQInt4_MOE is not None
_HAS_AVX2_RAWINT4_SUPPORT = AVX2RawInt4_MOE is not None
_HAS_AVX2_MXFP4_SUPPORT = AVX2MXFP4_MOE is not None
_HAS_AVX2_MXFP8_SUPPORT = AVX2MXFP8_MOE is not None
_HAS_AVXVNNI256_GPTQ_INT4_SUPPORT = AVXVNNI256GPTQInt4_MOE is not None
_HAS_AVXVNNI256_RAW_INT4_SUPPORT = AVXVNNI256RawInt4_MOE is not None
_HAS_SYCL_GPTQ_INT4_SUPPORT = SYCLGPTQInt4_MOE is not None
_AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE = 256
_AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE = 256


def _host_has_cpu_flag(*flag_names: str) -> bool:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].strip().split())
                    return any(name in flags for name in flag_names)
    except OSError:
        return False
    return False


_HOST_HAS_AVX_VNNI = _host_has_cpu_flag("avx_vnni", "avxvnni")


def _preflight_sycl_device() -> None:
    """Report the common Linux render-node permission error before C++ initialization."""
    if os.getenv("ONEAPI_DEVICE_SELECTOR", "").strip():
        return

    render_nodes = sorted(glob.glob("/dev/dri/renderD*"))
    if render_nodes and not any(os.access(path, os.R_OK | os.W_OK) for path in render_nodes):
        raise RuntimeError(
            "SYCL_GPTQ_INT4 selects a GPU by default, but the current user cannot access "
            "/dev/dri/renderD*. Add the user to the render group and re-login, or set "
            "ONEAPI_DEVICE_SELECTOR to an accessible SYCL GPU."
        )


def _supports_avxvnni256_gptq_int4_group_size(group_size: Optional[int]) -> bool:
    if group_size is None:
        return True
    return group_size > 0 and group_size % 32 == 0 and group_size <= _AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE


def _supports_avxvnni256_rawint4_group_size(group_size: Optional[int]) -> bool:
    if group_size is None:
        return True
    return group_size > 0 and group_size % 32 == 0 and group_size <= _AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE


def _select_gptq_int4_backend(group_size: Optional[int] = None):
    forced = os.getenv("KT_GPTQ_INT4_BACKEND", "").strip().lower()
    avxvnni_group_supported = _supports_avxvnni256_gptq_int4_group_size(group_size)

    if forced in {"avxvnni", "avxvnni256"}:
        if not _HAS_AVXVNNI256_GPTQ_INT4_SUPPORT:
            raise RuntimeError("KT_GPTQ_INT4_BACKEND=avxvnni requested, but AVXVNNI256GPTQInt4_MOE is not compiled in.")
        if not _HOST_HAS_AVX_VNNI:
            raise RuntimeError("KT_GPTQ_INT4_BACKEND=avxvnni requested, but the current CPU does not support avx_vnni.")
        if not avxvnni_group_supported:
            raise RuntimeError(
                "KT_GPTQ_INT4_BACKEND=avxvnni requested, but "
                f"group_size={group_size} is unsupported. AVX-VNNI-256 GPTQ_INT4 only supports "
                f"positive multiples of 32 up to {_AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE}."
            )
        return AVXVNNI256GPTQInt4_MOE

    if forced == "avx2":
        if not _HAS_AVX2_GPTQ_INT4_SUPPORT:
            raise RuntimeError("KT_GPTQ_INT4_BACKEND=avx2 requested, but AVX2GPTQInt4_MOE is not compiled in.")
        return AVX2GPTQInt4_MOE

    if _HAS_AVXVNNI256_GPTQ_INT4_SUPPORT and _HOST_HAS_AVX_VNNI and avxvnni_group_supported:
        return AVXVNNI256GPTQInt4_MOE
    if _HAS_AVX2_GPTQ_INT4_SUPPORT:
        return AVX2GPTQInt4_MOE
    return None


def _select_rawint4_backend(group_size: Optional[int] = None):
    forced = os.getenv("KT_RAWINT4_BACKEND", "").strip().lower()
    avxvnni_group_supported = _supports_avxvnni256_rawint4_group_size(group_size)

    if forced == "amx":
        if not _HAS_RAWINT4_SUPPORT:
            raise RuntimeError("KT_RAWINT4_BACKEND=amx requested, but AMXInt4_KGroup_MOE is not compiled in.")
        return AMXInt4_KGroup_MOE

    if forced in {"amx_blocked", "blocked"}:
        if not _HAS_RAWINT4_BLOCKED_SUPPORT:
            raise RuntimeError(
                "KT_RAWINT4_BACKEND=amx_blocked requested, but AMXInt4_KGroupBlocked_MOE is not compiled in."
            )
        return AMXInt4_KGroupBlocked_MOE

    if forced in {"avxvnni", "avxvnni256"}:
        if not _HAS_AVXVNNI256_RAW_INT4_SUPPORT:
            raise RuntimeError("KT_RAWINT4_BACKEND=avxvnni requested, but AVXVNNI256RawInt4_MOE is not compiled in.")
        if not _HOST_HAS_AVX_VNNI:
            raise RuntimeError("KT_RAWINT4_BACKEND=avxvnni requested, but the current CPU does not support avx_vnni.")
        if not avxvnni_group_supported:
            raise RuntimeError(
                "KT_RAWINT4_BACKEND=avxvnni requested, but "
                f"group_size={group_size} is unsupported. AVX-VNNI-256 RAWINT4 only supports "
                f"positive multiples of 32 up to {_AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE}."
            )
        return AVXVNNI256RawInt4_MOE

    if forced == "avx2":
        if not _HAS_AVX2_RAWINT4_SUPPORT:
            raise RuntimeError("KT_RAWINT4_BACKEND=avx2 requested, but AVX2RawInt4_MOE is not compiled in.")
        return AVX2RawInt4_MOE

    if _HAS_RAWINT4_SUPPORT:
        return AMXInt4_KGroup_MOE
    if _HAS_AVXVNNI256_RAW_INT4_SUPPORT and _HOST_HAS_AVX_VNNI and avxvnni_group_supported:
        return AVXVNNI256RawInt4_MOE
    if _HAS_AVX2_RAWINT4_SUPPORT:
        return AVX2RawInt4_MOE
    return None


def _select_mxfp4_backend():
    """Select MXFP4 backend: AMX/AVX-512 (preferred) > AVX2 (fallback).

    Override with KT_MXFP4_BACKEND=avx2|amx.
    Returns None if no MXFP4 backend is available.
    """
    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()

    if forced == "amx":
        if not _HAS_MXFP4_SUPPORT:
            raise RuntimeError(
                "KT_MXFP4_BACKEND=amx requested, but AMXFP4_KGroup_MOE is not compiled in. "
                "Recompile with AVX512F + AVX512BW + AVX512_BF16 enabled."
            )
        return AMXFP4_KGroup_MOE

    if forced == "avx2":
        if not _HAS_AVX2_MXFP4_SUPPORT:
            raise RuntimeError(
                "KT_MXFP4_BACKEND=avx2 requested, but AVX2MXFP4_MOE is not compiled in. "
                "Recompile with AVX2 + FMA enabled."
            )
        return AVX2MXFP4_MOE

    if _HAS_MXFP4_SUPPORT:
        return AMXFP4_KGroup_MOE
    if _HAS_AVX2_MXFP4_SUPPORT:
        return AVX2MXFP4_MOE
    return None


def _select_mxfp8_backend():
    """Select MXFP8 backend: AMX/AVX-512 (preferred) > AVX2 (fallback).

    Override with KT_MXFP8_BACKEND=avx2|amx.
    Returns None if no MXFP8 backend is available.
    """
    forced = os.getenv("KT_MXFP8_BACKEND", "").strip().lower()

    if forced == "amx":
        if not _HAS_MXFP8_SUPPORT:
            raise RuntimeError(
                "KT_MXFP8_BACKEND=amx requested, but AMXMXFP8_KGroup_MOE is not compiled in. "
                "Recompile with AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI enabled."
            )
        if not _host_has_cpu_flag("amx_tile", "amx_bf16"):
            raise RuntimeError(
                "KT_MXFP8_BACKEND=amx requested, but the host CPU lacks AMX (amx_tile / amx_bf16). "
                "This would SIGILL at first forward. Unset the env to fall back to AVX2."
            )
        return AMXMXFP8_KGroup_MOE

    if forced == "avx2":
        if not _HAS_AVX2_MXFP8_SUPPORT:
            raise RuntimeError(
                "KT_MXFP8_BACKEND=avx2 requested, but AVX2MXFP8_MOE is not compiled in. "
                "Recompile with AVX2 + FMA enabled."
            )
        return AVX2MXFP8_MOE

    # Auto-select: prefer AMX iff the .so was built with it AND the runtime CPU has AMX.
    # Compile-time-only check would SIGILL on AVX-512 CPUs lacking AMX (pre-Sapphire Rapids).
    if _HAS_MXFP8_SUPPORT and _host_has_cpu_flag("amx_tile", "amx_bf16"):
        return AMXMXFP8_KGroup_MOE
    if _HAS_AVX2_MXFP8_SUPPORT:
        return AVX2MXFP8_MOE
    return None


class AMXMoEWrapper(BaseMoEWrapper):
    """
    AMX-based MoE wrapper implementation.
    Supports AMXINT4 and AMXINT8 quantization methods.
    """

    _safetensor_loader_instance = None  # Singleton SafeTensorLoader
    _gguf_loader_instance = None  # Singleton GGUFLoader (keeps the mmap alive)
    _ggml_inited = False
    _smart_down_nodes = {}  # layer_idx -> down node, for the (prev,cur) edge log

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
    ):
        """
        Initialize AMX MoE Wrapper.

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
            weight_path: Path to AMX weights (SafeTensor format)
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer. Defaults to 0.
            method: AMX quantization method ("AMXINT4" or "AMXINT8")
        """
        if method == "AMXINT4" and not _HAS_AMXINT4_SUPPORT:
            raise RuntimeError(
                "AMXINT4 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (VNNI optional)\n"
                "Please recompile kt_kernel_ext with AVX512 enabled."
            )
        if method == "AMXINT8" and not _HAS_AMXINT8_SUPPORT:
            raise RuntimeError(
                "AMXINT8 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (VNNI optional)\n"
                "Please recompile kt_kernel_ext with AVX512 enabled."
            )
        if method in ("AMXINT4_KGROUP", "AMXINT4_SMART") and not _HAS_AMXINT4_KGROUP_SUPPORT:
            raise RuntimeError(
                f"{method} backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (VNNI optional)\n"
                "Please recompile kt_kernel_ext with AVX512 enabled."
            )

        # Initialize base class
        super().__init__(
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
            numa_nodes=numa_nodes,
        )

        # AMX-specific: Check if we should load merged safetensor weights
        self.load_merged_weight = False
        # GGUF source: when weight_path is a .gguf file/dir, the C++ side
        # dequantizes 64-row strips straight from the mmap'd GGUF blocks
        # ("online quant from gguf") and the first boot writes the INT8 cache
        # (see gguf_cache.py). No BF16/FP32 materialization in Python.
        self.gguf_loader = None
        self.gguf_cache = None
        self.is_gguf = _is_gguf_path(weight_path)
        self._smart_errs = None
        self._smart_pair = None

        if self.is_gguf:
            # ggml_init() fills ggml_table_f32_f16 — required by conversion.h's
            # to_float fallback for exotic GGML types (IQ*, ...). Idempotent.
            if not AMXMoEWrapper._ggml_inited:
                kt_kernel_ext.utils.ggml_init()
                AMXMoEWrapper._ggml_inited = True
            if AMXMoEWrapper._gguf_loader_instance is None:
                AMXMoEWrapper._gguf_loader_instance = GGUFLoader(weight_path)
            self.gguf_loader = AMXMoEWrapper._gguf_loader_instance
            if method in ("AMXINT4_KGROUP",):
                # No disk cache: K2 KGroup quantizes every boot.
                self.gguf_cache = None
                logger.info(
                    "[AMXMoEWrapper] Layer %d: GGUF source %s (%s, no cache)",
                    layer_idx,
                    weight_path,
                    method,
                )
            else:
                self.gguf_cache = GGUFCacheManager(
                    weight_path,
                    self.gguf_loader,
                    method=method,
                    threadpool_count=threadpool_count,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_intermediate_size,
                    expert_num=num_experts,
                )
            logger.info(
                "[AMXMoEWrapper] Layer %d: GGUF source %s (INT8 cache: %s)",
                layer_idx,
                weight_path,
                self.gguf_cache.cache_dir if self.gguf_cache is not None else "disabled",
            )
        elif glob.glob(os.path.join(weight_path, "*.safetensors")):
            self.load_merged_weight = True

        # Initialize SafeTensor loader (singleton)
        if self.load_merged_weight:
            if AMXMoEWrapper._safetensor_loader_instance is None:
                AMXMoEWrapper._safetensor_loader_instance = SafeTensorLoader(weight_path)
            self.safetensor_loader = AMXMoEWrapper._safetensor_loader_instance

        # AMX-specific weight storage
        self.gate_weights = None
        self.up_weights = None
        self.down_weights = None
        self.gate_scales = None
        self.up_scales = None
        self.down_scales = None

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
        # Free old C++ MoE object BEFORE creating a new one.
        # ~AMX_MOE_BASE() now frees the aligned_alloc'd BufferB weights
        # (~3.6GB/layer for MiniMax 1536x3072 INT8 × 256 experts) in its
        # destructor (owned_aligned_allocs_), so destroying the previous
        # layer's object actually returns that memory — without this the
        # per-layer quantize loop accumulates ~3.6GB/layer and OOMs
        # around layer 60 of a 61-layer model.
        if self.moe is not None:
            del self.moe
            self.moe = None
            import gc

            gc.collect()

        # Store tensors as instance variables to keep them alive
        self.gate_proj = gate_proj.contiguous()
        self.up_proj = up_proj.contiguous()
        self.down_proj = down_proj.contiguous()

        # Configure MoE with online quantization (cpu_save mode)
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size

        # Enable save mode for online quantization
        moe_config.save = True
        moe_config.load = False

        # Set weight pointers
        moe_config.gate_proj = self.gate_proj.data_ptr()
        moe_config.up_proj = self.up_proj.data_ptr()
        moe_config.down_proj = self.down_proj.data_ptr()

        # Set output path for quantized weights
        moe_config.path = self.weight_path

        # Create MoE module based on AMX method
        if self.method == "AMXINT4":
            self.moe = AMXInt4_MOE(moe_config)
        elif self.method == "AMXINT8":
            self.moe = AMXInt8_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported AMX method: {self.method}")

        # Submit quantization and save task
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()

    def _make_moe_config(self) -> MOEConfig:
        """Build a MOEConfig with the common fields for this layer."""
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size
        return moe_config

    def _create_and_load_moe(self, moe_config: MOEConfig, physical_to_logical_map_cpu: torch.Tensor) -> None:
        """Create the AMX MoE module and run its load_weights task to completion."""
        # Free the previous layer's C++ MoE object first; ~AMX_MOE_BASE() frees
        # the aligned_alloc'd BufferB weights, so the per-layer quantize loop
        # does not accumulate ~3.4 GB/layer.
        if self.moe is not None:
            del self.moe
            self.moe = None
            import gc

            gc.collect()

        if self.method == "AMXINT4":
            self.moe = AMXInt4_MOE(moe_config)
        elif self.method == "AMXINT8":
            self.moe = AMXInt8_MOE(moe_config)
        elif self.method == "AMXINT4_KGROUP":
            moe_config.quant_config.group_size = _AMXINT4_KGROUP_SIZE
            self.moe = AMXInt4_KGroup_MOE(moe_config)
        elif self.method == "AMXINT4_SMART":
            # Fused-stage routing: the load-time per-attribute nodes form the
            # (upstream, downstream) pair. Uniform pairs use the plain single-
            # precision MOEs; mixed pairs are the computational wrappers
            # (F4x8 / F8x16 / F4x16) around the downstream node's default
            # kernel.
            up = getattr(moe_config, "upstream_precision", 0)
            dw = getattr(moe_config, "downstream_precision", 0)
            pair = (up, dw)
            # Mixed stage pairs are COMPUTATIONAL wrappers: the layer keeps
            # its per-attribute precisions in RAM (gate/up at the upstream
            # node, down at the downstream node — smallest possible footprint)
            # and the mixed GEMMs run through the fused two-stage kernels at
            # compute time. Only the activations are widened in the fused
            # decode, never the stored weights. Each fused wrapper serves
            # BOTH orientations of its pair: the wrapper decides at entry
            # (from the per-attribute nodes) which stage is the wider one.
            # Uniform pairs use the plain single-precision MOEs.
            fused_map = {
                (0, 1): _FUSED_4x8_MOE,
                (1, 2): _FUSED_8x16_MOE,
                (0, 2): _FUSED_4x16_MOE,
                (1, 0): _FUSED_4x8_MOE,
                (2, 1): _FUSED_8x16_MOE,
                (2, 0): _FUSED_4x16_MOE,
            }
            if pair in fused_map and fused_map[pair] is not None:
                self.moe = fused_map[pair](moe_config)
            elif pair == (0, 0):
                self.moe = AMXInt4_MOE(moe_config)
            elif pair == (1, 1):
                self.moe = AMXInt8_MOE(moe_config)
            elif pair == (2, 2):
                self.moe = AMXBF16_MOE(moe_config)
            else:
                # defensive: every (0..2)x(0..2) pair is covered above
                self.moe = {0: AMXInt4_MOE, 1: AMXInt8_MOE, 2: AMXBF16_MOE}[max(up, dw)](moe_config)
        elif self.method == "BF16":
            self.moe = AMXBF16_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported AMX method: {self.method}")
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()

    def _load_weights_gguf(self, physical_to_logical_map_cpu: torch.Tensor) -> None:
        """
        Load layer weights from a GGUF source (C++ strip dequant + INT8 cache).

        Cache hit: load the packed INT8 .kt files directly — no GGUF access,
        RAM stays at INT8 size. Cache miss (or first boot): feed the C++ side
        mmap pointers + GGML types into the existing online-quant path; the
        save step populates the cache as a side effect. No BF16/FP32 tensor is
        ever materialized in Python.
        """
        cache = self.gguf_cache
        if cache is not None and cache.enabled and cache.layer_complete(self.layer_idx):
            logger.info(
                "[AMXMoEWrapper] Layer %d: INT8 cache hit, loading from %s",
                self.layer_idx,
                cache.cache_dir,
            )
            moe_config = self._make_moe_config()
            moe_config.load = True
            moe_config.save = False
            moe_config.path = cache.cache_dir
            self._create_and_load_moe(moe_config, physical_to_logical_map_cpu)
            return

        t0 = time.time()
        E, I, H = self.num_experts, self.moe_intermediate_size, self.hidden_size
        loader = self.gguf_loader
        if loader is None:
            raise RuntimeError("GGUF loader not initialized for GGUF weight path")
        base = f"blk.{self.layer_idx}"
        gate_src = loader.get_expert_gguf_source(f"{base}.ffn_gate_exps.weight", expected_shape=[E, I, H])
        up_src = loader.get_expert_gguf_source(f"{base}.ffn_up_exps.weight", expected_shape=[E, I, H])
        down_src = loader.get_expert_gguf_source(f"{base}.ffn_down_exps.weight", expected_shape=[E, H, I])

        moe_config = self._make_moe_config()
        moe_config.gate_gguf = gate_src["ptr"]
        moe_config.gate_gguf_stride = gate_src["stride"]
        moe_config.gate_gguf_type = gate_src["ggml_type"]
        moe_config.up_gguf = up_src["ptr"]
        moe_config.up_gguf_stride = up_src["stride"]
        moe_config.up_gguf_type = up_src["ggml_type"]
        moe_config.down_gguf = down_src["ptr"]
        moe_config.down_gguf_stride = down_src["stride"]
        moe_config.down_gguf_type = down_src["ggml_type"]
        moe_config.gguf_full_intermediate_size = I

        if self.method == "AMXINT4_SMART":
            # Per-attribute dtype nodes (3 storage dtypes per layer, from the
            # GGUF types alone — no error measurement in the routing): any
            # F32/F16/BF16 -> node 2 (BF16 storage); else any tensor in
            # (Q4, Q8] (Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/IQ*/I*) -> node 1 (AMXINT8
            # storage); else (all at-or-below Q4) -> node 0 (per-row AMXINT4
            # storage). The per-row INT4 error is still measured for the log
            # only.
            errs = kt_kernel_ext.moe.per_row_int4_err(
                gate_src["ptr"],
                gate_src["stride"],
                gate_src["ggml_type"],
                up_src["ptr"],
                up_src["stride"],
                up_src["ggml_type"],
                down_src["ptr"],
                down_src["stride"],
                down_src["ggml_type"],
                I,
                H,
                128,
            )
            self._smart_errs = [float(e) if e >= 0.0 else 0.0 for e in errs]

            # 3 storage dtypes per layer (the rule is unchanged): the layer's
            # tensors decide — any F32/F16/BF16 -> BF16 storage; else any
            # tensor in (Q4, Q8] (Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/IQ*/I*) -> AMXINT8
            # storage; else (all at-or-below Q4) -> per-row AMXINT4 storage.
            def _cls(ggml_type: int) -> int:
                if ggml_type in (0, 1, 30):  # F32/F16/BF16 -> BF16 storage
                    return 2
                if ggml_type in (6, 7, 8, 13, 14) or ggml_type in (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29):
                    return 1  # (Q4, Q8] -> AMXINT8 storage
                return 0  # at-or-below Q4 -> AMXINT4 storage

            stored = max(_cls(gate_src["ggml_type"]), _cls(up_src["ggml_type"]), _cls(down_src["ggml_type"]))
            up = max(_cls(gate_src["ggml_type"]), _cls(up_src["ggml_type"]))
            dw = _cls(down_src["ggml_type"])
            moe_config.upstream_precision = up
            moe_config.downstream_precision = dw
            self._smart_pair = (up, dw)
            # the pair is an edge between two layers: (prev_down -> this_gate).
            # Stash this layer's stored class for the next layer's edge log.
            prev = AMXMoEWrapper._smart_down_nodes.get(self.layer_idx - 1)
            if prev is not None:
                logger.info(
                    "[AMXMoEWrapper] (%d, %d): stage edge (%d -> %d)",
                    self.layer_idx - 1,
                    self.layer_idx,
                    prev,
                    moe_config.upstream_precision,
                )
            AMXMoEWrapper._smart_down_nodes[self.layer_idx] = stored
            _names = {
                0: "F32",
                1: "F16",
                2: "Q4_0",
                3: "Q4_1",
                6: "Q5_0",
                7: "Q5_1",
                8: "Q8_0",
                10: "Q2_K",
                11: "Q3_K",
                12: "Q4_K",
                13: "Q5_K",
                14: "Q6_K",
                30: "BF16",
            }
            logger.info(
                "[AMXMoEWrapper] Layer %d: AMXINT4_SMART original gate=%s up=%s down=%s "
                "-> stored %s (per-row INT4 err %.4f/%.4f/%.4f)",
                self.layer_idx,
                _names.get(gate_src["ggml_type"], str(gate_src["ggml_type"])),
                _names.get(up_src["ggml_type"], str(up_src["ggml_type"])),
                _names.get(down_src["ggml_type"], str(down_src["ggml_type"])),
                ("AMXINT4" if stored == 0 else "AMXINT8" if stored == 1 else "BF16"),
                self._smart_errs[0],
                self._smart_errs[1],
                self._smart_errs[2],
            )

        if cache is not None and cache.enabled:
            # First boot / refresh: quantize and populate the cache.
            moe_config.save = True
            moe_config.load = False
            moe_config.path = cache.cache_dir
        else:
            # KT_GGUF_CACHE=0: pure online quant from GGUF every boot.
            moe_config.save = False
            moe_config.load = False
            moe_config.path = ""

        self._create_and_load_moe(moe_config, physical_to_logical_map_cpu)

        if cache is not None and cache.enabled:
            cache.mark_layer_complete(self.layer_idx)
            logger.info(
                "[AMXMoEWrapper] Layer %d: GGUF→INT8 quantize + cache done in %.1fs (%s)",
                self.layer_idx,
                time.time() - t0,
                cache.cache_dir,
            )

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        # GGUF path: C++ dequantizes 64-row strips straight from the mmap'd
        # GGUF blocks and packs them to INT8. The first boot writes the INT8
        # cache as a side effect; later boots load the cache directly — no
        # GGUF access, RAM stays at INT8 size.
        if self.is_gguf:
            self._load_weights_gguf(physical_to_logical_map_cpu)
            return

        gate_ptr = 0
        up_ptr = 0
        down_ptr = 0

        gate_ptrs = []
        up_ptrs = []
        down_ptrs = []

        gate_scale_ptrs = []
        up_scale_ptrs = []
        down_scale_ptrs = []

        if self.load_merged_weight:
            base_key = f"blk.{self.layer_idx}"
            w = self.safetensor_loader.load_experts(base_key)

            self.gate_weights = w["gate"]
            self.up_weights = w["up"]
            self.down_weights = w["down"]
            self.gate_scales = w["gate_scale"]
            self.up_scales = w["up_scale"]
            self.down_scales = w["down_scale"]

            # Get pointers to weight arrays
            gate_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.gate_weights
            ]

            up_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.up_weights
            ]

            down_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.down_weights
            ]

            gate_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.gate_scales
            ]

            up_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.up_scales
            ]

            down_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.down_scales
            ]

        # Configure MoE
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size

        moe_config.gate_proj = gate_ptr
        moe_config.up_proj = up_ptr
        moe_config.down_proj = down_ptr
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs

        if self.cpu_save:
            moe_config.save = True
            moe_config.load = False
            base_key = f"model.layers.{self.layer_idx}"
            try:
                w = self.safetensor_loader.load_experts(base_key)
            except (ValueError, KeyError):
                base_key = f"model.language_model.layers.{self.layer_idx}"
                w = self.safetensor_loader.load_experts(base_key)

            self.gate_proj = torch.cat(w["gate_weight"], dim=0).contiguous()
            self.up_proj = torch.cat(w["up_weight"], dim=0).contiguous()
            self.down_proj = torch.cat(w["down_weight"], dim=0).contiguous()

            moe_config.gate_proj = self.gate_proj.data_ptr()
            moe_config.up_proj = self.up_proj.data_ptr()
            moe_config.down_proj = self.down_proj.data_ptr()
        else:
            moe_config.load = True

        if not self.load_merged_weight:
            moe_config.path = self.weight_path

        # Create MoE module based on AMX method
        if self.method == "AMXINT4":
            self.moe = AMXInt4_MOE(moe_config)
        elif self.method == "AMXINT8":
            self.moe = AMXInt8_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported AMX method: {self.method}")

        # Load weights
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()

        # Clean up temporary weight storage if using merged weights
        if self.load_merged_weight:
            del self.gate_weights
            del self.up_weights
            del self.down_weights
            del self.gate_scales
            del self.up_scales
            del self.down_scales


class NativeMoEWrapper(BaseMoEWrapper):
    """Wrapper for native CPU/SYCL experts stored in compressed SafeTensor format."""

    _native_loader_instance = None

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
        method: str = "RAWINT4",
        numa_nodes: Optional[List[int]] = None,
        swiglu_limit: float = 0.0,
        swiglu_alpha: float = 0.0,
    ):
        self._swiglu_alpha = float(swiglu_alpha)
        # Defence in depth: reject swiglu_limit on non-MXFP4/MXFP8 methods even
        # if the experts.py guard is bypassed (e.g., by a future caller
        # that constructs NativeMoEWrapper directly). Origin: kt-sglang 耦合.
        if swiglu_limit != 0.0 and method not in ("MXFP4", "MXFP8"):
            raise ValueError(
                f"NativeMoEWrapper received swiglu_limit={swiglu_limit} with "
                f"method={method!r}; the clamp only applies to MXFP4/MXFP8. "
                f"This indicates a missing guard in the caller."
            )
        if method == "RAWINT4" and not (
            _HAS_RAWINT4_SUPPORT or _HAS_AVX2_RAWINT4_SUPPORT or _HAS_AVXVNNI256_RAW_INT4_SUPPORT
        ):
            raise RuntimeError(
                "RAWINT4 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (for AMX backend), or\n"
                "  - AVX2 + FMA (for AVX2 fallback backend)\n"
                "AVX-VNNI-256 will be selected automatically when available on the current CPU.\n"
                "Please recompile kt_kernel_ext with AVX512 or AVX2 enabled."
            )
        if method == "FP8" and not _HAS_FP8_SUPPORT and not _HAS_AVX2_FP8_SUPPORT:
            raise RuntimeError(
                "FP8 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI (for AMX), or\n"
                "  - AVX2 + FMA (for AVX2 fallback)\n"
                "Please recompile kt_kernel_ext with AVX512 + BF16 + VBMI enabled."
            )
        if method == "FP8_PERCHANNEL" and not _HAS_FP8_PERCHANNEL_SUPPORT:
            raise RuntimeError(
                "FP8_PERCHANNEL backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI\n"
                "Please recompile kt_kernel_ext with AVX512 + BF16 + VBMI enabled."
            )
        if method == "BF16" and not _HAS_BF16_SUPPORT and not _HAS_AVX2_BF16_SUPPORT:
            raise RuntimeError(
                "BF16 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 (for AMX backend), or\n"
                "  - AVX2 + FMA (for AVX2 fallback backend)\n"
                "Please recompile kt_kernel_ext with AVX512+BF16 or AVX2 enabled."
            )
        if method == "GPTQ_INT4" and not (_HAS_AVX2_GPTQ_INT4_SUPPORT or _HAS_AVXVNNI256_GPTQ_INT4_SUPPORT):
            raise RuntimeError(
                "GPTQ_INT4 backend not available.\n"
                "Please recompile kt_kernel_ext with GPTQ INT4 support enabled.\n"
                "AVX-VNNI-256 will be selected automatically when available on the current CPU."
            )
        if method == "SYCL_GPTQ_INT4" and not _HAS_SYCL_GPTQ_INT4_SUPPORT:
            raise RuntimeError(
                "SYCL_GPTQ_INT4 backend not available. Rebuild kt_kernel_ext with "
                "CPUINFER_USE_SYCL=1 using a SYCL compiler such as icpx."
            )
        if method == "MXFP4" and not (_HAS_MXFP4_SUPPORT or _HAS_AVX2_MXFP4_SUPPORT):
            raise RuntimeError(
                "MXFP4 backend not available. Required ISA (any one of):\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 (for AMX/AVX-512 backend)\n"
                "  - AVX2 + FMA (for AVX2 fallback backend)\n"
                "Please recompile kt_kernel_ext with one of the above enabled."
            )
        if method == "MXFP8" and not (_HAS_MXFP8_SUPPORT or _HAS_AVX2_MXFP8_SUPPORT):
            raise RuntimeError(
                "MXFP8 backend not available. Required ISA (any one of):\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI (for AMX/AVX-512 backend)\n"
                "  - AVX2 + FMA (for AVX2 fallback backend)\n"
                "Please recompile kt_kernel_ext with one of the above enabled."
            )

        super().__init__(
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
            numa_nodes=numa_nodes,
            swiglu_limit=swiglu_limit,
        )

        if NativeMoEWrapper._native_loader_instance is None:
            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)
        self.loader = NativeMoEWrapper._native_loader_instance

        self.gate_weights = None
        self.up_weights = None
        self.down_weights = None
        self.gate_scales = None
        self.up_scales = None
        self.down_scales = None

    @staticmethod
    def _create_loader(method: str, weight_path: str):
        if method == "RAWINT4":
            return CompressedSafeTensorLoader(weight_path)
        elif method == "FP8":
            return FP8SafeTensorLoader(weight_path)
        elif method == "FP8_PERCHANNEL":
            return FP8SafeTensorLoader(weight_path, scale_suffix="weight_scale")
        elif method == "BF16":
            return BF16SafeTensorLoader(weight_path)
        elif method in ("GPTQ_INT4", "SYCL_GPTQ_INT4"):
            return GPTQSafeTensorLoader(weight_path)
        elif method == "MXFP4":
            return MXFP4SafeTensorLoader(weight_path)
        elif method == "MXFP8":
            return MXFP8SafeTensorLoader(weight_path)
        else:
            raise NotImplementedError(f"Unsupported method for NativeMoEWrapper: {method}")

    @staticmethod
    def _release_loader(layer_idx: int = -1):
        if NativeMoEWrapper._native_loader_instance is not None:
            NativeMoEWrapper._native_loader_instance.close_all_handles()
            NativeMoEWrapper._native_loader_instance = None
            if layer_idx >= 0:
                logger.info(
                    "[KT] Released NativeMoEWrapper loader after layer %d: " "safetensors mmap handles freed.",
                    layer_idx,
                )
            else:
                logger.info("[KT] Released NativeMoEWrapper loader: safetensors mmap handles freed.")

    @staticmethod
    def force_release_loader():
        NativeMoEWrapper._release_loader()

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        raise NotImplementedError("RAWINT4 wrapper expects pre-quantized safetensor weights.")

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        import time

        if NativeMoEWrapper._native_loader_instance is None:
            t_recreate_start = time.time()
            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(self.method, self.weight_path)
            self.loader = NativeMoEWrapper._native_loader_instance
            t_recreate_elapsed = (time.time() - t_recreate_start) * 1000
            logger.info(
                "[KT] Recreated NativeMoEWrapper loader for layer %d (took %.1fms)",
                self.layer_idx,
                t_recreate_elapsed,
            )
        else:
            self.loader = NativeMoEWrapper._native_loader_instance

        t0 = time.time()
        _candidates = [
            f"model.layers.{self.layer_idx}",
            f"language_model.model.layers.{self.layer_idx}",
            f"model.language_model.layers.{self.layer_idx}",
        ]
        weights = None
        for base_key in _candidates:
            try:
                weights = self.loader.load_experts(base_key)
                break
            except (ValueError, KeyError):
                continue
        if weights is None:
            raise ValueError(f"No experts found for layer {self.layer_idx} under any prefix: {_candidates}")
        t1 = time.time()

        # Keep individual tensors instead of stacking - avoid expensive memory copy
        # weights["gate"], weights["up"], weights["down"] are lists of tensors per expert
        self.gate_weights = weights["gate"]  # list of tensors
        self.up_weights = weights["up"]
        self.down_weights = weights["down"]

        # BF16 has no scales, others have scales
        if self.method == "BF16":
            # BF16 doesn't have scales
            self.gate_scales = None
            self.up_scales = None
            self.down_scales = None
        else:
            # Convert scales to bf16 individually
            # self.gate_scales = [t.to(torch.bfloat16).contiguous() for t in weights["gate_scale"]]
            # self.up_scales = [t.to(torch.bfloat16).contiguous() for t in weights["up_scale"]]
            # self.down_scales = [t.to(torch.bfloat16).contiguous() for t in weights["down_scale"]]
            self.gate_scales = weights["gate_scale"]
            self.up_scales = weights["up_scale"]
            self.down_scales = weights["down_scale"]
            if self.method == "RAWINT4":
                assert self.gate_scales[0].dtype == torch.bfloat16, "Expected bf16 scales for RAWINT4"
            elif self.method == "FP8":
                if self.gate_scales[0].dtype != torch.float32:
                    self.gate_scales = [t.to(torch.float32).contiguous() for t in weights["gate_scale"]]
                    self.up_scales = [t.to(torch.float32).contiguous() for t in weights["up_scale"]]
                    self.down_scales = [t.to(torch.float32).contiguous() for t in weights["down_scale"]]
                assert self.gate_scales[0].dtype == torch.float32, "Expected float32 scales for FP8"
            elif self.method == "FP8_PERCHANNEL":
                if self.gate_scales[0].dtype != torch.float32:
                    self.gate_scales = [t.to(torch.float32).contiguous() for t in weights["gate_scale"]]
                    self.up_scales = [t.to(torch.float32).contiguous() for t in weights["up_scale"]]
                    self.down_scales = [t.to(torch.float32).contiguous() for t in weights["down_scale"]]
                assert self.gate_scales[0].dtype == torch.float32, "Expected float32 scales for FP8_PERCHANNEL"
            elif self.method == "MXFP4":
                # ue8m0 is losslessly representable in bf16 (8-bit exponent, 0 mantissa);
                # the loader has already done that conversion.
                assert self.gate_scales[0].dtype == torch.bfloat16, "Expected bf16 scales for MXFP4"
            elif self.method == "MXFP8":
                # ue8m0 scales stay as uint8; C++ convert_ue8m0_to_fp32 handles conversion.
                assert self.gate_scales[0].dtype == torch.uint8, "Expected uint8 (ue8m0) scales for MXFP8"

        t2 = time.time()

        # Build pointer lists: [numa_id][expert_id] -> pointer
        # Since RAWINT4/FP8/BF16 has no numa sharding, numa dimension is 1
        gate_ptrs = [[t.data_ptr() for t in self.gate_weights]]
        up_ptrs = [[t.data_ptr() for t in self.up_weights]]
        down_ptrs = [[t.data_ptr() for t in self.down_weights]]

        # BF16 has no scales, pass empty lists (will use 0/nullptr for consistency)
        if self.method == "BF16":
            gate_scale_ptrs = [[0 for _ in self.gate_weights]]
            up_scale_ptrs = [[0 for _ in self.up_weights]]
            down_scale_ptrs = [[0 for _ in self.down_weights]]
        else:
            gate_scale_ptrs = [[t.data_ptr() for t in self.gate_scales]]
            up_scale_ptrs = [[t.data_ptr() for t in self.up_scales]]
            down_scale_ptrs = [[t.data_ptr() for t in self.down_scales]]
        t3 = time.time()

        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size
        # V4-Flash 2604B SwiGLU clamp; 0.0 = disabled (default for non-MXFP4
        # paths). Read by `act_fn` in operators/amx/la/amx.hpp via
        # `apply_activation` in operators/amx/moe_base.hpp. Re-checked here
        # (defence in depth) so a future caller that bypasses both the
        # experts.py and the __init__ guards still cannot apply the clamp
        # on RAWINT4 / FP8 / BF16 / FP8_PERCHANNEL / GPTQ_INT4 paths.
        # Origin: kt-sglang 耦合.
        if self.swiglu_limit != 0.0 and self.method not in ("MXFP4", "MXFP8"):
            raise ValueError(
                f"NativeMoEWrapper.load_weights: swiglu_limit="
                f"{self.swiglu_limit} with method={self.method!r}; clamp is "
                f"only valid for MXFP4/MXFP8."
            )
        moe_config.swiglu_limit = self.swiglu_limit

        # Use gate_projs instead of gate_proj for per-expert pointers
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs

        # Infer group_size from scale shape (column-major layout)
        # For gate/up projection: in_features = hidden_size
        # So: group_size = hidden_size / scale.shape[1]

        if self.method == "RAWINT4":
            group_size = self.hidden_size // self.gate_scales[0].shape[1]
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = group_size
            moe_config.quant_config.zero_point = False
            backend_cls = _select_rawint4_backend(group_size)
            if backend_cls is None:
                raise RuntimeError(
                    "No RAWINT4 backend is available after runtime selection for "
                    f"group_size={group_size}. AMX (AMXInt4_KGroup_MOE) is preferred; "
                    f"AVX-VNNI-256 supports positive multiples of 32 up to "
                    f"{_AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE}; AVX2 (AVX2RawInt4_MOE) is used as the final fallback."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "MXFP4":
            # MXFP4: E2M1 nibble-packed weights, ue8m0/bf16 per-32 group scale
            # (e.g. DeepSeek-V4-Flash routed experts)
            group_size = self.hidden_size // self.gate_scales[0].shape[1]
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = group_size
            moe_config.quant_config.zero_point = False
            backend_cls = _select_mxfp4_backend()
            if backend_cls is None:
                raise RuntimeError(
                    "No MXFP4 backend available after runtime selection. "
                    "Compile with AVX512_BF16 (AMXFP4_KGroup_MOE) or AVX2 (AVX2MXFP4_MOE)."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "MXFP8":
            # MXFP8: FP8 E4M3fn byte weights, ue8m0/uint8 per-32 group scale
            # (e.g. MiniMax-M3-Preview)
            group_size = self.hidden_size // self.gate_scales[0].shape[1]
            moe_config.quant_config.bits = 8
            moe_config.quant_config.group_size = group_size
            moe_config.quant_config.zero_point = False
            moe_config.swiglu_alpha = getattr(self, "_swiglu_alpha", 0.0)
            backend_cls = _select_mxfp8_backend()
            if backend_cls is None:
                raise RuntimeError(
                    "No MXFP8 backend available after runtime selection. "
                    "Compile with AVX512+VBMI (AMXMXFP8_KGroup_MOE) or AVX2 (AVX2MXFP8_MOE)."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "FP8":
            moe_config.quant_config.bits = 8
            moe_config.quant_config.group_size = 128
            moe_config.quant_config.zero_point = False
            if _HAS_FP8_SUPPORT:
                self.moe = AMXFP8_MOE(moe_config)
            else:
                self.moe = AVX2FP8_MOE(moe_config)
        elif self.method == "FP8_PERCHANNEL":
            moe_config.quant_config.bits = 8
            moe_config.quant_config.per_channel = True
            moe_config.quant_config.zero_point = False
            self.moe = AMXFP8PerChannel_MOE(moe_config)
        elif self.method == "GPTQ_INT4":
            # GPTQ symmetric INT4: qweight (int32) + scales (fp32)
            group_size = self.gate_scales[0].shape[0]  # scales shape [K/gs, N], first dim = num_groups
            # hidden_size / num_groups = group_size
            actual_gs = self.hidden_size // group_size
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = actual_gs
            moe_config.quant_config.zero_point = False
            backend_cls = _select_gptq_int4_backend(actual_gs)
            if backend_cls is None:
                raise RuntimeError(
                    "No GPTQ_INT4 backend is available after runtime selection for "
                    f"group_size={actual_gs}. AVX-VNNI-256 supports positive multiples of 32 up to "
                    f"{_AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE}; AVX2 is used as the fallback when available."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "SYCL_GPTQ_INT4":
            # Same symmetric GPTQ tensor layout as GPTQ_INT4; execution is on
            # the selected SYCL device.
            num_groups = self.gate_scales[0].shape[0]
            actual_gs = self.hidden_size // num_groups
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = actual_gs
            moe_config.quant_config.zero_point = False
            _preflight_sycl_device()
            self.moe = SYCLGPTQInt4_MOE(moe_config)
        elif self.method == "BF16":
            # BF16 has no quantization config needed
            # Prefer AMX backend, fall back to AVX2
            if _HAS_BF16_SUPPORT:
                self.moe = AMXBF16_MOE(moe_config)
            else:
                self.moe = AVX2BF16_MOE(moe_config)
        t4 = time.time()

        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()
        t5 = time.time()

        del self.gate_weights
        del self.up_weights
        del self.down_weights
        if self.gate_scales is not None:
            del self.gate_scales
            del self.up_scales
            del self.down_scales

        NativeMoEWrapper._release_loader(layer_idx=self.layer_idx)
        t6 = time.time()

        print(
            f"[NativeMoEWrapper Layer {self.layer_idx}] "
            f"load_experts: {(t1-t0)*1000:.1f}ms, "
            f"prepare_tensors: {(t2-t1)*1000:.1f}ms, "
            f"build_ptrs: {(t3-t2)*1000:.1f}ms, "
            f"create_moe: {(t4-t3)*1000:.1f}ms, "
            f"cpp_load_weights: {(t5-t4)*1000:.1f}ms, "
            f"cleanup: {(t6-t5)*1000:.1f}ms, "
            f"total: {(t6-t0)*1000:.1f}ms"
        )

    def submit_write_weight_scale_to_buffer(
        self,
        gpu_tp_count: int,
        expert_id: int,
        w13_weight_ptrs,
        w13_scale_ptrs,
        w2_weight_ptrs,
        w2_scale_ptrs,
    ):
        """
        Submit the write_weight_scale_to_buffer task for RAWINT4 KGroup AMX implementation.

        This method submits the C++-exposed task `write_weight_scale_to_buffer_task` to the
        shared CPUInfer queue. The pointer lists should be plain integer lists (e.g. from
        tensor.data_ptr()).
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot submit write_weight_scale_to_buffer task.")

        if not hasattr(self.moe, "write_weight_scale_to_buffer_task"):
            raise NotImplementedError(
                "write_weight_scale_to_buffer_task is not available for this backend implementation."
            )

        self.cpu_infer.submit(
            self.moe.write_weight_scale_to_buffer_task(
                gpu_tp_count,
                expert_id,
                w13_weight_ptrs,
                w13_scale_ptrs,
                w2_weight_ptrs,
                w2_scale_ptrs,
            )
        )

    def sync_write_weight_scale_to_buffer(self):
        """
        Block until previously submitted write_weight_scale_to_buffer tasks finish.
        """
        # The CPUInfer.sync() call blocks until pending tasks complete.
        self.cpu_infer.sync()
