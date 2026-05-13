import gc
import logging
import os
import torch
import ctypes
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Use relative imports for package structure
from ..experts_base import BaseMoEWrapper
from .loader import (
    SafeTensorLoader,
    CompressedSafeTensorLoader,
    FP8SafeTensorLoader,
    BF16SafeTensorLoader,
    GPTQSafeTensorLoader,
    MXFP4SafeTensorLoader,
)
from kt_kernel_ext.moe import MOEConfig
import kt_kernel_ext.moe as _moe_mod

AMXInt4_MOE = getattr(_moe_mod, "AMXInt4_MOE", None)
AMXInt8_MOE = getattr(_moe_mod, "AMXInt8_MOE", None)
AMXInt4_KGroup_MOE = getattr(_moe_mod, "AMXInt4_KGroup_MOE", None)
AMXFP4_KGroup_MOE = getattr(_moe_mod, "AMXFP4_KGroup_MOE", None)
AMXFP8_MOE = getattr(_moe_mod, "AMXFP8_MOE", None)
AMXBF16_MOE = getattr(_moe_mod, "AMXBF16_MOE", None)
AMXFP8PerChannel_MOE = getattr(_moe_mod, "AMXFP8PerChannel_MOE", None)
AVX2BF16_MOE = getattr(_moe_mod, "AVX2BF16_MOE", None)
AVX2FP8_MOE = getattr(_moe_mod, "AVX2FP8_MOE", None)
AVX2GPTQInt4_MOE = getattr(_moe_mod, "AVX2GPTQInt4_MOE", None)
AVX2RawInt4_MOE = getattr(_moe_mod, "AVX2RawInt4_MOE", None)
AVXVNNI256GPTQInt4_MOE = getattr(_moe_mod, "AVXVNNI256GPTQInt4_MOE", None)
AVXVNNI256RawInt4_MOE = getattr(_moe_mod, "AVXVNNI256RawInt4_MOE", None)

_HAS_AMXINT4_SUPPORT = AMXInt4_MOE is not None
_HAS_AMXINT8_SUPPORT = AMXInt8_MOE is not None
_HAS_RAWINT4_SUPPORT = AMXInt4_KGroup_MOE is not None
_HAS_MXFP4_SUPPORT = AMXFP4_KGroup_MOE is not None
_HAS_FP8_SUPPORT = AMXFP8_MOE is not None
_HAS_BF16_SUPPORT = AMXBF16_MOE is not None
_HAS_FP8_PERCHANNEL_SUPPORT = AMXFP8PerChannel_MOE is not None
_HAS_AVX2_BF16_SUPPORT = AVX2BF16_MOE is not None
_HAS_AVX2_FP8_SUPPORT = AVX2FP8_MOE is not None
_HAS_AVX2_GPTQ_INT4_SUPPORT = AVX2GPTQInt4_MOE is not None
_HAS_AVX2_RAWINT4_SUPPORT = AVX2RawInt4_MOE is not None
_HAS_AVXVNNI256_GPTQ_INT4_SUPPORT = AVXVNNI256GPTQInt4_MOE is not None
_HAS_AVXVNNI256_RAW_INT4_SUPPORT = AVXVNNI256RawInt4_MOE is not None
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


class AMXMoEWrapper(BaseMoEWrapper):
    """
    AMX-based MoE wrapper implementation.
    Supports AMXINT4 and AMXINT8 quantization methods.
    """

    _safetensor_loader_instance = None  # Singleton SafeTensorLoader
    _safetensor_loader_path = None

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
        numa_nodes: Optional[List[int]] = None,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "AMXINT4",
        weight_strategy: str = "auto",
        max_tier0_experts: Optional[int] = None,
        num_moe_layers: Optional[int] = None,
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
            numa_nodes=numa_nodes,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
            weight_strategy=weight_strategy,
            max_tier0_experts=max_tier0_experts,
            num_moe_layers=num_moe_layers,
        )

        # AMX-specific: Check if we should load merged safetensor weights
        self.load_merged_weight = False
        import glob

        if glob.glob(os.path.join(weight_path, "*.safetensors")):
            self.load_merged_weight = True

        # Initialize SafeTensor loader (singleton)
        if self.load_merged_weight:
            resolved_weight_path = os.path.abspath(weight_path)
            if (
                AMXMoEWrapper._safetensor_loader_instance is None
                or AMXMoEWrapper._safetensor_loader_path != resolved_weight_path
            ):
                AMXMoEWrapper._safetensor_loader_instance = SafeTensorLoader(weight_path)
                AMXMoEWrapper._safetensor_loader_path = resolved_weight_path
            self.safetensor_loader = AMXMoEWrapper._safetensor_loader_instance

        # AMX-specific weight storage
        self.gate_weights = None
        self.up_weights = None
        self.down_weights = None
        self.gate_scales = None
        self.up_scales = None
        self.down_scales = None
        self._mmap_keepalive = None
        self._uses_mmap_weights = False
        self._uses_iouring_weights = False

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
        moe_config.resident_cache_policy = self.residency_policy

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

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        gate_ptr = 0
        up_ptr = 0
        down_ptr = 0

        gate_ptrs = []
        up_ptrs = []
        down_ptrs = []

        gate_scale_ptrs = []
        up_scale_ptrs = []
        down_scale_ptrs = []

        # Determine I/O backend: io_uring or mmap
        use_iouring = hasattr(self, "io_backend") and self.io_backend == "IOURING"
        use_mmap = self.weight_strategy == "tiered" and not use_iouring

        if use_mmap and not self.load_merged_weight:
            print(
                f"[AMXMoEWrapper] layer={self.layer_idx} requested tiered mmap loading, "
                "but merged safetensors were not found; falling back to legacy resident loading"
            )
            use_mmap = False
            self.weight_strategy = "legacy"

        if use_mmap and self.cpu_save:
            print(
                f"[AMXMoEWrapper] layer={self.layer_idx} requested tiered mmap loading during cpu_save, "
                "which requires resident buffers; falling back to legacy loading"
            )
            use_mmap = False
            self.weight_strategy = "legacy"

        if use_iouring:
            if not self.load_merged_weight:
                raise RuntimeError(
                    f"[AMXMoEWrapper] layer={self.layer_idx} io_uring requires merged AMX safetensors "
                    f"under weight_path={self.weight_path!r}"
                )
            if self.cpu_save:
                raise RuntimeError(f"[AMXMoEWrapper] layer={self.layer_idx} io_uring cannot be used during cpu_save")

        if self.load_merged_weight:
            base_key = f"blk.{self.layer_idx}"

            if use_iouring:
                # io_uring path: load file descriptors and offsets.
                from .async_io_manager import get_global_async_reader

                direct_io_requested = os.environ.get("KT_IOURING_DIRECT", "1") not in ("0", "false", "False")
                file_slots = self.safetensor_loader.load_experts_iouring(
                    base_key,
                    use_direct_io=direct_io_requested,
                )
                if direct_io_requested and not file_slots.get("direct_io", False):
                    raise RuntimeError(
                        f"[AMXMoEWrapper] layer={self.layer_idx} requested KT_IOURING_DIRECT=1 "
                        "but SafeTensorLoader did not return direct I/O slots"
                    )
                print(
                    "[AMXMoEWrapper] "
                    f"layer={self.layer_idx} backend=IOURING direct_io={file_slots.get('direct_io', False)} "
                    "mmap_baseline=false "
                    f"policy={self.residency_policy} capacity={self.max_resident_experts or self.max_tier0_experts}"
                )

                # Store file slots for C++ consumption
                # Format: [tp_part_idx][expert_id] -> (fd, offset, size)
                self.gate_file_slots = file_slots["gate"]
                self.up_file_slots = file_slots["up"]
                self.down_file_slots = file_slots["down"]
                self.gate_scale_file_slots = file_slots["gate_scale"]
                self.up_scale_file_slots = file_slots["up_scale"]
                self.down_scale_file_slots = file_slots["down_scale"]

                # Get global async reader
                self.async_reader = get_global_async_reader()

            elif use_mmap:
                # mmap path: load as memory-mapped views
                w = self.safetensor_loader.load_experts_mmap(base_key)
            else:
                # legacy path: load into malloc buffers
                w = self.safetensor_loader.load_experts(base_key)

            if not use_iouring:
                self.gate_weights = w["gate"]
                self.up_weights = w["up"]
                self.down_weights = w["down"]
                self.gate_scales = w["gate_scale"]
                self.up_scales = w["up_scale"]
                self.down_scales = w["down_scale"]

                # Get pointers to weight arrays
                gate_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in self.gate_weights]

                up_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in self.up_weights]

                down_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in self.down_weights]

                gate_scale_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in self.gate_scales]

                up_scale_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in self.up_scales]

                down_scale_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in self.down_scales]

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
        moe_config.use_mmap = use_mmap
        moe_config.max_tier0_experts = self.max_tier0_experts
        moe_config.max_resident_experts = self._mesh_slot_pool_capacity()
        if hasattr(moe_config, "mesh_prefill_layer_mode_enabled"):
            moe_config.mesh_prefill_layer_mode_enabled = self._mesh_prefill_layer_mode_enabled()
        if hasattr(moe_config, "mesh_decode_resident_experts"):
            moe_config.mesh_decode_resident_experts = self._mesh_config_resident_experts()
        moe_config.resident_cache_policy = self.residency_policy
        if hasattr(moe_config, "enable_cache_stats"):
            moe_config.enable_cache_stats = self.enable_cache_stats
        if hasattr(moe_config, "mesh_lookahead_enabled"):
            lookahead_env = os.environ.get("KT_MESH_LOOKAHEAD")
            moe_config.mesh_lookahead_enabled = use_iouring and lookahead_env not in ("0", "false", "False", "FALSE")
            if hasattr(moe_config, "mesh_topk_fallback_enabled"):
                full_gate_enabled = os.environ.get("KT_MESH_FULL_GATE", "1") not in ("0", "false", "False", "FALSE")
                topk_fallback_default = "0" if full_gate_enabled else "1"
                moe_config.mesh_topk_fallback_enabled = os.environ.get(
                    "KT_MESH_TOPK_FALLBACK", topk_fallback_default
                ) not in ("0", "false", "False", "FALSE")
            moe_config.mesh_lookahead_weight = float(os.environ.get("KT_MESH_LOOKAHEAD_WEIGHT", "1.0"))
            moe_config.mesh_heat_gamma = float(os.environ.get("KT_MESH_HEAT_GAMMA", "0.7"))
            moe_config.mesh_heat_beta = float(os.environ.get("KT_MESH_HEAT_BETA", "0.5"))
            moe_config.mesh_transition_alpha = float(os.environ.get("KT_MESH_TRANSITION_ALPHA", "0.5"))
            # Heat is an eviction signal in the paper path. It must not trigger
            # proactive expert reads; only deferred experts are prefetched.
            moe_config.mesh_prefetch_budget = 0
            moe_config.mesh_coldstart_prefill_enabled = False
            moe_config.mesh_coldstart_prefill_limit = 0
            if hasattr(moe_config, "mesh_memory_guard_enabled"):
                moe_config.mesh_memory_guard_enabled = use_iouring and os.environ.get(
                    "KT_MESH_MEMORY_GUARD", "1"
                ) not in ("0", "false", "False", "FALSE")
                moe_config.mesh_memory_high_watermark = float(os.environ.get("KT_MESH_MEMORY_HIGH_WATERMARK", "0.95"))
                moe_config.mesh_memory_target_watermark = float(
                    os.environ.get("KT_MESH_MEMORY_TARGET_WATERMARK", "0.90")
                )
                moe_config.mesh_memory_check_interval = int(os.environ.get("KT_MESH_MEMORY_CHECK_INTERVAL", "64"))
                moe_config.mesh_memory_max_demotes_per_check = int(
                    os.environ.get("KT_MESH_MEMORY_MAX_DEMOTES_PER_CHECK", "8")
                )
        if hasattr(moe_config, "iouring_direct_io"):
            moe_config.iouring_direct_io = bool(file_slots.get("direct_io", False)) if use_iouring else False
        if use_iouring:
            moe_config.use_mmap = False
            moe_config.set_iouring_file_slots(
                self.gate_file_slots,
                self.gate_scale_file_slots,
                self.up_file_slots,
                self.up_scale_file_slots,
                self.down_file_slots,
                self.down_scale_file_slots,
                self.async_reader,
            )
            self._async_reader_keepalive = self.async_reader

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

        if use_mmap:
            self._register_moe_with_provider()
            self._register_amx_mmap_regions()

        # Load weights
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()
        self._uses_mmap_weights = use_mmap
        self._uses_iouring_weights = use_iouring

        # Clean up temporary weight storage if using merged weights
        if self.load_merged_weight and not use_mmap and not use_iouring:
            del self.gate_weights
            del self.up_weights
            del self.down_weights
            del self.gate_scales
            del self.up_scales
            del self.down_scales

    def cache_stats_snapshot(self):
        if self.moe is None or not hasattr(self.moe, "cache_stats_snapshot"):
            return {}
        return dict(self.moe.cache_stats_snapshot())

    def _register_amx_mmap_regions(self):
        """Register AMX mmap source regions for provider prefetch."""
        if self._provider is None:
            return

        from .weight_provider import MmapWeightRegion

        self._provider.clear_layer_regions(self.layer_idx)

        for proj_name, weights, scales in (
            ("gate", self.gate_weights, self.gate_scales),
            ("up", self.up_weights, self.up_scales),
            ("down", self.down_weights, self.down_scales),
        ):
            # AMX INT4/INT8 expert tensors are sharded by NUMA node:
            #   weights[numa_idx][expert_id]
            # Register every NUMA slice for the same logical expert so provider
            # prefetch can warm all mmap regions backing that expert.
            for numa_idx, numa_weights in enumerate(weights):
                for expert_id, weight in enumerate(numa_weights):
                    weight_region = MmapWeightRegion.__new__(MmapWeightRegion)
                    weight_region.ptr = int(weight.ctypes.data)
                    weight_region.n_bytes = int(weight.nbytes)
                    weight_region._view = weight
                    self._provider.register_mmap_region(self.layer_idx, f"{proj_name}_weight", expert_id, weight_region)

                    if scales is not None:
                        scale = scales[numa_idx][expert_id]
                        scale_region = MmapWeightRegion.__new__(MmapWeightRegion)
                        scale_region.ptr = int(scale.ctypes.data)
                        scale_region.n_bytes = int(scale.nbytes)
                        scale_region._view = scale
                        self._provider.register_mmap_region(
                            self.layer_idx, f"{proj_name}_scale", expert_id, scale_region
                        )


class NativeMoEWrapper(BaseMoEWrapper):
    """Wrapper for RAWINT4/FP8/FP8_PERCHANNEL/BF16 experts stored in compressed SafeTensor format."""

    _native_loader_instance = None
    _native_loader_signature = None

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
        numa_nodes: Optional[List[int]] = None,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "RAWINT4",
        weight_strategy: str = "auto",
        max_tier0_experts: Optional[int] = None,
        num_moe_layers: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        # Defence in depth: reject swiglu_limit on non-MXFP4 methods even
        # if the experts.py guard is bypassed (e.g., by a future caller
        # that constructs NativeMoEWrapper directly). Origin: kt-sglang 耦合.
        if swiglu_limit != 0.0 and method != "MXFP4":
            raise ValueError(
                f"NativeMoEWrapper received swiglu_limit={swiglu_limit} with "
                f"method={method!r}; the V4-2604B clamp only applies to MXFP4. "
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
        if method == "MXFP4" and not _HAS_MXFP4_SUPPORT:
            raise RuntimeError(
                "MXFP4 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16\n"
                "Please recompile kt_kernel_ext with AVX512 + BF16 enabled."
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
            numa_nodes=numa_nodes,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
            weight_strategy=weight_strategy,
            max_tier0_experts=max_tier0_experts,
            num_moe_layers=num_moe_layers,
            swiglu_limit=swiglu_limit,
        )

        resolved_weight_path = os.path.abspath(weight_path)
        loader_signature = (method, resolved_weight_path)
        if (
            NativeMoEWrapper._native_loader_instance is None
            or NativeMoEWrapper._native_loader_signature != loader_signature
        ):
            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)
            NativeMoEWrapper._native_loader_signature = loader_signature
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
        elif method == "GPTQ_INT4":
            return GPTQSafeTensorLoader(weight_path)
        elif method == "MXFP4":
            return MXFP4SafeTensorLoader(weight_path)
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
        base_key = f"model.layers.{self.layer_idx}"
        use_mmap = self.weight_strategy == "tiered" and self.method == "BF16"
        if use_mmap and self.cpu_save:
            print(
                f"[NativeMoEWrapper] layer={self.layer_idx} requested tiered mmap loading during cpu_save, "
                "which requires resident buffers; falling back to legacy loading"
            )
            use_mmap = False
            self.weight_strategy = "legacy"

        try:
            weights = self.loader.load_experts_mmap(base_key) if use_mmap else self.loader.load_experts(base_key)
        except (ValueError, KeyError):
            # For VL/multimodal models (e.g. Qwen3.5) with 'language_model' prefix.
            base_key = f"model.language_model.layers.{self.layer_idx}"
            weights = self.loader.load_experts_mmap(base_key) if use_mmap else self.loader.load_experts(base_key)
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

        t2 = time.time()

        # Build pointer lists: [numa_id][expert_id] -> pointer
        # Since RAWINT4/FP8/BF16 has no numa sharding, numa dimension is 1
        def _ptr(weight):
            if isinstance(weight, np.ndarray):
                return int(weight.ctypes.data)
            return weight.data_ptr()

        gate_ptrs = [[_ptr(t) for t in self.gate_weights]]
        up_ptrs = [[_ptr(t) for t in self.up_weights]]
        down_ptrs = [[_ptr(t) for t in self.down_weights]]

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
        if self.swiglu_limit != 0.0 and self.method != "MXFP4":
            raise ValueError(
                f"NativeMoEWrapper.load_weights: swiglu_limit="
                f"{self.swiglu_limit} with method={self.method!r}; clamp is "
                f"only valid for MXFP4."
            )
        moe_config.swiglu_limit = self.swiglu_limit

        # Use gate_projs instead of gate_proj for per-expert pointers
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs
        moe_config.use_mmap = use_mmap
        moe_config.max_tier0_experts = self.max_tier0_experts
        moe_config.max_resident_experts = self._mesh_slot_pool_capacity()
        if hasattr(moe_config, "mesh_prefill_layer_mode_enabled"):
            moe_config.mesh_prefill_layer_mode_enabled = self._mesh_prefill_layer_mode_enabled()
        if hasattr(moe_config, "mesh_decode_resident_experts"):
            moe_config.mesh_decode_resident_experts = self._mesh_config_resident_experts()
        moe_config.resident_cache_policy = self.residency_policy

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
            self.moe = AMXFP4_KGroup_MOE(moe_config)
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
        elif self.method == "BF16":
            # BF16 has no quantization config needed
            # Prefer AMX backend, fall back to AVX2
            if _HAS_BF16_SUPPORT:
                self.moe = AMXBF16_MOE(moe_config)
            else:
                self.moe = AVX2BF16_MOE(moe_config)
        t4 = time.time()

        if use_mmap and self.method == "BF16":
            self._uses_mmap_weights = True
            self._register_moe_with_provider()
            self._register_bf16_mmap_regions()

        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()
        t5 = time.time()

        if use_mmap:
            self._mmap_keepalive = (
                self.gate_weights,
                self.up_weights,
                self.down_weights,
            )
            self._uses_mmap_weights = True
        else:
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

    def _register_bf16_mmap_regions(self):
        """Register BF16 mmap source regions for provider prefetch."""
        if self._provider is None:
            return

        from .weight_provider import MmapWeightRegion

        self._provider.clear_layer_regions(self.layer_idx)

        for proj_name, weights in (
            ("gate", self.gate_weights),
            ("up", self.up_weights),
            ("down", self.down_weights),
        ):
            for expert_id, weight in enumerate(weights):
                region = MmapWeightRegion.__new__(MmapWeightRegion)
                region.ptr = int(weight.ctypes.data)
                region.n_bytes = int(weight.nbytes)
                region._view = weight
                self._provider.register_mmap_region(self.layer_idx, proj_name, expert_id, region)

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

    def close(self):
        super().close()
        self._mmap_keepalive = None
        if BaseMoEWrapper._active_wrapper_count == 0:
            NativeMoEWrapper._native_loader_instance = None
            NativeMoEWrapper._native_loader_signature = None
