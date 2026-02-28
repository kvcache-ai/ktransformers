import os
import torch
import ctypes
from typing import Optional

# Use relative imports for package structure
from ..experts_base import BaseMoEWrapper
from .loader import SafeTensorLoader, CompressedSafeTensorLoader, FP8SafeTensorLoader, BF16SafeTensorLoader
from kt_kernel_ext.moe import MOEConfig
import kt_kernel_ext.moe as _moe_mod

AMXInt4_MOE = getattr(_moe_mod, "AMXInt4_MOE", None)
AMXInt8_MOE = getattr(_moe_mod, "AMXInt8_MOE", None)
AMXInt4_KGroup_MOE = getattr(_moe_mod, "AMXInt4_KGroup_MOE", None)
AMXFP8_MOE = getattr(_moe_mod, "AMXFP8_MOE", None)
AMXBF16_MOE = getattr(_moe_mod, "AMXBF16_MOE", None)
AMXFP8PerChannel_MOE = getattr(_moe_mod, "AMXFP8PerChannel_MOE", None)

_HAS_AMXINT4_SUPPORT = AMXInt4_MOE is not None
_HAS_AMXINT8_SUPPORT = AMXInt8_MOE is not None
_HAS_RAWINT4_SUPPORT = AMXInt4_KGroup_MOE is not None
_HAS_FP8_SUPPORT = AMXFP8_MOE is not None
_HAS_BF16_SUPPORT = AMXBF16_MOE is not None
_HAS_FP8_PERCHANNEL_SUPPORT = AMXFP8PerChannel_MOE is not None


class AMXMoEWrapper(BaseMoEWrapper):
    """
    AMX-based MoE wrapper implementation.
    Supports AMXINT4 and AMXINT8 quantization methods.
    """

    _safetensor_loader_instance = None  # Singleton SafeTensorLoader

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
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
        )

        # AMX-specific: Check if we should load merged safetensor weights
        self.load_merged_weight = False
        import glob

        if glob.glob(os.path.join(weight_path, "*.safetensors")):
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
    """Wrapper for RAWINT4/FP8/FP8_PERCHANNEL/BF16 experts stored in compressed SafeTensor format."""

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
    ):
        if method == "RAWINT4" and not _HAS_RAWINT4_SUPPORT:
            raise RuntimeError(
                "RAWINT4 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (VNNI optional)\n"
                "Please recompile kt_kernel_ext with AVX512 enabled."
            )
        if method == "FP8" and not _HAS_FP8_SUPPORT:
            raise RuntimeError(
                "FP8 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI\n"
                "Please recompile kt_kernel_ext with AVX512 + BF16 + VBMI enabled."
            )
        if method == "FP8_PERCHANNEL" and not _HAS_FP8_PERCHANNEL_SUPPORT:
            raise RuntimeError(
                "FP8_PERCHANNEL backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI\n"
                "Please recompile kt_kernel_ext with AVX512 + BF16 + VBMI enabled."
            )
        if method == "BF16" and not _HAS_BF16_SUPPORT:
            raise RuntimeError(
                "BF16 backend not available. Required ISA:\n"
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
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
        )

        if NativeMoEWrapper._native_loader_instance is None:
            if method == "RAWINT4":
                NativeMoEWrapper._native_loader_instance = CompressedSafeTensorLoader(weight_path)
            elif method == "FP8":
                NativeMoEWrapper._native_loader_instance = FP8SafeTensorLoader(weight_path)
            elif method == "FP8_PERCHANNEL":
                # Use FP8SafeTensorLoader with per-channel scale format
                NativeMoEWrapper._native_loader_instance = FP8SafeTensorLoader(weight_path, scale_suffix="weight_scale")
            elif method == "BF16":
                NativeMoEWrapper._native_loader_instance = BF16SafeTensorLoader(weight_path)
            else:
                raise NotImplementedError(f"Unsupported method for NativeMoEWrapper: {method}")
        self.loader = NativeMoEWrapper._native_loader_instance

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
        raise NotImplementedError("RAWINT4 wrapper expects pre-quantized safetensor weights.")

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        import time

        t0 = time.time()
        base_key = f"model.layers.{self.layer_idx}"
        weights = self.loader.load_experts(base_key)
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
            self.moe = AMXInt4_KGroup_MOE(moe_config)
        elif self.method == "FP8":
            moe_config.quant_config.bits = 8
            moe_config.quant_config.group_size = 128
            moe_config.quant_config.zero_point = False
            self.moe = AMXFP8_MOE(moe_config)
        elif self.method == "FP8_PERCHANNEL":
            moe_config.quant_config.bits = 8
            moe_config.quant_config.per_channel = True
            moe_config.quant_config.zero_point = False
            self.moe = AMXFP8PerChannel_MOE(moe_config)
        elif self.method == "BF16":
            # BF16 has no quantization config needed
            self.moe = AMXBF16_MOE(moe_config)
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
