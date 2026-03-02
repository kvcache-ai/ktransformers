import os
import torch
import ctypes
from typing import Optional

# Use relative imports for package structure
from ..experts_base import BaseMoEWrapper
from .loader import SafeTensorLoader, CompressedSafeTensorLoader, FP8SafeTensorLoader, BF16SafeTensorLoader
from kt_kernel_ext.moe import MOEConfig

try:
    from kt_kernel_ext.moe import AMXInt4_MOE, AMXInt8_MOE, AMXInt4_KGroup_MOE, AMXFP8_MOE, AMXBF16_MOE

    _HAS_AMX_SUPPORT = True
except (ImportError, AttributeError):
    _HAS_AMX_SUPPORT = False
    AMXInt4_MOE, AMXInt8_MOE, AMXInt4_KGroup_MOE, AMXFP8_MOE, AMXBF16_MOE = None, None, None, None, None

try:
    from kt_kernel_ext.moe import AMXFP8PerChannel_MOE

    _HAS_FP8_PERCHANNEL_SUPPORT = True
except (ImportError, AttributeError):
    _HAS_FP8_PERCHANNEL_SUPPORT = False
    AMXFP8PerChannel_MOE = None

from typing import Optional


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
        if not _HAS_AMX_SUPPORT:
            raise RuntimeError(
                "AMX backend not available. kt_kernel_ext was not compiled with AMX support.\n"
                "Please recompile with AMX enabled."
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
        if not _HAS_AMX_SUPPORT:
            raise RuntimeError("AMX backend is not available.")
        if method == "RAWINT4" and AMXInt4_KGroup_MOE is None:
            raise RuntimeError("AMX backend with RAWINT4 support is not available.")
        if method == "FP8" and AMXFP8_MOE is None:
            raise RuntimeError("AMX backend with FP8 support is not available.")
        if method == "FP8_PERCHANNEL" and not _HAS_FP8_PERCHANNEL_SUPPORT:
            raise RuntimeError("AMX backend with FP8 per-channel support is not available.")
        if method == "BF16" and AMXBF16_MOE is None:
            raise RuntimeError("AMX backend with BF16 support is not available.")

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

    def submit_write_weight_scale_to_buffer_async(
        self,
        cuda_stream: int,
        gpu_tp_count: int,
        expert_id: int,
        w13_weight_ptrs,
        w13_scale_ptrs,
        w2_weight_ptrs,
        w2_scale_ptrs,
    ):
        """
        Submit the write_weight_scale_to_buffer task asynchronously using CUDA stream.

        Unlike submit_write_weight_scale_to_buffer, this uses cudaLaunchHostFunc to
        queue the CPU task when the GPU stream reaches this point, allowing non-blocking
        Python execution.

        Args:
            cuda_stream: CUDA stream handle (from torch.cuda.Stream.cuda_stream)
            gpu_tp_count: Number of GPU tensor parallel instances
            expert_id: Expert ID to write
            w13_weight_ptrs: Pointer list for w13 (gate/up) weights
            w13_scale_ptrs: Pointer list for w13 scales
            w2_weight_ptrs: Pointer list for w2 (down) weights
            w2_scale_ptrs: Pointer list for w2 scales
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot submit write_weight_scale_to_buffer task.")

        if not hasattr(self.moe, "write_weight_scale_to_buffer_task"):
            raise NotImplementedError(
                "write_weight_scale_to_buffer_task is not available for this backend implementation."
            )

        self.cpu_infer.submit_with_cuda_stream(
            cuda_stream,
            self.moe.write_weight_scale_to_buffer_task(
                gpu_tp_count,
                expert_id,
                w13_weight_ptrs,
                w13_scale_ptrs,
                w2_weight_ptrs,
                w2_scale_ptrs,
            ),
        )

    def sync_write_weight_scale_to_buffer_async(self, cuda_stream: int, allow_pending: int = 0):
        """
        Synchronize write_weight_scale_to_buffer tasks on a CUDA stream (non-blocking to Python).

        This uses cudaLaunchHostFunc to queue a synchronization callback on the GPU stream.
        Python execution continues immediately; the GPU stream will wait for CPU tasks to complete.

        Args:
            cuda_stream: CUDA stream handle (from torch.cuda.Stream.cuda_stream)
            allow_pending: Number of pending tasks to allow (default 0)
        """
        self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)

    def setup_batch_load_buffers(
        self,
        gpu_tp_count: int,
        pinned_w13_weight_ptrs,
        pinned_w13_scale_ptrs,
        pinned_w2_weight_ptrs,
        pinned_w2_scale_ptrs,
        gpu_w13_weight_ptrs_per_rank,
        gpu_w13_scale_ptrs_per_rank,
        gpu_w2_weight_ptrs_per_rank,
        gpu_w2_scale_ptrs_per_rank,
        cuda_streams,
        w13_weight_expert_nbytes: int,
        w13_scale_expert_nbytes: int,
        w2_weight_expert_nbytes: int,
        w2_scale_expert_nbytes: int,
    ):
        """
        V2 API: Register fixed buffer pointers once during initialization.

        Args:
            gpu_tp_count: Number of GPU tensor parallel instances
            pinned_w13_weight_ptrs: List of pinned buffer pointers [slot0_rank0, ..., slot1_rank0, ...]
            pinned_w13_scale_ptrs: List of pinned buffer pointers for w13 scales
            pinned_w2_weight_ptrs: List of pinned buffer pointers for w2 weights
            pinned_w2_scale_ptrs: List of pinned buffer pointers for w2 scales
            gpu_w13_weight_ptrs_per_rank: GPU destination pointers per rank (IPC)
            gpu_w13_scale_ptrs_per_rank: GPU destination pointers per rank (IPC)
            gpu_w2_weight_ptrs_per_rank: GPU destination pointers per rank (IPC)
            gpu_w2_scale_ptrs_per_rank: GPU destination pointers per rank (IPC)
            cuda_streams: List of per-rank CUDA stream handles
            w13_weight_expert_nbytes: Per-expert w13 weight size in bytes (TP-sharded)
            w13_scale_expert_nbytes: Per-expert w13 scale size in bytes (TP-sharded)
            w2_weight_expert_nbytes: Per-expert w2 weight size in bytes (TP-sharded)
            w2_scale_expert_nbytes: Per-expert w2 scale size in bytes (TP-sharded)
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot setup batch load buffers.")

        if not hasattr(self.moe, "setup_batch_load_buffers"):
            raise NotImplementedError("setup_batch_load_buffers is not available for this backend implementation.")

        self.moe.setup_batch_load_buffers(
            gpu_tp_count,
            list(pinned_w13_weight_ptrs),
            list(pinned_w13_scale_ptrs),
            list(pinned_w2_weight_ptrs),
            list(pinned_w2_scale_ptrs),
            list(gpu_w13_weight_ptrs_per_rank),
            list(gpu_w13_scale_ptrs_per_rank),
            list(gpu_w2_weight_ptrs_per_rank),
            list(gpu_w2_scale_ptrs_per_rank),
            list(cuda_streams),
            w13_weight_expert_nbytes,
            w13_scale_expert_nbytes,
            w2_weight_expert_nbytes,
            w2_scale_expert_nbytes,
        )

    def submit_batch_load_cpu_experts_to_gpu(self, cpu_expert_ids):
        """
        V2 API: Submit batch CPU expert weight loading task to cpuinfer.

        This is a non-blocking call. cpuinfer will:
        1. For each expert, call write_weights_to_buffer() to write to pinned buffer
        2. Issue async cudaMemcpyAsync (H2D) to all ranks using pre-registered IPC pointers
        3. Use double buffering to implement write(e+1) || copy(e) pipeline

        Note: setup_batch_load_buffers() must be called once before using this method.

        Args:
            cpu_expert_ids: List of CPU expert IDs to load
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot submit batch_load_cpu_experts_to_gpu task.")

        if not hasattr(self.moe, "batch_load_cpu_experts_to_gpu_task"):
            raise NotImplementedError(
                "batch_load_cpu_experts_to_gpu_task is not available for this backend implementation."
            )

        self.cpu_infer.submit(self.moe.batch_load_cpu_experts_to_gpu_task(list(cpu_expert_ids)))

    def sync_batch_load_cpu_experts_to_gpu(self):
        """
        Block until previously submitted batch_load_cpu_experts_to_gpu tasks finish.
        """
        self.cpu_infer.sync()

    # ==================== V3 Polling-Based Batch Load API ====================

    def setup_polling_batch_load(
        self,
        num_ranks: int,
        sync_slot_ptrs,
        src_buffer_ptrs_per_rank,
        dst_w13_weight_per_rank,
        dst_w13_scale_per_rank,
        dst_w2_weight_per_rank,
        dst_w2_scale_per_rank,
        stream_ptrs,
        w13_weight_size: int,
        w13_scale_size: int,
        w2_weight_size: int,
        w2_scale_size: int,
    ):
        """
        V3 API: Setup polling-based batch load with per-rank sync slots.

        This sets up persistent polling kernels for each rank. Each kernel
        polls a shared CPU memory flag and copies data when signaled.

        Args:
            num_ranks: Number of GPU TP ranks
            sync_slot_ptrs: Per-rank sync slot pointers (pinned memory, 64 bytes each)
            src_buffer_ptrs_per_rank: List of [8] source buffer pointers per rank
                Format per rank: [w13_w_s0, w13_w_s1, w13_s_s0, w13_s_s1, w2_w_s0, w2_w_s1, w2_s_s0, w2_s_s1]
            dst_w13_weight_per_rank: Per-rank GPU destination for w13 weight
            dst_w13_scale_per_rank: Per-rank GPU destination for w13 scale
            dst_w2_weight_per_rank: Per-rank GPU destination for w2 weight
            dst_w2_scale_per_rank: Per-rank GPU destination for w2 scale
            stream_ptrs: Per-rank CUDA stream handles for persistent kernels
            w13_weight_size: Per-expert size for w13 weight in bytes
            w13_scale_size: Per-expert size for w13 scale in bytes
            w2_weight_size: Per-expert size for w2 weight in bytes
            w2_scale_size: Per-expert size for w2 scale in bytes
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot setup polling batch load.")

        if not hasattr(self.moe, "setup_polling_batch_load"):
            raise NotImplementedError("setup_polling_batch_load is not available for this backend implementation.")

        self.moe.setup_polling_batch_load(
            num_ranks,
            list(sync_slot_ptrs),
            [list(bufs) for bufs in src_buffer_ptrs_per_rank],
            list(dst_w13_weight_per_rank),
            list(dst_w13_scale_per_rank),
            list(dst_w2_weight_per_rank),
            list(dst_w2_scale_per_rank),
            list(stream_ptrs),
            w13_weight_size,
            w13_scale_size,
            w2_weight_size,
            w2_scale_size,
        )

    def launch_polling_kernel(self, local_rank: int, total_experts: int = -1):
        """
        V3 API: Launch persistent polling kernel for the local rank.

        Each rank calls this to launch its own persistent kernel.
        Kernels run until total_experts reached or shutdown.

        Args:
            local_rank: The rank to launch kernel for (must match calling process's rank)
            total_experts: Total number of experts to process (-1 for infinite loop)
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot launch polling kernel.")

        if not hasattr(self.moe, "launch_polling_kernel"):
            raise NotImplementedError("launch_polling_kernel is not available for this backend implementation.")

        self.moe.launch_polling_kernel(local_rank, total_experts)

    def shutdown_polling_kernel(self, local_rank: int):
        """
        V3 API: Shutdown polling kernel for the local rank.

        Args:
            local_rank: The rank to shutdown (must match calling process's rank)
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot shutdown polling kernel.")

        if not hasattr(self.moe, "shutdown_polling_kernel"):
            raise NotImplementedError("shutdown_polling_kernel is not available for this backend implementation.")

        self.moe.shutdown_polling_kernel(local_rank)

    def shutdown_all_polling_kernels(self):
        """
        V3 API: Shutdown all persistent polling kernels (for cleanup).
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot shutdown polling kernels.")

        if not hasattr(self.moe, "shutdown_all_polling_kernels"):
            raise NotImplementedError("shutdown_all_polling_kernels is not available for this backend implementation.")

        self.moe.shutdown_all_polling_kernels()

    def submit_batch_load_cpu_experts_polling(self, cpu_expert_ids):
        """
        V3 API: Submit polling-based batch CPU expert weight loading task.

        This is a non-blocking call. The cpuinfer will:
        1. For each expert, write to pinned double buffer
        2. Signal GPU via shared memory flag
        3. Wait for GPU to complete copy before signaling next

        Note: setup_polling_batch_load() and launch_polling_kernels() must be
        called before using this method.

        Args:
            cpu_expert_ids: List of CPU expert IDs to load
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot submit polling batch load task.")

        if not hasattr(self.moe, "batch_load_cpu_experts_to_gpu_polling_task"):
            raise NotImplementedError(
                "batch_load_cpu_experts_to_gpu_polling_task is not available for this backend implementation."
            )

        self.cpu_infer.submit(self.moe.batch_load_cpu_experts_to_gpu_polling_task(list(cpu_expert_ids)))

    def sync_batch_load_cpu_experts_polling(self):
        """
        V3 API: Block until previously submitted polling batch load tasks finish.
        """
        self.cpu_infer.sync()


# ============================================================================
# V4 API: Polling Memcpy Worker (dedicated worker thread per rank)
# ============================================================================

def get_polling_memcpy_worker_manager():
    """
    Get the singleton PollingMemcpyWorkerManager instance.

    Returns:
        PollingMemcpyWorkerManager instance from kt_kernel_ext.polling_worker
    """
    try:
        from kt_kernel_ext import polling_worker
        return polling_worker.PollingMemcpyWorkerManager.instance()
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"PollingMemcpyWorkerManager is not available. "
            f"Ensure kt_kernel_ext was compiled with AMX/AVX512 support. Error: {e}"
        )


def create_polling_memcpy_worker(
    rank: int,
    cuda_device: int,
    cpu_core: int,
    sync_slot_ptr: int,
    src_buffer_ptrs: list,
    dst_w13_weight: int,
    dst_w13_scale: int,
    dst_w2_weight: int,
    dst_w2_scale: int,
    w13_weight_size: int,
    w13_scale_size: int,
    w2_weight_size: int,
    w2_scale_size: int,
):
    """
    V4 API: Create a polling memcpy worker for the local rank.

    Each rank should call this to create its own worker thread that will:
    1. Poll sync_slot->signal for DATA_READY
    2. Perform cudaMemcpyAsync for all 4 weight/scale buffers
    3. Set signal = GPU_DONE

    Args:
        rank: Worker's rank ID
        cuda_device: CUDA device ID for this rank
        cpu_core: CPU core to bind worker thread (-1 for auto-select)
        sync_slot_ptr: Pointer to sync slot (pinned memory)
        src_buffer_ptrs: List of 8 source buffer pointers:
            [w13_w_s0, w13_w_s1, w13_s_s0, w13_s_s1, w2_w_s0, w2_w_s1, w2_s_s0, w2_s_s1]
        dst_w13_weight: GPU destination for w13 weight (base pointer)
        dst_w13_scale: GPU destination for w13 scale
        dst_w2_weight: GPU destination for w2 weight
        dst_w2_scale: GPU destination for w2 scale
        w13_weight_size: Per-expert w13 weight size in bytes
        w13_scale_size: Per-expert w13 scale size in bytes
        w2_weight_size: Per-expert w2 weight size in bytes
        w2_scale_size: Per-expert w2 scale size in bytes
    """
    manager = get_polling_memcpy_worker_manager()
    manager.create_worker(
        rank, cuda_device, cpu_core, sync_slot_ptr, src_buffer_ptrs,
        dst_w13_weight, dst_w13_scale, dst_w2_weight, dst_w2_scale,
        w13_weight_size, w13_scale_size, w2_weight_size, w2_scale_size
    )


def start_polling_memcpy_worker():
    """
    V4 API: Start the local polling memcpy worker thread.
    """
    manager = get_polling_memcpy_worker_manager()
    manager.start_worker()


def stop_polling_memcpy_worker():
    """
    V4 API: Stop the local polling memcpy worker thread.
    """
    manager = get_polling_memcpy_worker_manager()
    manager.stop_worker()


def is_polling_memcpy_worker_running() -> bool:
    """
    V4 API: Check if the local polling memcpy worker is running.

    Returns:
        True if worker exists and is running
    """
    manager = get_polling_memcpy_worker_manager()
    return manager.is_worker_running()
