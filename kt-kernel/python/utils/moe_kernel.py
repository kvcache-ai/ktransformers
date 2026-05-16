import os
import torch
from typing import List, Optional

# Use relative imports for package structure
from ..experts_base import BaseMoEWrapper
from .loader import SafeTensorLoader
from kt_kernel_ext.moe import MOEConfig
from .mesh import amx_helpers as mesh_amx

try:
    from kt_kernel_ext.moe import Int8_KERNEL_MOE

    _HAS_INT8_SUPPORT = True
except (ImportError, AttributeError):
    Int8_KERNEL_MOE = None
    _HAS_INT8_SUPPORT = False
try:
    from kt_kernel_ext.moe import Int4_KERNEL_MOE

    _HAS_INT4_SUPPORT = True
except (ImportError, AttributeError):
    Int4_KERNEL_MOE = None
    _HAS_INT4_SUPPORT = False


class GeneralMoEWrapper(BaseMoEWrapper):
    """
    moe-based MoE wrapper implementation.
    Supports MOE_INT4 and MOE_INT8 quantization methods.
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
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "MOE_INT8",
        numa_nodes: Optional[List[int]] = None,
        weight_strategy: str = "legacy",
        max_tier0_experts: Optional[int] = None,
        num_moe_layers: Optional[int] = None,
    ):
        """
        Initialize general MoE Wrapper.

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
            numa_nodes: Explicit NUMA node IDs for the CPU subpools. If None,
                        use detected NUMA nodes in ascending order.
            weight_path: Path to weights (SafeTensor format)
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer. Defaults to 0.
            method: general quantization method ("MOE_INT4" or "MOE_INT8")
        """
        if not _HAS_INT4_SUPPORT and method == "MOE_INT4":
            raise RuntimeError(
                "MoE_INT4 backend not available. kt_kernel_ext was not compiled with int4 support.\n"
                "Please recompile with int4 enabled."
            )
        if not _HAS_INT8_SUPPORT and method == "MOE_INT8":
            raise RuntimeError(
                "MoE_INT8 backend not available. kt_kernel_ext was not compiled with int8 support.\n"
                "Please recompile with int8 enabled."
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

        # moe-specific: Check if we should load merged safetensor weights
        self.load_merged_weight = False
        import glob

        if glob.glob(os.path.join(weight_path, "*.safetensors")):
            self.load_merged_weight = True

        # Initialize SafeTensor loader (singleton)
        if self.load_merged_weight:
            resolved_weight_path = os.path.abspath(weight_path)
            if (
                GeneralMoEWrapper._safetensor_loader_instance is None
                or GeneralMoEWrapper._safetensor_loader_path != resolved_weight_path
            ):
                GeneralMoEWrapper._safetensor_loader_instance = SafeTensorLoader(weight_path)
                GeneralMoEWrapper._safetensor_loader_path = resolved_weight_path
            self.safetensor_loader = GeneralMoEWrapper._safetensor_loader_instance

        # moe-specific weight storage
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

        # Create MoE module based on method
        if self.method == "MOE_INT4":
            self.moe = Int4_KERNEL_MOE(moe_config)
        elif self.method == "MOE_INT8":
            self.moe = Int8_KERNEL_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported MoE method: {self.method}")

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

            gate_ptrs, up_ptrs, down_ptrs, gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs = (
                mesh_amx.assign_amx_weight_views(self, w)
            )

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
        mesh_amx.apply_mesh_moe_config(self, moe_config, use_iouring=False)

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

        # Create MoE module based on moe method
        if self.method == "MOE_INT4":
            self.moe = Int4_KERNEL_MOE(moe_config)
        elif self.method == "MOE_INT8":
            self.moe = Int8_KERNEL_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported MoE method: {self.method}")

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

    def close(self):
        super().close()
        self.gate_weights = None
        self.up_weights = None
        self.down_weights = None
        self.gate_scales = None
        self.up_scales = None
        self.down_scales = None
        if BaseMoEWrapper._active_wrapper_count == 0:
            if GeneralMoEWrapper._safetensor_loader_instance is not None:
                GeneralMoEWrapper._safetensor_loader_instance.close_all_handles()
            GeneralMoEWrapper._safetensor_loader_instance = None
            GeneralMoEWrapper._safetensor_loader_path = None
