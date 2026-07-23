# AMX SFT MoE Wrapper implementation
# SPDX-License-Identifier: Apache-2.0

"""
AMX-based SFT MoE Wrapper. Forward/backward buffer management is in base class;
this file handles weight loading, LoRA init, and C++ task construction.
"""

from __future__ import annotations

import ctypes
import logging
import os
import glob as _glob
import torch
from typing import Optional, List

from kt_kernel_ext.moe import MOESFTConfig

logger = logging.getLogger(__name__)

from ..utils.loader import BF16SafeTensorLoader, SafeTensorLoader

try:
    from kt_kernel_ext.moe import (
        AMXBF16_SFT_MOE,
        AMXInt8_SFT_MOE,
        AMXInt4_SFT_MOE,
        AMXBF16_SFT_MOE_SkipLoRA,
        AMXInt8_SFT_MOE_SkipLoRA,
        AMXInt4_SFT_MOE_SkipLoRA,
    )

    _HAS_AMX_SFT_SUPPORT = True
except (ImportError, AttributeError):
    _HAS_AMX_SFT_SUPPORT = False
    AMXBF16_SFT_MOE = None
    AMXInt8_SFT_MOE = None
    AMXInt4_SFT_MOE = None
    AMXBF16_SFT_MOE_SkipLoRA = None
    AMXInt8_SFT_MOE_SkipLoRA = None
    AMXInt4_SFT_MOE_SkipLoRA = None

from .base import BaseSFTMoEWrapper, KExpertsSFTBuffer, _supports_authoritative_optimizer_grads

_AMX_M_STEP = 32


# Mapping from method string to C++ SFT MOE class
_SFT_METHOD_TO_CLASS = {
    "AMXBF16_SFT": AMXBF16_SFT_MOE,
    "AMXINT8_SFT": AMXInt8_SFT_MOE,
    "AMXINT4_SFT": AMXInt4_SFT_MOE,
    "AMXBF16_SFT_SkipLoRA": AMXBF16_SFT_MOE_SkipLoRA,
    "AMXINT8_SFT_SkipLoRA": AMXInt8_SFT_MOE_SkipLoRA,
    "AMXINT4_SFT_SkipLoRA": AMXInt4_SFT_MOE_SkipLoRA,
}


class AMXSFTMoEWrapper(BaseSFTMoEWrapper):
    """
    AMX-based SFT MoE wrapper.

    Supports BF16, INT8, INT4, and SkipLoRA variants.
    Forward/backward buffer management is in BaseSFTMoEWrapper;
    this class implements weight loading and C++ task construction.
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
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        max_cache_depth: int = 1,
        method: str = "AMXBF16_SFT",
        group_size: int = 128,
        zero_point: bool = True,
        full_weight_grad: bool = False,
    ):
        if not _HAS_AMX_SFT_SUPPORT:
            raise RuntimeError(
                "AMX SFT backend not available. kt_kernel_ext was not compiled with AMX SFT support.\n"
                "Please recompile with AMX SFT enabled."
            )

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
            full_weight_grad=full_weight_grad,
        )

        self.method = method
        self._is_skip_lora = "SkipLoRA" in method
        # Authoritative optimizer gradients currently rely on the BF16 SFT
        # kernel's overwrite/accumulate/lazy-clear implementation.  Quantized
        # and SkipLoRA backends intentionally retain their legacy lifecycle.
        self._uses_authoritative_optimizer_grads = _supports_authoritative_optimizer_grads(
            method, self.num_gpu_experts
        )
        self.group_size = group_size
        self.zero_point = zero_point

        if method not in _SFT_METHOD_TO_CLASS:
            raise ValueError(f"Unknown SFT method: {method}. Supported: {list(_SFT_METHOD_TO_CLASS.keys())}")

        moe_class = _SFT_METHOD_TO_CLASS[method]
        if moe_class is None:
            raise RuntimeError(f"AMX SFT method '{method}' not available in current build.")

        self.gate_proj: Optional[torch.Tensor] = None
        self.up_proj: Optional[torch.Tensor] = None
        self.down_proj: Optional[torch.Tensor] = None

        self._moe_class = moe_class

    # ========== Template method: C++ task construction ==========

    def _make_forward_task(self, buffer: KExpertsSFTBuffer, save_for_backward: bool):
        return self.moe.forward_sft_task(
            buffer.bsz_tensor.data_ptr(),
            self.num_experts_per_tok,
            buffer.expert_ids_cpu.data_ptr(),
            buffer.weights_cpu.data_ptr(),
            buffer.input_cpu.data_ptr(),
            buffer.output_cpu.data_ptr(),
            save_for_backward,
        )

    def _make_backward_task(
        self,
        buffer: KExpertsSFTBuffer,
        accumulate_optimizer_grads: bool = False,
        optimizer_grad_scale: float = 1.0,
    ):
        if self._is_skip_lora:
            return self.moe.backward_task(
                buffer.grad_output_cpu.data_ptr(),
                buffer.grad_input_cpu.data_ptr(),
                0,
                0,
                0,
                0,
                0,
                0,
                buffer.grad_weights.data_ptr(),
                0,
                0,
                0,  # grad_gate_proj, grad_up_proj, grad_down_proj
            )

        # Base weight grad pointers (nullptr if not in full mode)
        grad_gate_proj_ptr = (
            self.grad_gate_proj_buf.data_ptr() if self._full_weight_grad and self.grad_gate_proj_buf is not None else 0
        )
        grad_up_proj_ptr = (
            self.grad_up_proj_buf.data_ptr() if self._full_weight_grad and self.grad_up_proj_buf is not None else 0
        )
        grad_down_proj_ptr = (
            self.grad_down_proj_buf.data_ptr() if self._full_weight_grad and self.grad_down_proj_buf is not None else 0
        )

        backward_args = (
            buffer.grad_output_cpu.data_ptr(),
            buffer.grad_input_cpu.data_ptr(),
            self.grad_gate_lora_a.data_ptr() if self.lora_rank > 0 else 0,
            self.grad_gate_lora_b.data_ptr() if self.lora_rank > 0 else 0,
            self.grad_up_lora_a.data_ptr() if self.lora_rank > 0 else 0,
            self.grad_up_lora_b.data_ptr() if self.lora_rank > 0 else 0,
            self.grad_down_lora_a.data_ptr() if self.lora_rank > 0 else 0,
            self.grad_down_lora_b.data_ptr() if self.lora_rank > 0 else 0,
            buffer.grad_weights.data_ptr(),
            grad_gate_proj_ptr,
            grad_up_proj_ptr,
            grad_down_proj_ptr,
        )
        if (
            self._uses_authoritative_optimizer_grads
            or bool(accumulate_optimizer_grads)
            or float(optimizer_grad_scale) != 1.0
        ):
            return self.moe.backward_task(
                *backward_args,
                bool(accumulate_optimizer_grads),
                float(optimizer_grad_scale),
            )
        return self.moe.backward_task(*backward_args)

    # ========== Weight loading ==========

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor) -> None:
        if self._weights_loaded:
            return

        if physical_to_logical_map_cpu is None:
            physical_to_logical_map_cpu = torch.arange(self.num_experts, dtype=torch.int64)
        self._physical_to_logical_map_cpu = physical_to_logical_map_cpu.to(
            dtype=torch.int64, device="cpu"
        ).contiguous()
        if self._physical_to_logical_map_cpu.numel() < self.num_experts:
            raise ValueError(
                "physical_to_logical_map_cpu must contain at least "
                f"{self.num_experts} entries, got {self._physical_to_logical_map_cpu.numel()}."
            )

        if self.gate_proj is None and not getattr(self, "_use_projs_path", False):
            self._load_base_weights_from_file()

        config = MOESFTConfig()
        config.expert_num = self.num_experts
        config.num_experts_per_tok = self.num_experts_per_tok
        config.hidden_size = self.hidden_size
        config.intermediate_size = self.moe_intermediate_size
        config.lora_rank = self.lora_rank
        config.lora_alpha = self.lora_alpha
        config.max_cache_depth = self.max_cache_depth
        config.max_len = self._aligned_max_len()
        config.layer_idx = self.layer_idx
        config.share_backward_bb = getattr(self, "share_backward_bb", False)
        config.share_cache_pool = getattr(self, "share_cache_pool", False)
        config.full_weight_grad = self._full_weight_grad
        config.authoritative_optimizer_grads = self._uses_authoritative_optimizer_grads
        config.physical_to_logical_map = self._physical_to_logical_map_cpu.data_ptr()

        if getattr(self, "_use_kt_direct_load", False):
            config.load = True
            config.path = self.weight_path
        elif getattr(self, "_use_projs_path", False):
            config.gate_projs = self._gate_projs_ptrs
            config.up_projs = self._up_projs_ptrs
            config.down_projs = self._down_projs_ptrs
            config.gate_scales = self._gate_scale_ptrs
            config.up_scales = self._up_scale_ptrs
            config.down_scales = self._down_scale_ptrs
            if getattr(self, "_bf16_gate_proj", None) is not None:
                config.gate_proj = self._bf16_gate_proj.data_ptr()
                config.up_proj = self._bf16_up_proj.data_ptr()
                config.down_proj = self._bf16_down_proj.data_ptr()
            if getattr(self, "_has_bwd_projs", False):
                config.gate_bwd_projs = self._gate_bwd_projs_ptrs
                config.up_bwd_projs = self._up_bwd_projs_ptrs
                config.down_bwd_projs = self._down_bwd_projs_ptrs
                config.gate_bwd_scales = self._gate_bwd_scale_ptrs
                config.up_bwd_scales = self._up_bwd_scale_ptrs
                config.down_bwd_scales = self._down_bwd_scale_ptrs
        else:
            config.gate_proj = self.gate_proj.data_ptr()
            config.up_proj = self.up_proj.data_ptr()
            config.down_proj = self.down_proj.data_ptr()

        if self._lora_initialized:
            config.gate_lora_a = self.gate_lora_a.data_ptr()
            config.gate_lora_b = self.gate_lora_b.data_ptr()
            config.up_lora_a = self.up_lora_a.data_ptr()
            config.up_lora_b = self.up_lora_b.data_ptr()
            config.down_lora_a = self.down_lora_a.data_ptr()
            config.down_lora_b = self.down_lora_b.data_ptr()

        config.pool = self.cpu_infer.backend_

        if self.method in ("AMXINT4_KGroup_SFT", "AMXINT4_1KGroup_SFT"):
            config.quant_config.group_size = self.group_size
            config.quant_config.zero_point = self.zero_point

        # Release old C++ MOE object before creating a new one to avoid memory leak
        old_moe = getattr(self, "moe", None)
        if old_moe is not None:
            del old_moe
            import gc
            gc.collect()

        self.moe = self._moe_class(config)

        self.cpu_infer.submit(self.moe.load_weights_task())
        self.cpu_infer.sync()

        if os.environ.get("KT_SFT_ENABLE_WARMUP", "0") == "1":
            self.cpu_infer.submit(self.moe.warm_up_task())
            self.cpu_infer.sync()

        # Release Python-side weight tensors (C++ copied them)
        # In full_weight_grad mode, keep them for nn.Parameter initialization
        if not self._full_weight_grad:
            self.gate_proj = None
            self.up_proj = None
            self.down_proj = None

        if getattr(self, "_bf16_gate_proj", None) is not None:
            self._bf16_gate_proj = None
            self._bf16_up_proj = None
            self._bf16_down_proj = None

        if getattr(self, "_use_projs_path", False):
            for attr in [
                "_gate_weights_per_numa",
                "_up_weights_per_numa",
                "_down_weights_per_numa",
                "_gate_scales_per_numa",
                "_up_scales_per_numa",
                "_down_scales_per_numa",
                "_gate_projs_ptrs",
                "_up_projs_ptrs",
                "_down_projs_ptrs",
                "_gate_scale_ptrs",
                "_up_scale_ptrs",
                "_down_scale_ptrs",
            ]:
                setattr(self, attr, None)
            if getattr(self, "_has_bwd_projs", False):
                for attr in [
                    "_gate_bwd_weights_per_numa",
                    "_up_bwd_weights_per_numa",
                    "_down_bwd_weights_per_numa",
                    "_gate_bwd_scales_per_numa",
                    "_up_bwd_scales_per_numa",
                    "_down_bwd_scales_per_numa",
                    "_gate_bwd_projs_ptrs",
                    "_up_bwd_projs_ptrs",
                    "_down_bwd_projs_ptrs",
                    "_gate_bwd_scale_ptrs",
                    "_up_bwd_scale_ptrs",
                    "_down_bwd_scale_ptrs",
                ]:
                    setattr(self, attr, None)

        self._weights_loaded = True

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ) -> None:
        self.gate_proj = gate_proj.contiguous()
        self.up_proj = up_proj.contiguous()
        self.down_proj = down_proj.contiguous()
        self.load_weights(physical_to_logical_map_cpu)
        del gate_proj, up_proj, down_proj

    def _load_base_weights_from_file(self) -> None:
        if not hasattr(self, "weight_path") or self.weight_path is None:
            raise RuntimeError(
                "weight_path not set. Cannot load weights from file. "
                "Either set weight_path or call load_weights_from_tensors() instead."
            )

        kt_layer_dir = os.path.join(self.weight_path, f"_layer_{self.layer_idx}")
        if os.path.isdir(kt_layer_dir):
            kt_files = _glob.glob(os.path.join(kt_layer_dir, "_numa_0", "*.kt"))
            if kt_files:
                self._use_kt_direct_load = True
                return

        if "BF16" in self.method:
            loader = BF16SafeTensorLoader(self.weight_path)
            base_key = f"model.layers.{self.layer_idx}"
        else:
            loader = SafeTensorLoader(self.weight_path)
            base_key = f"blk.{self.layer_idx}"

        experts_data = loader.load_experts(base_key, device="cpu")

        gate_weights: List[torch.Tensor] = experts_data["gate"]
        up_weights: List[torch.Tensor] = experts_data["up"]
        down_weights: List[torch.Tensor] = experts_data["down"]

        if "BF16" in self.method:
            self.gate_proj = torch.stack(gate_weights, dim=0).contiguous()
            self.up_proj = torch.stack(up_weights, dim=0).contiguous()
            self.down_proj = torch.stack(down_weights, dim=0).contiguous()
        else:

            def _make_ptrs(arrays_per_numa):
                return [
                    [
                        ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                        for et in numa_array
                    ]
                    for numa_array in arrays_per_numa
                ]

            self._gate_weights_per_numa = gate_weights
            self._up_weights_per_numa = up_weights
            self._down_weights_per_numa = down_weights
            self._gate_scales_per_numa = experts_data["gate_scale"]
            self._up_scales_per_numa = experts_data["up_scale"]
            self._down_scales_per_numa = experts_data["down_scale"]
            self._validate_prepartitioned_weights()

            self._gate_projs_ptrs = _make_ptrs(gate_weights)
            self._up_projs_ptrs = _make_ptrs(up_weights)
            self._down_projs_ptrs = _make_ptrs(down_weights)
            self._gate_scale_ptrs = _make_ptrs(experts_data["gate_scale"])
            self._up_scale_ptrs = _make_ptrs(experts_data["up_scale"])
            self._down_scale_ptrs = _make_ptrs(experts_data["down_scale"])

            if "gate_bwd" in experts_data:
                self._gate_bwd_weights_per_numa = experts_data["gate_bwd"]
                self._up_bwd_weights_per_numa = experts_data["up_bwd"]
                self._down_bwd_weights_per_numa = experts_data["down_bwd"]
                self._gate_bwd_scales_per_numa = experts_data["gate_bwd_scale"]
                self._up_bwd_scales_per_numa = experts_data["up_bwd_scale"]
                self._down_bwd_scales_per_numa = experts_data["down_bwd_scale"]

                self._gate_bwd_projs_ptrs = _make_ptrs(experts_data["gate_bwd"])
                self._up_bwd_projs_ptrs = _make_ptrs(experts_data["up_bwd"])
                self._down_bwd_projs_ptrs = _make_ptrs(experts_data["down_bwd"])
                self._gate_bwd_scale_ptrs = _make_ptrs(experts_data["gate_bwd_scale"])
                self._up_bwd_scale_ptrs = _make_ptrs(experts_data["up_bwd_scale"])
                self._down_bwd_scale_ptrs = _make_ptrs(experts_data["down_bwd_scale"])
                self._has_bwd_projs = True
            else:
                self._has_bwd_projs = False

            self.gate_proj = None
            self.up_proj = None
            self.down_proj = None
            self._use_projs_path = True

        loader.close_all_handles()

    def _aligned_max_len(self) -> int:
        return ((self.chunked_prefill_size + _AMX_M_STEP - 1) // _AMX_M_STEP) * _AMX_M_STEP

    def _validate_prepartitioned_weights(self) -> None:
        numa_count = len(self._gate_weights_per_numa)
        if self.moe_intermediate_size % self.threadpool_count != 0:
            raise ValueError(
                f"moe_intermediate_size={self.moe_intermediate_size} must be divisible by "
                f"threadpool_count={self.threadpool_count} for {self.method} SFT."
            )
        if numa_count != self.threadpool_count:
            raise ValueError(
                f"{self.method} SFT pre-partitioned expert weights have {numa_count} NUMA partitions, "
                f"but CPUInfer was created with threadpool_count={self.threadpool_count}. "
                f"Use --kt-threadpool-count {numa_count} for this weight directory, or convert weights "
                "for the requested threadpool count."
            )

        collections = {
            "gate": self._gate_weights_per_numa,
            "up": self._up_weights_per_numa,
            "down": self._down_weights_per_numa,
            "gate_scale": self._gate_scales_per_numa,
            "up_scale": self._up_scales_per_numa,
            "down_scale": self._down_scales_per_numa,
        }
        for name, per_numa in collections.items():
            if len(per_numa) != numa_count:
                raise ValueError(f"{name} has {len(per_numa)} NUMA partitions, expected {numa_count}.")
            for numa_id, entries in enumerate(per_numa):
                if len(entries) != self.num_experts:
                    raise ValueError(
                        f"{name}[numa={numa_id}] has {len(entries)} experts, expected {self.num_experts}."
                    )

        for numa_id in range(numa_count):
            gate_scale_len = self._gate_scales_per_numa[numa_id][0].size
            up_scale_len = self._up_scales_per_numa[numa_id][0].size
            down_scale_len = self._down_scales_per_numa[numa_id][0].size
            expected_intermediate = self.moe_intermediate_size // self.threadpool_count
            if gate_scale_len != expected_intermediate or up_scale_len != expected_intermediate:
                raise ValueError(
                    f"{self.method} gate/up scale length for NUMA {numa_id} is "
                    f"{gate_scale_len}/{up_scale_len}, expected {expected_intermediate}."
                )
            if down_scale_len != self.hidden_size:
                raise ValueError(
                    f"{self.method} down scale length for NUMA {numa_id} is "
                    f"{down_scale_len}, expected {self.hidden_size}."
                )

    # ========== LoRA ==========

    def init_lora_weights(
        self,
        gate_lora_a: torch.Tensor,
        gate_lora_b: torch.Tensor,
        up_lora_a: torch.Tensor,
        up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        grad_gate_lora_a: torch.Tensor,
        grad_gate_lora_b: torch.Tensor,
        grad_up_lora_a: torch.Tensor,
        grad_up_lora_b: torch.Tensor,
        grad_down_lora_a: torch.Tensor,
        grad_down_lora_b: torch.Tensor,
    ) -> None:
        expected_shapes = {
            "gate_lora_a": (self.num_experts, self.lora_rank, self.hidden_size),
            "gate_lora_b": (self.num_experts, self.moe_intermediate_size, self.lora_rank),
            "up_lora_a": (self.num_experts, self.lora_rank, self.hidden_size),
            "up_lora_b": (self.num_experts, self.moe_intermediate_size, self.lora_rank),
            "down_lora_a": (self.num_experts, self.lora_rank, self.moe_intermediate_size),
            "down_lora_b": (self.num_experts, self.hidden_size, self.lora_rank),
        }
        provided = {
            "gate_lora_a": gate_lora_a,
            "gate_lora_b": gate_lora_b,
            "up_lora_a": up_lora_a,
            "up_lora_b": up_lora_b,
            "down_lora_a": down_lora_a,
            "down_lora_b": down_lora_b,
        }
        for name, tensor in provided.items():
            expected = expected_shapes[name]
            if tensor.shape != expected:
                raise ValueError(f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}")
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"{name} must be a CPU tensor for {self.method} SFT, got {tensor.device}."
                )

        self.gate_lora_a = gate_lora_a.contiguous()
        self.gate_lora_b = gate_lora_b.contiguous()
        self.up_lora_a = up_lora_a.contiguous()
        self.up_lora_b = up_lora_b.contiguous()
        self.down_lora_a = down_lora_a.contiguous()
        self.down_lora_b = down_lora_b.contiguous()

        self.grad_gate_lora_a = grad_gate_lora_a.contiguous()
        self.grad_gate_lora_b = grad_gate_lora_b.contiguous()
        self.grad_up_lora_a = grad_up_lora_a.contiguous()
        self.grad_up_lora_b = grad_up_lora_b.contiguous()
        self.grad_down_lora_a = grad_down_lora_a.contiguous()
        self.grad_down_lora_b = grad_down_lora_b.contiguous()

        self._lora_initialized = True

        if self._weights_loaded and self.moe is not None:
            self.update_lora_weights()

    def update_lora_weights(self) -> None:
        if not self._weights_loaded:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")
        if self._is_skip_lora:
            return
        if not self._lora_initialized:
            if self.lora_rank <= 0:
                return  # Full mode without LoRA — no LoRA weights to update
            raise RuntimeError("LoRA weights not initialized. Call init_lora_weights() first.")

        # Weight pointer updates are load-time synchronous work. Calling the
        # direct binding avoids nesting an update task inside CPUInfer's queue
        # while SGLang is still in distributed model-loading barriers.
        self.moe.update_lora_weights(
            self.gate_lora_a.data_ptr(),
            self.gate_lora_b.data_ptr(),
            self.up_lora_a.data_ptr(),
            self.up_lora_b.data_ptr(),
            self.down_lora_a.data_ptr(),
            self.down_lora_b.data_ptr(),
        )

    def update_base_weights(self) -> None:
        """Sync updated base weight parameters back to C++ kernel after optimizer step."""
        if not self._weights_loaded:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")
        if not self._full_weight_grad:
            return  # No base weights to update in LoRA mode
        if self.gate_proj_buf is None:
            raise RuntimeError("Base weight buffers not initialized. Call init_full_weight_grad_buffers() first.")

        logger.info(f"Layer {self.layer_idx}: update_base_weights() - syncing updated weights to C++ kernel")

        # Preferred path: update config pointers on existing C++ object and re-quantize.
        # This avoids full C++ MOE object recreation (~0.6s/layer vs ~1.9s/layer).
        if hasattr(self.moe, "set_base_weight_pointers"):
            self.moe.set_base_weight_pointers(
                self.gate_proj_buf.data.data_ptr(),
                self.up_proj_buf.data.data_ptr(),
                self.down_proj_buf.data.data_ptr(),
            )
            self.cpu_infer.submit(self.moe.load_weights_task())
            self.cpu_infer.sync()
            logger.info(f"Layer {self.layer_idx}: update_base_weights() - re-quantized existing kernel")
            return

        # Fallback: full reload path (creates new C++ MOE object)
        # This is slower but works without C++ set_base_weight_pointers support.
        logger.warning(
            f"Layer {self.layer_idx}: set_base_weight_pointers not available, "
            f"falling back to full C++ MOE object recreation"
        )
        old_moe = getattr(self, "moe", None)
        if old_moe is not None:
            del old_moe

        self.gate_proj = self.gate_proj_buf.data
        self.up_proj = self.up_proj_buf.data
        self.down_proj = self.down_proj_buf.data
        physical_to_logical_map = torch.arange(self.num_experts, dtype=torch.int64, device="cpu")
        self._weights_loaded = False  # Allow re-load
        self.load_weights_from_tensors(
            gate_proj=self.gate_proj,
            up_proj=self.up_proj,
            down_proj=self.down_proj,
            physical_to_logical_map_cpu=physical_to_logical_map,
        )

    def save_backward_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        output_path: str,
    ) -> None:
        if not self._weights_loaded:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")
        gate_proj = gate_proj.contiguous()
        up_proj = up_proj.contiguous()
        down_proj = down_proj.contiguous()
        self.moe.prepare_and_save_bwd(
            gate_proj.data_ptr(),
            up_proj.data_ptr(),
            down_proj.data_ptr(),
            output_path,
        )
