from __future__ import annotations

import os

import numpy as np


def tensor_data_ptr(weight) -> int:
    if isinstance(weight, np.ndarray):
        return int(weight.ctypes.data)
    return int(weight.data_ptr())


def resolve_amx_weight_mode(wrapper) -> tuple[bool, bool]:
    use_iouring = getattr(wrapper, "io_backend", None) == "IOURING"
    use_mmap = wrapper.weight_strategy == "tiered" and not use_iouring

    if use_mmap and not wrapper.load_merged_weight:
        print(
            f"[AMXMoEWrapper] layer={wrapper.layer_idx} requested tiered mmap loading, "
            "but merged safetensors were not found; falling back to legacy resident loading"
        )
        use_mmap = False
        wrapper.weight_strategy = "legacy"

    if use_mmap and wrapper.cpu_save:
        print(
            f"[AMXMoEWrapper] layer={wrapper.layer_idx} requested tiered mmap loading during cpu_save, "
            "which requires resident buffers; falling back to legacy loading"
        )
        use_mmap = False
        wrapper.weight_strategy = "legacy"

    if use_iouring:
        if not wrapper.load_merged_weight:
            raise RuntimeError(
                f"[AMXMoEWrapper] layer={wrapper.layer_idx} io_uring requires merged AMX safetensors "
                f"under weight_path={wrapper.weight_path!r}"
            )
        if wrapper.cpu_save:
            raise RuntimeError(f"[AMXMoEWrapper] layer={wrapper.layer_idx} io_uring cannot be used during cpu_save")

    return use_iouring, use_mmap


def load_amx_iouring_file_slots(wrapper, base_key: str):
    from ..async_io_manager import get_async_readers

    direct_io_requested = os.environ.get("KT_IOURING_DIRECT", "1") not in ("0", "false", "False")
    file_slots = wrapper.safetensor_loader.load_experts_iouring(
        base_key,
        use_direct_io=direct_io_requested,
    )
    if direct_io_requested and not file_slots.get("direct_io", False):
        raise RuntimeError(
            f"[AMXMoEWrapper] layer={wrapper.layer_idx} requested KT_IOURING_DIRECT=1 "
            "but SafeTensorLoader did not return direct I/O slots"
        )
    print(
        "[AMXMoEWrapper] "
        f"layer={wrapper.layer_idx} backend=IOURING direct_io={file_slots.get('direct_io', False)} "
        "mmap_baseline=false "
        f"policy={wrapper.residency_policy} capacity={wrapper.max_resident_experts or wrapper.max_tier0_experts}"
    )

    wrapper.gate_file_slots = file_slots["gate"]
    wrapper.up_file_slots = file_slots["up"]
    wrapper.down_file_slots = file_slots["down"]
    wrapper.gate_scale_file_slots = file_slots["gate_scale"]
    wrapper.up_scale_file_slots = file_slots["up_scale"]
    wrapper.down_scale_file_slots = file_slots["down_scale"]
    reader_count = max(1, len(wrapper.gate_file_slots))
    wrapper.async_readers = get_async_readers(reader_count)
    wrapper.async_reader = wrapper.async_readers[0]
    return file_slots


def assign_amx_weight_views(wrapper, weights: dict) -> tuple[list, list, list, list, list, list]:
    wrapper.gate_weights = weights["gate"]
    wrapper.up_weights = weights["up"]
    wrapper.down_weights = weights["down"]
    wrapper.gate_scales = weights["gate_scale"]
    wrapper.up_scales = weights["up_scale"]
    wrapper.down_scales = weights["down_scale"]

    gate_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in wrapper.gate_weights]
    up_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in wrapper.up_weights]
    down_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in wrapper.down_weights]
    gate_scale_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in wrapper.gate_scales]
    up_scale_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in wrapper.up_scales]
    down_scale_ptrs = [[int(et.ctypes.data) for et in numa_array] for numa_array in wrapper.down_scales]
    return gate_ptrs, up_ptrs, down_ptrs, gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs


def apply_mesh_moe_config(wrapper, moe_config, *, use_iouring: bool, use_mmap: bool, file_slots=None) -> None:
    moe_config.use_mmap = use_mmap
    moe_config.max_tier0_experts = wrapper.max_tier0_experts
    moe_config.max_resident_experts = wrapper._mesh_slot_pool_capacity()
    if hasattr(moe_config, "mesh_prefill_layer_mode_enabled"):
        moe_config.mesh_prefill_layer_mode_enabled = wrapper._mesh_prefill_layer_mode_enabled()
    if hasattr(moe_config, "mesh_prefill_static_experts"):
        moe_config.mesh_prefill_static_experts = wrapper._mesh_prefill_static_resident_capacity()
    if hasattr(moe_config, "mesh_decode_resident_experts"):
        moe_config.mesh_decode_resident_experts = wrapper._mesh_config_resident_experts()
    moe_config.resident_cache_policy = wrapper.residency_policy
    if hasattr(moe_config, "enable_cache_stats"):
        moe_config.enable_cache_stats = wrapper.enable_cache_stats
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
        moe_config.mesh_prefetch_budget = 0
        moe_config.mesh_coldstart_prefill_enabled = False
        moe_config.mesh_coldstart_prefill_limit = 0
        if hasattr(moe_config, "mesh_memory_guard_enabled"):
            moe_config.mesh_memory_guard_enabled = use_iouring and os.environ.get("KT_MESH_MEMORY_GUARD", "1") not in (
                "0",
                "false",
                "False",
                "FALSE",
            )
            moe_config.mesh_memory_high_watermark = float(os.environ.get("KT_MESH_MEMORY_HIGH_WATERMARK", "0.95"))
            moe_config.mesh_memory_target_watermark = float(os.environ.get("KT_MESH_MEMORY_TARGET_WATERMARK", "0.90"))
            moe_config.mesh_memory_check_interval = int(os.environ.get("KT_MESH_MEMORY_CHECK_INTERVAL", "64"))
            moe_config.mesh_memory_max_demotes_per_check = int(
                os.environ.get("KT_MESH_MEMORY_MAX_DEMOTES_PER_CHECK", "8")
            )
    if hasattr(moe_config, "iouring_direct_io"):
        moe_config.iouring_direct_io = bool(file_slots.get("direct_io", False)) if use_iouring and file_slots else False
    if use_iouring:
        moe_config.use_mmap = False
        if not hasattr(moe_config, "set_iouring_file_slots_for_readers"):
            raise RuntimeError("io_uring TP reader binding is unavailable in kt_kernel_ext")
        moe_config.set_iouring_file_slots_for_readers(
            wrapper.gate_file_slots,
            wrapper.gate_scale_file_slots,
            wrapper.up_file_slots,
            wrapper.up_scale_file_slots,
            wrapper.down_file_slots,
            wrapper.down_scale_file_slots,
            wrapper.async_readers,
        )
        wrapper._async_reader_keepalive = wrapper.async_readers


def register_amx_mmap_regions(wrapper) -> None:
    if wrapper._provider is None:
        return

    from ..weight_provider import MmapWeightRegion

    wrapper._provider.clear_layer_regions(wrapper.layer_idx)

    for proj_name, weights, scales in (
        ("gate", wrapper.gate_weights, wrapper.gate_scales),
        ("up", wrapper.up_weights, wrapper.up_scales),
        ("down", wrapper.down_weights, wrapper.down_scales),
    ):
        for numa_idx, numa_weights in enumerate(weights):
            for expert_id, weight in enumerate(numa_weights):
                weight_region = MmapWeightRegion.__new__(MmapWeightRegion)
                weight_region.ptr = int(weight.ctypes.data)
                weight_region.n_bytes = int(weight.nbytes)
                weight_region._view = weight
                wrapper._provider.register_mmap_region(
                    wrapper.layer_idx, f"{proj_name}_weight", expert_id, weight_region
                )

                if scales is not None:
                    scale = scales[numa_idx][expert_id]
                    scale_region = MmapWeightRegion.__new__(MmapWeightRegion)
                    scale_region.ptr = int(scale.ctypes.data)
                    scale_region.n_bytes = int(scale.nbytes)
                    scale_region._view = scale
                    wrapper._provider.register_mmap_region(
                        wrapper.layer_idx, f"{proj_name}_scale", expert_id, scale_region
                    )


def register_bf16_mmap_regions(wrapper) -> None:
    if wrapper._provider is None:
        return

    from ..weight_provider import MmapWeightRegion

    wrapper._provider.clear_layer_regions(wrapper.layer_idx)

    for proj_name, weights in (
        ("gate", wrapper.gate_weights),
        ("up", wrapper.up_weights),
        ("down", wrapper.down_weights),
    ):
        for expert_id, weight in enumerate(weights):
            region = MmapWeightRegion.__new__(MmapWeightRegion)
            region.ptr = int(weight.ctypes.data)
            region.n_bytes = int(weight.nbytes)
            region._view = weight
            wrapper._provider.register_mmap_region(wrapper.layer_idx, proj_name, expert_id, region)
