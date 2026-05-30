from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch

BaseMoEWrapper = None
KExpertsCPUBuffer = None
_PIN_MEMORY = False


def _mesh_log_once(key, message: str) -> None:
    logged = getattr(BaseMoEWrapper, "_mesh_runtime_config_logged", None)
    if logged is None:
        print(message)
        return
    if key in logged:
        return
    print(message)
    logged.add(key)


def _mesh_score_transform_id(self) -> int:
    raw = os.environ.get("KT_MESH_SCORE_TRANSFORM", "softmax").strip().lower()
    if raw in ("none", "identity", "raw", "0"):
        return 0
    if raw in ("sigmoid", "2"):
        return 2
    return 1


def _configure_base_runtime(
    self,
    kt_kernel_ext,
    *,
    method: str,
    weight_strategy: str,
    max_tier0_experts,
    num_moe_layers,
    cpuinfer_threads: int,
    threadpool_count: int,
    numa_nodes,
) -> None:
    env_weight_strategy = os.environ.get("KT_WEIGHT_STRATEGY")
    if env_weight_strategy:
        weight_strategy = env_weight_strategy

    requested_weight_strategy = weight_strategy or "legacy"
    from ..weight_provider import normalize_residency_policy_name

    if requested_weight_strategy in {"auto", "tiered"}:
        self.weight_strategy = "legacy"
        _mesh_log_once(
            ("weight_strategy_disabled", method, requested_weight_strategy),
            "[KTMeshRuntime] adaptive weight_strategy is disabled; using legacy resident loading "
            f"(method={method}, requested={requested_weight_strategy})",
        )
    else:
        self.weight_strategy = requested_weight_strategy

    env_residency_policy = os.environ.get("KT_RESIDENCY_POLICY")
    self.residency_policy = normalize_residency_policy_name(env_residency_policy or "baseline")
    self.io_backend = os.environ.get("KT_IO_BACKEND", "IOURING").upper()
    verbose_runtime_config = os.environ.get("KT_MESH_VERBOSE", "0") in ("1", "true", "True", "TRUE")
    explicit_mesh_runtime = (
        env_weight_strategy is not None
        or env_residency_policy is not None
        or "KT_IO_BACKEND" in os.environ
        or requested_weight_strategy != "legacy"
    )
    if verbose_runtime_config or explicit_mesh_runtime:
        _mesh_log_once(
            (
                "runtime_config",
                method,
                env_weight_strategy,
                requested_weight_strategy,
                self.weight_strategy,
                self.residency_policy,
                self.io_backend,
            ),
            "[KTMeshRuntime] "
            f"method={method!r} requested_weight_strategy={requested_weight_strategy!r} "
            f"resolved_weight_strategy={self.weight_strategy!r} "
            f"residency_policy={self.residency_policy!r} io_backend={self.io_backend}",
        )

    env_enable_cache_stats = os.environ.get("KT_ENABLE_CACHE_STATS", "0")
    self.enable_cache_stats = env_enable_cache_stats in ("1", "true", "True", "TRUE")
    if self.enable_cache_stats:
        _mesh_log_once(("cache_stats", method), f"[KTCacheStats] Cache statistics collection enabled for method={method}")

    rolling_requested = os.environ.get("KT_MESH_PREFILL_ROLLING") not in (None, "", "0", "false", "False", "FALSE", "no", "No", "NO")
    if rolling_requested and self.io_backend == "IOURING":
        rolling_depth_env = os.environ.get("KT_MESH_PREFILL_ROLLING_DEPTH", "10")
        try:
            rolling_depth = max(1, int(rolling_depth_env))
        except ValueError:
            rolling_depth = 10
        _mesh_log_once(
            ("rlp_enabled", method, rolling_depth),
            f"[KTMeshRuntime] [RLP] enabled depth={rolling_depth} method={method!r} io_backend={self.io_backend}",
        )
    elif rolling_requested:
        _mesh_log_once(
            ("rlp_disabled_backend", method, self.io_backend),
            f"[KTMeshRuntime] [RLP] requested but io_backend={self.io_backend} (need IOURING) — falling back to default prefill",
        )

    if max_tier0_experts is not None:
        parsed_max_tier0 = int(max_tier0_experts)
        max_tier0_experts = None if parsed_max_tier0 <= 0 else parsed_max_tier0
    elif "KT_MAX_TIER0_EXPERTS" in os.environ:
        raw_max_tier0 = os.environ["KT_MAX_TIER0_EXPERTS"].strip().lower()
        if raw_max_tier0 not in ("", "auto", "none", "0", "-1"):
            parsed_max_tier0 = int(raw_max_tier0)
            max_tier0_experts = None if parsed_max_tier0 <= 0 else parsed_max_tier0
    self.max_tier0_experts = int(max_tier0_experts) if max_tier0_experts is not None else 0
    self.max_resident_experts = (
        int(os.environ["KT_MAX_RESIDENT_EXPERTS"]) if "KT_MAX_RESIDENT_EXPERTS" in os.environ else 0
    )

    if threadpool_count <= 0:
        raise ValueError(f"threadpool_count must be positive, got {threadpool_count}")
    if cpuinfer_threads < threadpool_count:
        raise ValueError(f"cpuinfer_threads ({cpuinfer_threads}) must be >= threadpool_count ({threadpool_count})")

    if numa_nodes is not None:
        available_numa_ids = [int(node) for node in numa_nodes]
        if len(available_numa_ids) == 0:
            raise ValueError("numa_nodes must not be empty when provided")
    else:
        available_numa_ids = BaseMoEWrapper._detect_available_numa_node_ids()
    if threadpool_count > len(available_numa_ids):
        raise ValueError(f"threadpool_count ({threadpool_count}) exceeds detected NUMA nodes {available_numa_ids}")
    if numa_nodes is not None and threadpool_count != len(available_numa_ids):
        raise ValueError(f"threadpool_count ({threadpool_count}) must match explicit numa_nodes {available_numa_ids}")

    subpool_numa_map = available_numa_ids[:threadpool_count]
    subpool_thread_count = [
        cpuinfer_threads // threadpool_count + (1 if i < cpuinfer_threads % threadpool_count else 0)
        for i in range(threadpool_count)
    ]
    runtime_signature = (tuple(subpool_numa_map), tuple(subpool_thread_count))

    if BaseMoEWrapper._cpu_infer_instance is None:
        worker_config = kt_kernel_ext.WorkerPoolConfig()
        worker_config.subpool_count = threadpool_count
        worker_config.subpool_numa_map = subpool_numa_map
        worker_config.subpool_thread_count = subpool_thread_count
        BaseMoEWrapper._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)
        BaseMoEWrapper._cpu_infer_signature = runtime_signature
    elif BaseMoEWrapper._cpu_infer_signature != runtime_signature:
        if BaseMoEWrapper._active_wrapper_count > 0:
            raise RuntimeError(
                "CPUInfer is already initialized with a different NUMA/thread layout. "
                "Destroy existing MoE wrappers before creating a new runtime topology."
            )
        BaseMoEWrapper.reset_runtime_state(force=True)
        worker_config = kt_kernel_ext.WorkerPoolConfig()
        worker_config.subpool_count = threadpool_count
        worker_config.subpool_numa_map = subpool_numa_map
        worker_config.subpool_thread_count = subpool_thread_count
        BaseMoEWrapper._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)
        BaseMoEWrapper._cpu_infer_signature = runtime_signature

    self.cpu_infer = BaseMoEWrapper._cpu_infer_instance


def _mesh_global_resident_capacity(self) -> int:
    if int(self.max_resident_experts) > 0:
        return int(self.max_resident_experts)
    return int(self.max_tier0_experts)


def _mesh_prefill_layer_mode_enabled(self) -> bool:
    return self.io_backend == "IOURING" and self._env_flag("KT_MESH_PREFILL_LAYER_MODE", False)


def _mesh_prefill_rolling_enabled(self) -> bool:
    # Rolling Layer Prefetch (RLP): non-default opt-in. Requires IOURING because
    # the strategy depends on cross-layer async submission.
    return self.io_backend == "IOURING" and self._env_flag("KT_MESH_PREFILL_ROLLING", False)


def _mesh_prefill_rolling_depth(self) -> int:
    if not self._mesh_prefill_rolling_enabled():
        return 0
    return max(1, self._env_int("KT_MESH_PREFILL_ROLLING_DEPTH", 10))


def _mesh_prefill_full_layer_count(self) -> int:
    if not self._mesh_prefill_layer_mode_enabled():
        return 0
    # Explicit override via KT_MESH_PREFILL_LAYER_WINDOW.
    #   -1 (default): compute from cache config (sliding window behavior)
    #    0:           disable sliding — keep ALL layers' prefill state across
    #                 SGLang chunks. Memory cost = pool capacity, not 40 ×
    #                 cap. Use when KT_MESH_GLOBAL_POOL_CAPACITY is sized
    #                 for the full set.
    #    N > 0:       force window = N (clamped to total_layers).
    explicit = self._env_int("KT_MESH_PREFILL_LAYER_WINDOW", -1)
    total_layers = self._mesh_total_moe_layers()
    if explicit == 0:
        return max(1, int(total_layers))
    if explicit > 0:
        return max(1, min(int(total_layers), int(explicit)))
    configured = self._mesh_global_resident_capacity()
    if configured <= 0 or self.num_experts <= 0 or total_layers <= 0:
        return 0
    full_layers = (int(total_layers) * int(configured)) // int(self.num_experts)
    if configured >= self.num_experts:
        full_layers = total_layers
    return max(1, min(int(total_layers), int(full_layers)))


def _mesh_slot_pool_capacity(self) -> int:
    if self._mesh_prefill_layer_mode_enabled() and self._mesh_global_resident_capacity() > 0:
        return int(self.num_experts)
    return self._mesh_config_resident_experts()


def _mesh_prefill_static_resident_capacity(self) -> int:
    raw = os.environ.get("KT_MESH_PREFILL_STATIC_EXPERTS")
    configured = self._mesh_config_resident_experts()
    cpu_expert_count = self._mesh_current_cpu_expert_count()
    if raw is not None and str(raw).strip():
        try:
            configured = int(str(raw).strip())
        except ValueError:
            configured = 0
    if configured <= 0:
        return 0
    return min(int(configured), int(cpu_expert_count))


def _mesh_config_resident_experts(self) -> int:
    """Return the per-layer resident cap written into GeneralMOEConfig.

    Layer 0-4 get an independent MESH slot-pool cap. The default is all
    CPU-managed experts in that layer, not a hard-coded model expert count.
    """
    if self.io_backend != "IOURING" or int(self.layer_idx) >= 5:
        return int(self.max_resident_experts)

    cpu_expert_count = self._mesh_current_cpu_expert_count()
    raw = os.environ.get("KT_MESH_EARLY_LAYER_EXPERTS", "full").strip().lower()
    global_capacity = self._mesh_global_resident_capacity()
    mode = raw or "full"

    if mode in ("global", "inherit", "default"):
        return int(self.max_resident_experts)
    if mode in ("full", "all", "auto", "max"):
        capacity = cpu_expert_count
    else:
        try:
            requested = int(mode)
        except ValueError:
            requested = cpu_expert_count
            mode = "full"
        if requested <= 0:
            capacity = cpu_expert_count
            mode = "full"
        else:
            minimum = min(int(self.num_experts_per_tok), cpu_expert_count)
            capacity = min(cpu_expert_count, max(requested, minimum))

    log_key = (int(self.layer_idx), raw, int(cpu_expert_count), int(global_capacity), int(capacity))
    if log_key not in BaseMoEWrapper._mesh_early_capacity_logged:
        print(
            "[MESHEarlyLayerCapacity] "
            f"layer={self.layer_idx} env={raw!r} mode={mode} "
            f"cpu_experts={cpu_expert_count} global_capacity={global_capacity} "
            f"effective_capacity={capacity}"
        )
        BaseMoEWrapper._mesh_early_capacity_logged.add(log_key)
    return int(capacity)


def _mesh_resident_capacity(self) -> int:
    configured = self._mesh_config_resident_experts()
    if configured > 0:
        return configured
    return int(self.max_tier0_experts)


def _mesh_current_cpu_expert_count(self, cols: Optional[int] = None) -> int:
    width = int(self.num_experts if cols is None else min(int(cols), int(self.num_experts)))
    if width <= 0:
        return 0
    src = getattr(self, "gpu_experts_mask", None)
    if src is None or src.numel() <= 0:
        return width
    n = min(width, int(src.numel()))
    gpu_count = int(src[:n].sum().item()) if n > 0 else 0
    return max(0, width - gpu_count)


def _mesh_cpu_expert_mask(self, cols: int) -> torch.Tensor:
    """Return a CPU bool mask where True means MESH may manage this expert."""
    width = max(0, int(cols))
    mask = torch.zeros(width, dtype=torch.bool, device="cpu")
    if width <= 0:
        return mask
    valid = min(width, int(self.num_experts))
    if valid <= 0:
        return mask

    src = getattr(self, "gpu_experts_mask", None)
    if src is None or src.numel() <= 0:
        mask[:valid] = True
        return mask

    n = min(valid, int(src.numel()))
    if n > 0:
        mask[:n].copy_(~src[:n].to(dtype=torch.bool), non_blocking=False)
    if valid > n:
        mask[n:valid] = True
    return mask


def _mesh_full_gate_observation_enabled(self) -> bool:
    if os.environ.get("KT_MESH_FULL_GATE", "1") in ("0", "false", "False", "FALSE"):
        return False
    if os.environ.get("KT_MESH_LOOKAHEAD", "1") in ("0", "false", "False", "FALSE"):
        return False
    try:
        if float(os.environ.get("KT_MESH_LOOKAHEAD_WEIGHT", "1.0")) <= 0.0:
            return False
    except ValueError:
        pass
    current_gpu_experts = int(self.gpu_experts_mask.sum().item())
    cpu_expert_count = self._mesh_current_cpu_expert_count()
    resident_capacity = self._mesh_resident_capacity()
    if cpu_expert_count <= 0 or (resident_capacity > 0 and resident_capacity >= cpu_expert_count):
        log_key = (int(self.num_experts), current_gpu_experts, int(resident_capacity))
        if log_key not in BaseMoEWrapper._full_gate_skip_logged:
            print(
                "[MESHFullGate] skip full-router Heat observe: "
                f"resident_capacity={resident_capacity} cpu_experts={cpu_expert_count} "
                f"num_experts={self.num_experts} gpu_experts={current_gpu_experts}"
            )
            BaseMoEWrapper._full_gate_skip_logged.add(log_key)
        return False
    return True


def _mesh_full_gate_batched_enabled(self) -> bool:
    return os.environ.get("KT_MESH_FULL_GATE_BATCHED", "1") not in ("0", "false", "False", "FALSE")


def _cuda_graph_capture_active() -> bool:
    """Return True while SGLang is recording a CUDA graph warmup forward."""
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_current_stream_capturing", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _discard_full_gate_batch_slot(state: dict, slot: int) -> None:
    try:
        state["seen"][slot].clear()
        state["gpu_seen"][slot].clear()
        state["decode_seen"][slot] = False
        state["transforms"][slot].clear()
        state["keepalive"][slot] = None
        state["write_slot"] = (slot + 1) % len(state["seen"])
    except Exception:
        return


def _mesh_total_moe_layers(self) -> int:
    registered_layers = max(BaseMoEWrapper._wrappers_by_layer.keys()) + 1 if BaseMoEWrapper._wrappers_by_layer else 0
    if self.num_moe_layers is not None and self.num_moe_layers > 0:
        return max(self.num_moe_layers, registered_layers, self.layer_idx + 1)
    if registered_layers > 0:
        return max(registered_layers, self.layer_idx + 1)
    return self.layer_idx + 1


def _mesh_last_registered_layer_idx(self) -> int:
    if BaseMoEWrapper._wrappers_by_layer:
        return max(BaseMoEWrapper._wrappers_by_layer.keys())
    return self.layer_idx


def _mesh_last_full_gate_layer_idx(self) -> int:
    last_enabled = -1
    for layer_idx, wrapper in BaseMoEWrapper._wrappers_by_layer.items():
        if wrapper._mesh_full_gate_observation_enabled():
            last_enabled = max(last_enabled, layer_idx)
    return last_enabled if last_enabled >= 0 else self._mesh_last_registered_layer_idx()


def _prepare_router_score_vector(self, router_scores: torch.Tensor) -> Tuple[Optional[torch.Tensor], int, int]:
    scores = router_scores.detach()
    if scores.dim() == 1:
        scores = scores.view(1, -1)
    elif scores.dim() > 2:
        scores = scores.reshape(-1, scores.shape[-1])
    rows = int(scores.shape[0])
    cols = int(scores.shape[1])
    if rows <= 0 or cols <= 0:
        return None, 0, 0

    max_elements = int(os.environ.get("KT_MESH_FULL_GATE_MAX_ELEMENTS", "65536"))
    if max_elements > 0 and rows * cols > max_elements:
        return None, 0, 0

    if scores.dtype != torch.float32:
        scores = scores.float()
    if not scores.is_contiguous():
        scores = scores.contiguous()

    score_transform = self._mesh_score_transform_id()
    if rows == 1:
        return scores.view(-1), cols, score_transform

    if score_transform == 1:
        vector = torch.softmax(scores, dim=-1).amax(dim=0)
        score_transform = 0
    elif score_transform == 2:
        vector = torch.sigmoid(scores).amax(dim=0)
        score_transform = 0
    else:
        vector = scores.amax(dim=0)
    if not vector.is_contiguous():
        vector = vector.contiguous()
    return vector, cols, score_transform


def _full_gate_batch_key(cls, device: torch.device, total_layers: int, cols: int) -> Tuple[str, int, int, int]:
    return (device.type, -1 if device.index is None else int(device.index), int(total_layers), int(cols))


def _get_full_gate_batch_state(self, vector: torch.Tensor, total_layers: int, cols: int) -> dict:
    key = BaseMoEWrapper._full_gate_batch_key(vector.device, total_layers, cols)
    state = BaseMoEWrapper._full_gate_batch_states.get(key)
    depth = max(4, KExpertsCPUBuffer.buffer_depth)
    if state is None:
        cpu = torch.empty((depth, total_layers, cols), dtype=torch.float32, device="cpu", pin_memory=_PIN_MEMORY)
        gpu = None
        if vector.device.type == "cuda":
            gpu = torch.empty((depth, total_layers, cols), dtype=torch.float32, device=vector.device)
        state = {
            "cpu": cpu,
            "gpu": gpu,
            "write_slot": 0,
            "seen": [set() for _ in range(depth)],
            "gpu_seen": [set() for _ in range(depth)],
            "decode_seen": [False for _ in range(depth)],
            "transforms": [{} for _ in range(depth)],
            "keepalive": [None for _ in range(depth)],
        }
        BaseMoEWrapper._full_gate_batch_states[key] = state
    elif vector.device.type == "cuda" and state.get("gpu") is None:
        state["gpu"] = torch.empty((depth, total_layers, cols), dtype=torch.float32, device=vector.device)
    if "decode_seen" not in state:
        state["decode_seen"] = [False for _ in range(len(state["seen"]))]
    return state


def _mesh_full_gate_mask_batch(self, seen: List[int], cols: int) -> torch.Tensor:
    masks = torch.empty((len(seen), cols), dtype=torch.uint8, device="cpu", pin_memory=_PIN_MEMORY)
    for row, layer_idx in enumerate(seen):
        masks[row].zero_()
        wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
        if wrapper is None:
            continue
        src = getattr(wrapper, "gpu_experts_mask", None)
        if src is None:
            continue
        n = min(cols, int(src.numel()))
        if n > 0:
            masks[row, :n].copy_(src[:n].to(dtype=torch.uint8), non_blocking=False)
    return masks


def _mesh_full_gate_batch_owner(self, seen: List[int]):
    if self.layer_idx in seen and self.moe is not None and hasattr(self.moe, "observe_router_scores_batch_task"):
        return self
    for layer_idx in reversed(seen):
        wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
        if wrapper is not None and wrapper.moe is not None and hasattr(wrapper.moe, "observe_router_scores_batch_task"):
            return wrapper
    return None


def _mesh_bootstrap_prefetch_enabled(self) -> bool:
    return (
        self.io_backend == "IOURING"
        and self._env_flag("KT_MESH_BOOTSTRAP_PREFETCH", True)
        and self._mesh_resident_capacity() > 0
    )


def _mesh_bootstrap_prefetch_budget(self, cpu_expert_count: int) -> int:
    if cpu_expert_count <= 0:
        return 0
    explicit = self._env_int("KT_MESH_BOOTSTRAP_PREFETCH_LIMIT", 0)
    capacity = self._mesh_resident_capacity()
    if capacity <= 0:
        return 0
    if explicit > 0:
        return min(explicit, cpu_expert_count)
    return min(capacity, cpu_expert_count)


def _maybe_submit_mesh_bootstrap_from_full_gate_batch(
    self,
    state: dict,
    slot: int,
    seen: List[int],
    cols: int,
    cuda_stream,
    *,
    needs_cuda_sync: bool,
) -> None:
    if BaseMoEWrapper._mesh_bootstrap_done:
        return
    if not state.get("decode_seen", [False])[slot]:
        return
    if not self._env_flag("KT_MESH_BOOTSTRAP_PREFETCH", True):
        BaseMoEWrapper._mesh_bootstrap_done = True
        return

    if needs_cuda_sync and torch.cuda.is_available():
        if cuda_stream is not None:
            self._sync_external_cuda_stream(cuda_stream)
        else:
            torch.cuda.current_stream().synchronize()

    submitted_layers = 0
    candidate_total = 0
    for layer_idx in seen:
        wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
        if wrapper is None or wrapper.moe is None:
            continue
        if not wrapper._mesh_bootstrap_prefetch_enabled():
            continue

        cpu_mask = wrapper._mesh_cpu_expert_mask(cols)
        cpu_expert_count = int(cpu_mask.sum().item())
        budget = wrapper._mesh_bootstrap_prefetch_budget(cpu_expert_count)
        if budget <= 0:
            continue

        scores = state["cpu"][slot, layer_idx, :cols].detach().clone()
        scores[~cpu_mask] = float("-inf")
        candidate_scores, candidate_ids = torch.topk(scores, k=budget, largest=True, sorted=True)
        finite = torch.isfinite(candidate_scores)
        if not bool(finite.any().item()):
            continue
        candidate_ids = candidate_ids[finite].to(torch.long).contiguous()
        candidate_count = int(candidate_ids.numel())
        if candidate_count <= 0:
            continue

        protect_count = min(int(wrapper.num_experts_per_tok), candidate_count)
        protect_ids = candidate_ids[:protect_count].contiguous() if protect_count > 0 else None
        if wrapper._submit_iouring_prefetch(
            candidate_ids,
            candidate_count,
            protect_ids_cpu=protect_ids,
            protect_count=protect_count,
            max_to_submit=candidate_count,
            cuda_stream=cuda_stream,
            prefetch_kind=1,
        ):
            submitted_layers += 1
            candidate_total += candidate_count

    BaseMoEWrapper._mesh_bootstrap_done = True
    if submitted_layers > 0 and BaseMoEWrapper._mesh_bootstrap_log_count < 3:
        print(
            "[MESHBootstrap] submitted first-token CPU expert warm fill: "
            f"layers={submitted_layers} candidates={candidate_total} cols={cols}"
        )
        BaseMoEWrapper._mesh_bootstrap_log_count += 1


def _flush_full_gate_batch(self, state: dict, slot: int, total_layers: int, cols: int, cuda_stream) -> None:
    seen = sorted(state["seen"][slot])
    if not seen:
        return
    if self._cuda_graph_capture_active():
        self._discard_full_gate_batch_slot(state, slot)
        return

    gpu_seen = state["gpu_seen"][slot]
    needs_cuda_sync = bool(gpu_seen)
    if state.get("gpu") is not None and gpu_seen:
        if len(gpu_seen) == len(seen):
            state["cpu"][slot].copy_(state["gpu"][slot], non_blocking=True)
        else:
            for layer_idx in sorted(gpu_seen):
                state["cpu"][slot, layer_idx].copy_(state["gpu"][slot, layer_idx], non_blocking=True)

    batch_owner = self._mesh_full_gate_batch_owner(seen)
    if batch_owner is not None:
        batch_task = getattr(batch_owner.moe, "observe_router_scores_batch_task", None)
        batch_direct = getattr(batch_owner.moe, "observe_router_scores_batch", None)
        if batch_task is not None or batch_direct is not None:
            layer_indices = torch.tensor(seen, dtype=torch.int32, device="cpu")
            score_transforms = torch.tensor(
                [int(state["transforms"][slot].get(layer_idx, self._mesh_score_transform_id())) for layer_idx in seen],
                dtype=torch.int32,
                device="cpu",
            )
            gpu_masks = self._mesh_full_gate_mask_batch(seen, cols)
            state["keepalive"][slot] = (layer_indices, score_transforms, gpu_masks)
            scores_ptr = int(state["cpu"][slot].data_ptr())
            if batch_task is not None:
                task = batch_task(
                    scores_ptr,
                    cols,
                    cols,
                    int(layer_indices.data_ptr()),
                    int(score_transforms.data_ptr()),
                    int(gpu_masks.data_ptr()),
                    len(seen),
                )
                batch_owner._submit_cpuinfer_task(task, cuda_stream)
            else:
                batch_owner._sync_external_cuda_stream(cuda_stream or torch.cuda.current_stream().cuda_stream)
                batch_direct(
                    scores_ptr,
                    cols,
                    cols,
                    int(layer_indices.data_ptr()),
                    int(score_transforms.data_ptr()),
                    int(gpu_masks.data_ptr()),
                    len(seen),
                )
            state["seen"][slot].clear()
            state["gpu_seen"][slot].clear()
            self._maybe_submit_mesh_bootstrap_from_full_gate_batch(
                state,
                slot,
                seen,
                cols,
                cuda_stream,
                needs_cuda_sync=needs_cuda_sync,
            )
            state["decode_seen"][slot] = False
            state["transforms"][slot].clear()
            state["write_slot"] = (slot + 1) % len(state["seen"])
            return

    for layer_idx in seen:
        wrapper = BaseMoEWrapper._wrappers_by_layer.get(layer_idx)
        if wrapper is None or wrapper.moe is None:
            continue
        observe_task = getattr(wrapper.moe, "observe_router_scores_task", None)
        observe_direct = getattr(wrapper.moe, "observe_router_scores", None)
        score_transform = int(state["transforms"][slot].get(layer_idx, self._mesh_score_transform_id()))
        scores_ptr = int(state["cpu"][slot, layer_idx].data_ptr())
        if observe_task is not None:
            task = observe_task(scores_ptr, 1, cols, score_transform)
            wrapper._submit_cpuinfer_task(task, cuda_stream)
        elif observe_direct is not None:
            wrapper._sync_external_cuda_stream(cuda_stream or torch.cuda.current_stream().cuda_stream)
            observe_direct(scores_ptr, 1, cols, score_transform)

    state["seen"][slot].clear()
    state["gpu_seen"][slot].clear()
    self._maybe_submit_mesh_bootstrap_from_full_gate_batch(
        state,
        slot,
        seen,
        cols,
        cuda_stream,
        needs_cuda_sync=needs_cuda_sync,
    )
    state["decode_seen"][slot] = False
    state["transforms"][slot].clear()
    state["write_slot"] = (slot + 1) % len(state["seen"])


def _record_router_scores_for_token_batch(
    self,
    router_scores: torch.Tensor,
    cuda_stream,
    flush_on_last: bool = False,
    is_decode_token: bool = False,
) -> bool:
    if self._cuda_graph_capture_active():
        return True

    vector, cols, score_transform = self._prepare_router_score_vector(router_scores)
    if vector is None or cols <= 0:
        return True

    total_layers = max(self._mesh_total_moe_layers(), self.layer_idx + 1)
    state = self._get_full_gate_batch_state(vector, total_layers, cols)
    slot = int(state["write_slot"])
    if self.layer_idx >= total_layers:
        return True

    if vector.device.type == "cuda":
        state["gpu"][slot, self.layer_idx].copy_(vector, non_blocking=True)
        state["gpu_seen"][slot].add(self.layer_idx)
    else:
        state["cpu"][slot, self.layer_idx].copy_(vector, non_blocking=True)
    state["seen"][slot].add(self.layer_idx)
    if is_decode_token:
        state["decode_seen"][slot] = True
    state["transforms"][slot][self.layer_idx] = int(score_transform)

    if self.layer_idx == self._mesh_last_full_gate_layer_idx():
        if flush_on_last:
            self._flush_full_gate_batch(state, slot, total_layers, cols, cuda_stream)
        else:
            self._pending_full_gate_flush = (state, slot, total_layers, cols)
    return True


def _flush_pending_full_gate_batch(self, cuda_stream) -> None:
    pending = getattr(self, "_pending_full_gate_flush", None)
    if pending is None:
        return
    self._pending_full_gate_flush = None
    state, slot, total_layers, cols = pending
    if self._cuda_graph_capture_active():
        self._discard_full_gate_batch_slot(state, slot)
        return
    self._flush_full_gate_batch(state, slot, total_layers, cols, cuda_stream)


def observe_router_scores(self, router_scores: Optional[torch.Tensor], cuda_stream=None) -> None:
    """Feed the full per-token router score vector into the MESH Heat registry.

    The AMX MoE compute API only needs top-k ids/weights. MESH eviction can
    use the richer all-expert router vector when the caller provides it
    through this side channel.
    """
    if router_scores is None or not torch.is_tensor(router_scores):
        return
    if not self._mesh_full_gate_observation_enabled():
        return
    if self._cuda_graph_capture_active():
        return
    if self._mesh_full_gate_batched_enabled():
        self._record_router_scores_for_token_batch(router_scores, cuda_stream, flush_on_last=True)
        return
    observe_task = getattr(self.moe, "observe_router_scores_task", None)
    observe_direct = getattr(self.moe, "observe_router_scores", None)
    if observe_task is None and observe_direct is None:
        return

    scores = router_scores.detach()
    if scores.dim() == 1:
        scores = scores.view(1, -1)
    elif scores.dim() > 2:
        scores = scores.reshape(-1, scores.shape[-1])
    rows = int(scores.shape[0])
    cols = int(scores.shape[1])
    if rows <= 0 or cols <= 0:
        return

    max_elements = int(os.environ.get("KT_MESH_FULL_GATE_MAX_ELEMENTS", "65536"))
    if max_elements > 0 and rows * cols > max_elements:
        return

    if scores.dtype != torch.float32:
        scores = scores.float()
    if not scores.is_contiguous():
        scores = scores.contiguous()

    score_transform = self._mesh_score_transform_id()
    if scores.device.type == "cuda":
        flat = scores.view(-1)
        needed = flat.numel()
        buf = getattr(self, "_router_scores_cpu", None)
        if buf is None or buf.numel() < needed:
            buf = torch.empty(needed, dtype=torch.float32, device="cpu", pin_memory=_PIN_MEMORY)
            self._router_scores_cpu = buf
        dst = buf[:needed]
        dst.copy_(flat, non_blocking=True)
        if observe_task is not None:
            task = observe_task(dst.data_ptr(), rows, cols, score_transform)
            self._submit_cpuinfer_task(task, cuda_stream)
        else:
            self._sync_external_cuda_stream(cuda_stream or torch.cuda.current_stream(scores.device).cuda_stream)
            observe_direct(dst.data_ptr(), rows, cols, score_transform)
        return

    scores_cpu = scores.cpu() if scores.device.type != "cpu" else scores
    observe = observe_direct
    if observe is not None:
        observe(scores_cpu.data_ptr(), rows, cols, score_transform)


def _prepare_router_scores_for_forward(
    self,
    router_scores: Optional[torch.Tensor],
    current_slot: int,
    cuda_stream=None,
    is_decode_token: bool = False,
) -> Tuple[int, int, int, int]:
    """Prepare full router scores for piggyback Heat update in forward_task."""
    if router_scores is None or not torch.is_tensor(router_scores):
        return 0, 0, 0, 0
    if not self._mesh_full_gate_observation_enabled():
        return 0, 0, 0, 0
    if self._cuda_graph_capture_active():
        return 0, 0, 0, 0
    if self._mesh_full_gate_batched_enabled():
        self._record_router_scores_for_token_batch(
            router_scores,
            cuda_stream,
            flush_on_last=False,
            is_decode_token=is_decode_token,
        )
        return 0, 0, 0, 0

    scores = router_scores.detach()
    if scores.dim() == 1:
        scores = scores.view(1, -1)
    elif scores.dim() > 2:
        scores = scores.reshape(-1, scores.shape[-1])
    rows = int(scores.shape[0])
    cols = int(scores.shape[1])
    if rows <= 0 or cols <= 0:
        return 0, 0, 0, 0

    max_elements = int(os.environ.get("KT_MESH_FULL_GATE_MAX_ELEMENTS", "65536"))
    if max_elements > 0 and rows * cols > max_elements:
        return 0, 0, 0, 0

    if scores.dtype != torch.float32:
        scores = scores.float()
    if not scores.is_contiguous():
        scores = scores.contiguous()

    score_transform = self._mesh_score_transform_id()
    if scores.device.type == "cuda":
        flat = scores.view(-1)
        needed = flat.numel()
        buffers = getattr(self, "_router_scores_cpu_slots", None)
        if buffers is None:
            buffers = {}
            self._router_scores_cpu_slots = buffers
        buf = buffers.get(current_slot)
        if buf is None or buf.numel() < needed:
            buf = torch.empty(needed, dtype=torch.float32, device="cpu", pin_memory=_PIN_MEMORY)
            buffers[current_slot] = buf
        dst = buf[:needed]
        dst.copy_(flat, non_blocking=True)
        return int(dst.data_ptr()), rows, cols, score_transform

    scores_cpu = scores.cpu() if scores.device.type != "cpu" else scores
    self._router_scores_cpu_direct = scores_cpu
    return int(scores_cpu.data_ptr()), rows, cols, score_transform


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False", "FALSE", "no", "No", "NO")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _submit_iouring_prefetch(
    self,
    expert_ids_cpu: torch.Tensor,
    count: int,
    *,
    protect_ids_cpu: Optional[torch.Tensor] = None,
    protect_count: int = 0,
    max_to_submit: int = 0,
    cuda_stream=None,
    prefetch_kind: int = 0,
) -> bool:
    if self.io_backend != "IOURING" or self.moe is None or count <= 0:
        return False
    task_factory = getattr(self.moe, "prefetch_experts_task", None)
    if task_factory is None:
        return False

    expert_ids_cpu = expert_ids_cpu[:count]
    if expert_ids_cpu.dtype != torch.long:
        expert_ids_cpu = expert_ids_cpu.to(torch.long)
    if not expert_ids_cpu.is_contiguous():
        expert_ids_cpu = expert_ids_cpu.contiguous()

    protect_ptr = 0
    if protect_ids_cpu is not None and protect_count > 0:
        protect_ids_cpu = protect_ids_cpu[:protect_count]
        if protect_ids_cpu.dtype != torch.long:
            protect_ids_cpu = protect_ids_cpu.to(torch.long)
        if not protect_ids_cpu.is_contiguous():
            protect_ids_cpu = protect_ids_cpu.contiguous()
        self._mesh_prefetch_protect_keepalive = protect_ids_cpu
        protect_ptr = int(protect_ids_cpu.data_ptr())

    self._mesh_prefetch_ids_keepalive = expert_ids_cpu
    try:
        task = task_factory(
            int(expert_ids_cpu.data_ptr()),
            int(count),
            protect_ptr,
            int(protect_count),
            int(max_to_submit),
            int(prefetch_kind),
        )
    except TypeError:
        task = task_factory(
            int(expert_ids_cpu.data_ptr()),
            int(count),
            protect_ptr,
            int(protect_count),
            int(max_to_submit),
        )
    self._submit_cpuinfer_task(task, cuda_stream)
    return True


def _submit_mesh_noarg_task(self, task_name: str) -> bool:
    if self.io_backend != "IOURING" or self.moe is None:
        return False
    task_factory = getattr(self.moe, task_name, None)
    if task_factory is None:
        return False
    self._submit_cpuinfer_task(task_factory(), None)
    return True


def _mesh_before_submit_forward(self, qlen: int) -> None:
    self._maybe_mesh_prepare_prefill_layer_window(qlen)
    self._maybe_mesh_transition_to_decode_cache(qlen)


def _mesh_prefetch_previous_topk(self) -> None:
    return


def _mesh_select_forward_experts(
    self,
    topk_ids_long: torch.Tensor,
    topk_weights: torch.Tensor,
    immediate_experts_ids_cpu: torch.Tensor,
    deferred_experts_ids_cpu: torch.Tensor,
    deferred_output_cpu: torch.Tensor,
    cuda_stream,
    *,
    is_decode_token: bool,
):
    if self.max_deferred_experts_per_token <= 0:
        return topk_ids_long, None, False

    split_task_factory = getattr(self.moe, "split_deferred_experts_task", None)
    state_defer_enabled = self.io_backend == "IOURING" and self._env_flag("KT_MESH_STATE_DEFER", True)
    allow_prefill_state_defer = self._env_flag("KT_MESH_STATE_DEFER_PREFILL", False)
    if split_task_factory is not None and state_defer_enabled and (is_decode_token or allow_prefill_state_defer):
        immediate_experts_ids_cpu.copy_(topk_ids_long, non_blocking=True)
        deferred_experts_ids_cpu.fill_(-1)
        deferred_output_cpu.zero_()
        task = split_task_factory(
            int(immediate_experts_ids_cpu.data_ptr()),
            int(immediate_experts_ids_cpu.data_ptr()),
            int(deferred_experts_ids_cpu.data_ptr()),
            int(immediate_experts_ids_cpu.numel()),
            int(self.num_experts_per_tok),
            int(self.max_deferred_experts_per_token),
        )
        self._submit_cpuinfer_task(task, cuda_stream)
        return immediate_experts_ids_cpu, deferred_experts_ids_cpu, True

    if not is_decode_token:
        return topk_ids_long, None, False

    protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token
    immediate_ids, deferred_ids = self.select_deferred_experts(topk_ids_long, topk_weights, protected_k)
    return immediate_ids, deferred_ids, False


def _mesh_prefetch_deferred_ids(
    self,
    deferred_experts_ids_cpu: torch.Tensor,
    immediate_experts_ids_cpu: torch.Tensor,
    cuda_stream,
) -> None:
    if not self._env_flag("KT_MESH_DEFER_PREFETCH", True):
        return
    defer_prefetch_limit = self._env_int(
        "KT_MESH_DEFER_PREFETCH_LIMIT",
        max(1, int(self.max_deferred_experts_per_token)),
    )
    self._submit_iouring_prefetch(
        deferred_experts_ids_cpu,
        int(deferred_experts_ids_cpu.numel()),
        protect_ids_cpu=immediate_experts_ids_cpu,
        protect_count=int(immediate_experts_ids_cpu.numel()),
        max_to_submit=defer_prefetch_limit,
        cuda_stream=cuda_stream,
    )


def _mesh_after_sync_forward(
    self,
    flat_hidden_states: torch.Tensor,
    immediate_experts_ids_cpu: torch.Tensor,
    weights_cpu: torch.Tensor,
    output_cpu: torch.Tensor,
    topk_ids,
    cuda_stream,
) -> None:
    self._debug_log_moe_once(flat_hidden_states, immediate_experts_ids_cpu, weights_cpu, output_cpu)
    if self._mesh_full_gate_batched_enabled():
        self._flush_pending_full_gate_batch(cuda_stream)

    if topk_ids is not None:
        BaseMoEWrapper._prev_topk_ids_by_layer[self.layer_idx] = topk_ids.detach().cpu().numpy()


def _mesh_submit_forward_impl(
    self,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    cuda_stream,
    router_scores: Optional[torch.Tensor] = None,
) -> None:
    flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    qlen = int(flat_hidden_states.shape[0])
    self._mesh_before_submit_forward(qlen)

    (
        input_tensor_cpu,
        immediate_experts_ids_cpu,
        deferred_experts_ids_cpu,
        weights_cpu,
        output_cpu,
        bsz_tensor_cpu,
        _output_gpu,
    ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

    current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
    next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth
    bsz_slot_tensor = bsz_tensor_cpu[current_slot]

    topk_ids_long = topk_ids.to(torch.long)
    self._mesh_prefetch_previous_topk()
    immediate_ids, deferred_ids, state_defer_used = self._mesh_select_forward_experts(
        topk_ids_long,
        topk_weights,
        immediate_experts_ids_cpu[current_slot],
        deferred_experts_ids_cpu[current_slot],
        output_cpu[next_slot],
        cuda_stream,
        is_decode_token=qlen == 1,
    )

    input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
    weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
    if not state_defer_used:
        immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)
    if deferred_ids is not None and not state_defer_used:
        deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)
        self._mesh_prefetch_deferred_ids(
            deferred_experts_ids_cpu[current_slot],
            immediate_experts_ids_cpu[current_slot],
            cuda_stream,
        )

    router_scores_ptr, score_rows, score_cols, score_transform = self._prepare_router_scores_for_forward(
        router_scores,
        current_slot,
        cuda_stream,
        is_decode_token=qlen == 1,
    )

    incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
    immediate_task = self.moe.forward_task(
        bsz_slot_tensor.data_ptr(),
        immediate_experts_ids_cpu[current_slot].size(-1),
        immediate_experts_ids_cpu[current_slot].data_ptr(),
        weights_cpu[current_slot].data_ptr(),
        input_tensor_cpu[current_slot].data_ptr(),
        output_cpu[current_slot].data_ptr(),
        incremental,
        router_scores_ptr,
        score_rows,
        score_cols,
        score_transform,
    )
    self._submit_cpuinfer_task(immediate_task, cuda_stream)

    BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
    if deferred_ids is not None:
        deferred_task = self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            deferred_experts_ids_cpu[current_slot].size(-1),
            deferred_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[next_slot].data_ptr(),
            False,
        )
        self._submit_cpuinfer_task(deferred_task, cuda_stream)
        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True


def _mesh_sync_forward_impl(
    self, hidden_states: torch.Tensor, topk_ids_or_stream=None, cuda_stream=None
) -> torch.Tensor:
    topk_ids = topk_ids_or_stream
    if cuda_stream is None and topk_ids_or_stream is not None:
        is_stream_arg = hasattr(topk_ids_or_stream, "cuda_stream") or isinstance(topk_ids_or_stream, int)
        if is_stream_arg:
            cuda_stream = getattr(topk_ids_or_stream, "cuda_stream", topk_ids_or_stream)
            topk_ids = None

    flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    (
        _input_tensor_cpu,
        _immediate_experts_ids_cpu,
        _deferred_experts_ids_cpu,
        _weights_cpu,
        output_cpu,
        _bsz_tensor_cpu,
        output_gpu,
    ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

    current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
    allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
    self._sync_cpuinfer(allow_pending, cuda_stream)
    output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
    self._mesh_after_sync_forward(
        flat_hidden_states,
        _immediate_experts_ids_cpu[current_slot],
        _weights_cpu[current_slot],
        output_cpu[current_slot],
        topk_ids,
        cuda_stream,
    )

    return output_gpu[current_slot]


def _mesh_forward_impl(
    self,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    cuda_stream,
    router_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    self._mesh_submit_forward_impl(hidden_states, topk_ids, topk_weights, cuda_stream, router_scores)
    return self._mesh_sync_forward_impl(hidden_states, topk_ids, cuda_stream)


def _maybe_mesh_prepare_prefill_layer_window(self, qlen: int) -> None:
    if qlen <= 1 or not self._mesh_prefill_layer_mode_enabled():
        return
    window = self._mesh_prefill_full_layer_count()
    if window <= 0:
        return

    # NOTE: previously this branch unconditionally released every layer in
    # _mesh_prefill_window_layers when layer_idx==0, intending to reset
    # state at the start of a new prefill. But that fires on EVERY SGLang
    # prefill chunk's first layer, even when we're still inside the same
    # user request (no decode in between). Result: cross-chunk scratch
    # cache lost ⇒ 3x read amplification observed.
    #
    # We now only do the bulk release when we know we've completed a full
    # prefill→decode→prefill round trip (i.e. transition_to_decode has
    # cleared the session_seen flag). Continuous prefill chunks within
    # the same session keep their window state.
    if (
        self.layer_idx == 0
        and BaseMoEWrapper._mesh_prefill_window_layers
        and not BaseMoEWrapper._mesh_prefill_session_seen
    ):
        for old_layer in sorted(BaseMoEWrapper._mesh_prefill_window_layers):
            old_wrapper = BaseMoEWrapper._wrappers_by_layer.get(old_layer)
            if old_wrapper is not None:
                old_wrapper._submit_mesh_noarg_task("mesh_release_prefill_layer_task")
        BaseMoEWrapper._mesh_prefill_window_layers.clear()

    BaseMoEWrapper._mesh_prefill_session_seen = True
    BaseMoEWrapper._mesh_decode_transition_done = False

    release_layer = int(self.layer_idx) - int(window)
    if release_layer in BaseMoEWrapper._mesh_prefill_window_layers:
        old_wrapper = BaseMoEWrapper._wrappers_by_layer.get(release_layer)
        if old_wrapper is not None and old_wrapper._submit_mesh_noarg_task("mesh_release_prefill_layer_task"):
            BaseMoEWrapper._mesh_prefill_window_layers.discard(release_layer)

    if self.layer_idx not in BaseMoEWrapper._mesh_prefill_window_layers:
        if self._submit_mesh_noarg_task("mesh_prepare_prefill_layer_task"):
            BaseMoEWrapper._mesh_prefill_window_layers.add(int(self.layer_idx))

    if not BaseMoEWrapper._mesh_prefill_window_logged:
        configured = self._mesh_global_resident_capacity()
        print(
            "[MESHPrefillLayerMode] "
            f"enabled window_layers={window} total_layers={self._mesh_total_moe_layers()} "
            f"configured_experts={configured} num_experts={self.num_experts}"
        )
        BaseMoEWrapper._mesh_prefill_window_logged = True


def _maybe_mesh_transition_to_decode_cache(self, qlen: int) -> None:
    if qlen != 1 or not self._mesh_prefill_layer_mode_enabled():
        return
    if not BaseMoEWrapper._mesh_prefill_session_seen or BaseMoEWrapper._mesh_decode_transition_done:
        return
    if self.layer_idx != 0:
        return

    for layer_idx, wrapper in sorted(BaseMoEWrapper._wrappers_by_layer.items()):
        if wrapper.io_backend != "IOURING" or wrapper.moe is None:
            continue
        task_factory = getattr(wrapper.moe, "mesh_transition_decode_cache_task", None)
        if task_factory is None:
            continue
        decode_capacity = max(0, int(wrapper._mesh_resident_capacity()))
        fill_limit = wrapper._env_int("KT_MESH_DECODE_TRANSITION_FILL_LIMIT", decode_capacity)
        task = task_factory(int(decode_capacity), int(fill_limit))
        wrapper._submit_cpuinfer_task(task, None)

    BaseMoEWrapper._mesh_prefill_window_layers.clear()
    BaseMoEWrapper._mesh_decode_transition_done = True
    BaseMoEWrapper._mesh_prefill_session_seen = False


def _should_debug_layer(cls, layer_idx: int) -> bool:
    raw = os.environ.get("KT_DEBUG_MOE_LAYERS")
    if not raw:
        return False
    if cls._debug_moe_layers is None:
        values = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.add(int(part))
            except ValueError:
                pass
        cls._debug_moe_layers = values
    return layer_idx in cls._debug_moe_layers and layer_idx not in cls._debug_logged_layers


def _debug_stats_tensor(t: torch.Tensor) -> dict:
    x = t.detach().float()
    finite = torch.isfinite(x)
    return {
        "shape": list(x.shape),
        "min": float(x.min().item()) if x.numel() else 0.0,
        "max": float(x.max().item()) if x.numel() else 0.0,
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "std": float(x.std().item()) if x.numel() > 1 else 0.0,
        "l2": float(torch.linalg.vector_norm(x).item()) if x.numel() else 0.0,
        "finite_ratio": float(finite.float().mean().item()) if x.numel() else 1.0,
    }


def _debug_log_moe_once(
    self,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
) -> None:
    if not BaseMoEWrapper._should_debug_layer(self.layer_idx):
        return
    BaseMoEWrapper._debug_logged_layers.add(self.layer_idx)
    line = {
        "layer": self.layer_idx,
        "weight_strategy": self.weight_strategy,
        "hidden": self._debug_stats_tensor(hidden_states),
        "output": self._debug_stats_tensor(output_tensor),
    }
    if topk_ids is not None and torch.is_tensor(topk_ids):
        ids = topk_ids.detach().cpu().reshape(-1)
        line["topk_ids_head"] = ids[:32].tolist()
        line["topk_ids_unique"] = int(ids.unique().numel()) if ids.numel() else 0
    if topk_weights is not None and torch.is_tensor(topk_weights):
        line["topk_weights"] = self._debug_stats_tensor(topk_weights)
        line["topk_weights_head"] = topk_weights.detach().cpu().reshape(-1)[:32].tolist()
    debug_path = os.environ.get("KT_DEBUG_MOE_LOG", "/tmp/kt_moe_debug.log")
    with open(debug_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    dump_dir = os.environ.get("KT_DEBUG_MOE_DUMP_DIR")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f"layer_{self.layer_idx}.pt")
        payload = {
            "layer": self.layer_idx,
            "weight_strategy": self.weight_strategy,
            "hidden_states": hidden_states.detach().cpu(),
            "topk_ids": topk_ids.detach().cpu() if torch.is_tensor(topk_ids) else None,
            "topk_weights": topk_weights.detach().cpu() if torch.is_tensor(topk_weights) else None,
            "output_tensor": output_tensor.detach().cpu(),
        }
        torch.save(payload, dump_path)


def _detect_available_numa_node_ids() -> List[int]:
    """Return actual Linux NUMA node ids instead of assuming 0..N-1."""
    node_root = Path("/sys/devices/system/node")
    node_ids: List[int] = []
    if node_root.exists():
        for entry in node_root.iterdir():
            if entry.name.startswith("node") and entry.name[4:].isdigit():
                node_ids.append(int(entry.name[4:]))
    if not node_ids:
        return [0]
    return sorted(set(node_ids))


def _submit_cpuinfer_task(self, task, cuda_stream=None):
    """Submit a CPUInfer task, tolerating CPU-only extension builds."""
    if cuda_stream is None:
        self.cpu_infer.submit(task)
        return

    self._ensure_cuda_graph_stream_compatible(cuda_stream, "submit")
    submit_with_stream = getattr(self.cpu_infer, "submit_with_cuda_stream", None)
    if submit_with_stream is not None:
        submit_with_stream(cuda_stream, task)
        return

    self._sync_external_cuda_stream(cuda_stream)
    self.cpu_infer.submit(task)


def _sync_cpuinfer(self, allow_pending: int = 0, cuda_stream=None):
    """Synchronize CPUInfer, using stream-aware hooks when available."""
    if cuda_stream is None:
        self.cpu_infer.sync(allow_pending)
        return

    self._ensure_cuda_graph_stream_compatible(cuda_stream, "sync")
    sync_with_stream = getattr(self.cpu_infer, "sync_with_cuda_stream", None)
    if sync_with_stream is not None:
        sync_with_stream(cuda_stream, allow_pending)
        return

    self.cpu_infer.sync(allow_pending)


def _cpuinfer_cuda_stream_hooks_available(self) -> bool:
    return callable(getattr(self.cpu_infer, "submit_with_cuda_stream", None)) and callable(
        getattr(self.cpu_infer, "sync_with_cuda_stream", None)
    )


def _ensure_cuda_graph_stream_compatible(self, cuda_stream, operation: str) -> None:
    if cuda_stream is None or not self._cuda_graph_capture_active():
        return
    if self._cpuinfer_cuda_stream_hooks_available():
        return
    raise RuntimeError(
        f"MESH CPUInfer cannot {operation} inside CUDA graph capture because this kt_kernel_ext "
        "was built without CUDA stream hooks. Build kt-kernel with CPUINFER_USE_CUDA=1 "
        "or start SGLang with --disable-cuda-graph."
    )


def cpuinfer_cuda_stream_hooks_available() -> bool:
    cpu_infer = getattr(BaseMoEWrapper, "_cpu_infer_instance", None)
    if cpu_infer is None:
        return False
    return callable(getattr(cpu_infer, "submit_with_cuda_stream", None)) and callable(
        getattr(cpu_infer, "sync_with_cuda_stream", None)
    )


def _sync_external_cuda_stream(cls, cuda_stream):
    if not cls._cpu_infer_stream_fallback_logged:
        print("[CPUInfer] submit_with_cuda_stream unavailable; " "falling back to synchronous CUDA stream handoff")
        cls._cpu_infer_stream_fallback_logged = True
    torch.cuda.ExternalStream(cuda_stream).synchronize()


def reset_runtime_state(force: bool = False):
    """
    Reset global runtime singletons once no active wrappers remain.

    force=True should only be used when the caller knows no live wrappers will
    touch the old CPUInfer objects afterwards.
    """
    if not force and BaseMoEWrapper._active_wrapper_count > 0:
        raise RuntimeError("Cannot reset runtime state while MoE wrappers are still active")
    BaseMoEWrapper._cpu_infer_instance = None
    BaseMoEWrapper._cpu_infer_signature = None
    BaseMoEWrapper._layer_has_pending_deferred.clear()
    BaseMoEWrapper._prev_topk_ids_by_layer.clear()
    BaseMoEWrapper._wrappers_by_layer.clear()
    BaseMoEWrapper._full_gate_batch_states.clear()
    BaseMoEWrapper._full_gate_skip_logged.clear()
    BaseMoEWrapper._mesh_bootstrap_done = False
    BaseMoEWrapper._mesh_bootstrap_log_count = 0
    BaseMoEWrapper._mesh_prefill_window_layers.clear()
    BaseMoEWrapper._mesh_prefill_session_seen = False
    BaseMoEWrapper._mesh_decode_transition_done = False
    BaseMoEWrapper._mesh_prefill_window_logged = False
    try:
        from .async_io_manager import shutdown_async_reader

        shutdown_async_reader()
    except Exception:
        pass
    BaseMoEWrapper.clear_buffer_cache()


def close(self):
    """Release layer-specific registrations and tear down globals when the last wrapper dies."""
    if getattr(self, "_closed", False):
        return

    try:
        if getattr(self, "cpu_infer", None) is not None:
            self.cpu_infer.sync()
    except Exception:
        pass

    BaseMoEWrapper._layer_has_pending_deferred.pop(self.layer_idx, None)
    BaseMoEWrapper._prev_topk_ids_by_layer.pop(self.layer_idx, None)
    BaseMoEWrapper._mesh_prefill_window_layers.discard(self.layer_idx)
    if BaseMoEWrapper._wrappers_by_layer.get(self.layer_idx) is self:
        BaseMoEWrapper._wrappers_by_layer.pop(self.layer_idx, None)

    self.moe = None
    self._closed = True

    if BaseMoEWrapper._active_wrapper_count > 0:
        BaseMoEWrapper._active_wrapper_count -= 1

    if BaseMoEWrapper._active_wrapper_count == 0:
        BaseMoEWrapper.reset_runtime_state(force=True)


def __del__(self):
    try:
        self.close()
    except Exception:
        pass


_INSTANCE_METHODS = (
    "_configure_base_runtime",
    "_debug_log_moe_once",
    "close",
    "__del__",
    "_submit_cpuinfer_task",
    "_sync_cpuinfer",
    "_cpuinfer_cuda_stream_hooks_available",
    "_ensure_cuda_graph_stream_compatible",
    "_mesh_score_transform_id",
    "_mesh_global_resident_capacity",
    "_mesh_prefill_layer_mode_enabled",
    "_mesh_prefill_rolling_enabled",
    "_mesh_prefill_rolling_depth",
    "_mesh_prefill_full_layer_count",
    "_mesh_slot_pool_capacity",
    "_mesh_prefill_static_resident_capacity",
    "_mesh_config_resident_experts",
    "_mesh_resident_capacity",
    "_mesh_current_cpu_expert_count",
    "_mesh_cpu_expert_mask",
    "_mesh_full_gate_observation_enabled",
    "_mesh_full_gate_batched_enabled",
    "_mesh_total_moe_layers",
    "_mesh_last_registered_layer_idx",
    "_mesh_last_full_gate_layer_idx",
    "_prepare_router_score_vector",
    "_get_full_gate_batch_state",
    "_mesh_full_gate_mask_batch",
    "_mesh_full_gate_batch_owner",
    "_mesh_bootstrap_prefetch_enabled",
    "_mesh_bootstrap_prefetch_budget",
    "_maybe_submit_mesh_bootstrap_from_full_gate_batch",
    "_flush_full_gate_batch",
    "_record_router_scores_for_token_batch",
    "_flush_pending_full_gate_batch",
    "observe_router_scores",
    "_prepare_router_scores_for_forward",
    "_submit_iouring_prefetch",
    "_submit_mesh_noarg_task",
    "_mesh_before_submit_forward",
    "_mesh_prefetch_previous_topk",
    "_mesh_select_forward_experts",
    "_mesh_prefetch_deferred_ids",
    "_mesh_after_sync_forward",
    "_mesh_submit_forward_impl",
    "_mesh_sync_forward_impl",
    "_mesh_forward_impl",
    "_maybe_mesh_prepare_prefill_layer_window",
    "_maybe_mesh_transition_to_decode_cache",
)

_STATIC_METHODS = (
    "reset_runtime_state",
    "_debug_stats_tensor",
    "_detect_available_numa_node_ids",
    "_cuda_graph_capture_active",
    "_discard_full_gate_batch_slot",
    "_env_flag",
    "_env_int",
    "cpuinfer_cuda_stream_hooks_available",
)

_CLASS_METHODS = ("_should_debug_layer", "_sync_external_cuda_stream", "_full_gate_batch_key")


def install_base_moe_helpers(wrapper_cls, buffer_cls, pin_memory: bool) -> None:
    global BaseMoEWrapper, KExpertsCPUBuffer, _PIN_MEMORY
    BaseMoEWrapper = wrapper_cls
    KExpertsCPUBuffer = buffer_cls
    _PIN_MEMORY = bool(pin_memory)

    wrapper_cls._cpu_infer_instance = None
    wrapper_cls._cpu_infer_signature = None
    wrapper_cls._active_wrapper_count = 0
    wrapper_cls._layer_has_pending_deferred = {}
    wrapper_cls._prev_topk_ids_by_layer = {}
    wrapper_cls._wrappers_by_layer = {}
    wrapper_cls._full_gate_batch_states = {}
    wrapper_cls._full_gate_skip_logged = set()
    wrapper_cls._mesh_bootstrap_done = False
    wrapper_cls._mesh_bootstrap_log_count = 0
    wrapper_cls._mesh_early_capacity_logged = set()
    wrapper_cls._mesh_prefill_window_layers = set()
    wrapper_cls._mesh_prefill_session_seen = False
    wrapper_cls._mesh_decode_transition_done = False
    wrapper_cls._mesh_prefill_window_logged = False
    wrapper_cls._mesh_runtime_config_logged = set()
    wrapper_cls._cpu_infer_stream_fallback_logged = False
    wrapper_cls._debug_moe_layers = None
    wrapper_cls._debug_logged_layers = set()

    for name in _INSTANCE_METHODS:
        setattr(wrapper_cls, name, globals()[name])
    for name in _STATIC_METHODS:
        setattr(wrapper_cls, name, staticmethod(globals()[name]))
    for name in _CLASS_METHODS:
        setattr(wrapper_cls, name, classmethod(globals()[name]))
