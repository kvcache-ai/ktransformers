import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _import_kt_kernel_ext():
    try:
        import kt_kernel

        return kt_kernel.kt_kernel_ext
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
        import kt_kernel

        return kt_kernel.kt_kernel_ext


kt_kernel_ext = _import_kt_kernel_ext()


HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
K_GROUP_SIZE = 32


@dataclass
class ForwardCache:
    qlen: int
    k: int
    active_experts: List[int]
    local_counts: List[int]
    input: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    intermediate: torch.Tensor
    down: torch.Tensor
    down_lora_u: torch.Tensor


class FixedTopKRouter(nn.Module):
    def __init__(self, expert_ids: torch.Tensor, route_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("expert_ids", expert_ids.to(torch.int64).contiguous())
        self.register_buffer("route_weights", route_weights.to(torch.bfloat16).contiguous())

    def forward(self, hidden_states: torch.Tensor):
        qlen = hidden_states.shape[0]
        if qlen != self.expert_ids.shape[0]:
            raise ValueError(f"FixedTopKRouter qlen mismatch: expected {self.expert_ids.shape[0]}, got {qlen}")
        router_logits = torch.empty((qlen, self.expert_ids.max().item() + 1), dtype=torch.float32)
        return router_logits, self.route_weights, self.expert_ids


class DummyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Identity()
        self.up_proj = nn.Identity()
        self.down_proj = nn.Identity()


class DummyOriginalMoE(nn.Module):
    def __init__(self, expert_num: int, expert_ids: torch.Tensor, route_weights: torch.Tensor):
        super().__init__()
        self.gate = FixedTopKRouter(expert_ids, route_weights)
        self.experts = nn.ModuleList(DummyExpert() for _ in range(expert_num))


class LoraWeightModule(nn.Module):
    def __init__(self, weight: torch.Tensor, grad: torch.Tensor | None = None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=True)
        if grad is not None:
            self.weight.grad = grad


def zero_storage_parameter(shape: Tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> nn.Parameter:
    storage = torch.UntypedStorage(1, device="cpu")
    tensor = torch.tensor([], dtype=dtype, device="cpu").set_(
        storage,
        storage_offset=0,
        size=shape,
        stride=[0] * len(shape),
    )
    return nn.Parameter(tensor, requires_grad=False)


class FakePeftLinear(nn.Module):
    def __init__(self, weight_shape: Tuple[int, int], lora_a: torch.Tensor, lora_b: torch.Tensor):
        super().__init__()
        self.weight = zero_storage_parameter(weight_shape)
        self.lora_A = nn.ModuleDict({"default": LoraWeightModule(lora_a.clone())})
        self.lora_B = nn.ModuleDict({"default": LoraWeightModule(lora_b.clone())})
        self.active_adapter = ["default"]


class FakePeftExpert(nn.Module):
    def __init__(self, lora: Dict[str, torch.Tensor], expert_idx: int):
        super().__init__()
        self.gate_proj = FakePeftLinear(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            lora["gate_a"][expert_idx],
            lora["gate_b"][expert_idx],
        )
        self.up_proj = FakePeftLinear(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            lora["up_a"][expert_idx],
            lora["up_b"][expert_idx],
        )
        self.down_proj = FakePeftLinear(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            lora["down_a"][expert_idx],
            lora["down_b"][expert_idx],
        )


class FakeWrappedOriginalMoE(nn.Module):
    def __init__(
        self,
        expert_num: int,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        shadow: Dict[str, torch.Tensor] | None,
        lora: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.gate = FixedTopKRouter(expert_ids, route_weights)
        self.experts = nn.ModuleList(FakePeftExpert(lora, expert_idx) for expert_idx in range(expert_num))
        self._kt_kgroup_tensors = {
            "gate_proj": weights["gate_q"],
            "gate_scale": weights["gate_scales"],
            "up_proj": weights["up_q"],
            "up_scale": weights["up_scales"],
            "down_proj": weights["down_q"],
            "down_scale": weights["down_scales"],
        }
        if shadow is not None:
            self._kt_kgroup_tensors.update(
                {
                    "gate_bwd_shadow": shadow["gate"],
                    "up_bwd_shadow": shadow["up"],
                    "down_bwd_shadow": shadow["down"],
                }
            )


class FakeBlock(nn.Module):
    def __init__(self, moe: nn.Module):
        super().__init__()
        self.mlp = moe


class FakeKGroupModel(nn.Module):
    def __init__(
        self,
        expert_num: int,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        shadow: Dict[str, torch.Tensor] | None,
        lora: Dict[str, torch.Tensor],
        max_len: int,
        num_layers: int = 1,
    ):
        super().__init__()
        config = type("FakeQwen3MoeConfig", (), {})()
        config.architectures = ["Qwen3MoeForCausalLM"]
        config.hidden_size = HIDDEN_SIZE
        config.num_hidden_layers = num_layers
        config.num_experts = expert_num
        config.moe_intermediate_size = INTERMEDIATE_SIZE
        config.num_experts_per_tok = 2
        config.shared_expert_intermediate_size = 0
        config.max_position_embeddings = max_len
        self.config = config
        self.layers = nn.ModuleList(
            [
                FakeBlock(FakeWrappedOriginalMoE(expert_num, expert_ids, route_weights, weights, shadow, lora))
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer.mlp(hidden_states)
        return hidden_states


def _bf16_randn(shape: Tuple[int, ...], scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(torch.bfloat16).contiguous()


def _make_packed_weight(expert_num: int, rows: int, cols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bytes_per_expert = rows * cols // 2
    return torch.randint(0, 255, (expert_num, bytes_per_expert), generator=generator, dtype=torch.uint8).contiguous()


def _make_scales(expert_num: int, rows: int, cols: int, seed: int) -> torch.Tensor:
    groups = cols // K_GROUP_SIZE
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scales = torch.rand((expert_num, rows, groups), generator=generator, dtype=torch.float32) * 0.003 + 0.001
    return scales.to(torch.bfloat16).contiguous()


def dequant_kgroup_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    rows: int,
    cols: int,
    group_size: int = K_GROUP_SIZE,
) -> torch.Tensor:
    packed_flat = packed.reshape(-1).to(torch.int16)
    q = torch.empty((rows * cols,), dtype=torch.int16)
    q[0::2] = packed_flat & 0x0F
    q[1::2] = (packed_flat >> 4) & 0x0F
    q = q.reshape(rows, cols).float() - 8.0
    scale = scales.reshape(rows, cols // group_size).float().repeat_interleave(group_size, dim=1)
    return q * scale


def dequant_kgroup_weights(weights: Dict[str, torch.Tensor], expert_num: int) -> Dict[str, torch.Tensor]:
    gate = torch.stack(
        [
            dequant_kgroup_weight(
                weights["gate_q"][expert],
                weights["gate_scales"][expert],
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
            )
            for expert in range(expert_num)
        ],
        dim=0,
    )
    up = torch.stack(
        [
            dequant_kgroup_weight(
                weights["up_q"][expert],
                weights["up_scales"][expert],
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
            )
            for expert in range(expert_num)
        ],
        dim=0,
    )
    down = torch.stack(
        [
            dequant_kgroup_weight(
                weights["down_q"][expert],
                weights["down_scales"][expert],
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
            )
            for expert in range(expert_num)
        ],
        dim=0,
    )
    return {"gate": gate, "up": up, "down": down}


def make_weights(expert_num: int) -> Dict[str, torch.Tensor]:
    return {
        "gate_q": _make_packed_weight(expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE, 100),
        "up_q": _make_packed_weight(expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE, 101),
        "down_q": _make_packed_weight(expert_num, HIDDEN_SIZE, INTERMEDIATE_SIZE, 102),
        "gate_scales": _make_scales(expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE, 103),
        "up_scales": _make_scales(expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE, 104),
        "down_scales": _make_scales(expert_num, HIDDEN_SIZE, INTERMEDIATE_SIZE, 105),
    }


def make_shadow_weights(expert_num: int) -> Dict[str, torch.Tensor]:
    return {
        "gate": _bf16_randn((expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE), 0.02, 110),
        "up": _bf16_randn((expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE), 0.02, 111),
        "down": _bf16_randn((expert_num, HIDDEN_SIZE, INTERMEDIATE_SIZE), 0.02, 112),
    }


def make_cpuinfer(thread_count: int, tp_count: int = 1):
    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = tp_count
    worker_config.subpool_numa_map = list(range(tp_count))
    worker_config.subpool_thread_count = [thread_count] * tp_count
    return kt_kernel_ext.CPUInfer(worker_config)


def make_lora(expert_num: int, rank: int, scale: float, seed_offset: int) -> Dict[str, torch.Tensor]:
    return {
        "gate_a": _bf16_randn((expert_num, rank, HIDDEN_SIZE), scale, 200 + seed_offset),
        "gate_b": _bf16_randn((expert_num, INTERMEDIATE_SIZE, rank), scale, 201 + seed_offset),
        "up_a": _bf16_randn((expert_num, rank, HIDDEN_SIZE), scale, 202 + seed_offset),
        "up_b": _bf16_randn((expert_num, INTERMEDIATE_SIZE, rank), scale, 203 + seed_offset),
        "down_a": _bf16_randn((expert_num, rank, INTERMEDIATE_SIZE), scale, 204 + seed_offset),
        "down_b": _bf16_randn((expert_num, HIDDEN_SIZE, rank), scale, 205 + seed_offset),
    }


def make_routing(qlen: int, expert_num: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if k != 2:
        raise ValueError("make_routing currently expects k=2")
    expert_rows = []
    weight_rows = []
    weight_patterns = ((0.7, 0.3), (0.4, 0.6), (0.55, 0.45))
    for token_idx in range(qlen):
        first = token_idx % expert_num
        second = (token_idx + 1) % expert_num
        expert_rows.append((first, second))
        weight_rows.append(weight_patterns[token_idx % len(weight_patterns)])
    return (
        torch.tensor(expert_rows, dtype=torch.int64).contiguous(),
        torch.tensor(weight_rows, dtype=torch.float32).contiguous(),
    )


def expected_routing_meta(
    expert_ids: torch.Tensor,
    expert_num: int,
) -> Tuple[List[int], List[int], List[Tuple[int, int, int]]]:
    active = sorted({int(expert) for expert in expert_ids.flatten().tolist()})
    counts = [int((expert_ids == expert).sum().item()) for expert in range(expert_num)]
    meta = []
    for expert in active:
        for token_idx in range(expert_ids.shape[0]):
            for route_idx in range(expert_ids.shape[1]):
                if int(expert_ids[token_idx, route_idx]) == expert:
                    meta.append((token_idx, route_idx, expert))
    return active, counts, meta


def zero_lora_like(lora: Dict[str, torch.Tensor], zero_down_only: bool = False) -> Dict[str, torch.Tensor]:
    result = {}
    for name, tensor in lora.items():
        if not zero_down_only or name.startswith("down_"):
            result[name] = torch.zeros_like(tensor)
        else:
            result[name] = tensor
    return result


def make_moe(
    cpuinfer,
    weights: Dict[str, torch.Tensor],
    lora: Dict[str, torch.Tensor],
    shadow: Dict[str, torch.Tensor] | None,
    expert_num: int,
    max_len: int,
    rank: int,
    scaling: float,
    skip_lora: bool = False,
):
    config = kt_kernel_ext.moe.MOESFTConfig(expert_num, 2, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    config.max_len = max_len
    config.max_cache_depth = 1
    config.pool = cpuinfer.backend_
    config.quant_config.bits = 4
    config.quant_config.group_size = K_GROUP_SIZE
    config.quant_config.zero_point = False
    config.gate_proj = weights["gate_q"].data_ptr()
    config.up_proj = weights["up_q"].data_ptr()
    config.down_proj = weights["down_q"].data_ptr()
    config.gate_scale = weights["gate_scales"].data_ptr()
    config.up_scale = weights["up_scales"].data_ptr()
    config.down_scale = weights["down_scales"].data_ptr()
    if shadow is not None:
        config.gate_bwd_shadow = shadow["gate"].data_ptr()
        config.up_bwd_shadow = shadow["up"].data_ptr()
        config.down_bwd_shadow = shadow["down"].data_ptr()
    config.lora_rank = rank
    config.lora_alpha = scaling * rank
    config.gate_lora_a = lora["gate_a"].data_ptr()
    config.gate_lora_b = lora["gate_b"].data_ptr()
    config.up_lora_a = lora["up_a"].data_ptr()
    config.up_lora_b = lora["up_b"].data_ptr()
    config.down_lora_a = lora["down_a"].data_ptr()
    config.down_lora_b = lora["down_b"].data_ptr()

    moe_cls = kt_kernel_ext.moe.AMXInt4_KGroup_SFT_MOE_SkipLoRA if skip_lora else kt_kernel_ext.moe.AMXInt4_KGroup_SFT_MOE
    moe = moe_cls(config)
    moe.load_weights()
    return moe


def make_base_k2_moe(cpuinfer, weights: Dict[str, torch.Tensor], expert_num: int, max_len: int):
    config = kt_kernel_ext.moe.MOEConfig(expert_num, 2, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0)
    config.max_len = max_len
    config.quant_config.bits = 4
    config.quant_config.group_size = K_GROUP_SIZE
    config.quant_config.zero_point = False
    config.gate_proj = weights["gate_q"].data_ptr()
    config.up_proj = weights["up_q"].data_ptr()
    config.down_proj = weights["down_q"].data_ptr()
    config.gate_scale = weights["gate_scales"].data_ptr()
    config.up_scale = weights["up_scales"].data_ptr()
    config.down_scale = weights["down_scales"].data_ptr()
    config.pool = cpuinfer.backend_

    moe = kt_kernel_ext.moe.AMXInt4_KGroup_MOE(config)
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu")
    cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpuinfer.sync()
    return moe


def make_wrapper(
    weights: Dict[str, torch.Tensor],
    lora: Dict[str, torch.Tensor],
    shadow: Dict[str, torch.Tensor] | None,
    expert_num: int,
    max_len: int,
    rank: int,
    scaling: float,
    thread_count: int,
    tp_count: int = 1,
) -> Tuple[object, Dict[str, torch.Tensor]]:
    from kt_kernel.sft.amx import AMXSFTMoEWrapper

    wrapper = AMXSFTMoEWrapper(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=2,
        hidden_size=HIDDEN_SIZE,
        moe_intermediate_size=INTERMEDIATE_SIZE,
        num_gpu_experts=0,
        cpuinfer_threads=thread_count,
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=max_len,
        lora_rank=rank,
        lora_alpha=scaling * rank,
        max_cache_depth=1,
        method="AMXINT4_KGroup_SFT",
        group_size=K_GROUP_SIZE,
        zero_point=False,
    )
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu")
    wrapper.load_kgroup_weights_from_tensors(
        gate_proj=weights["gate_q"],
        gate_scale=weights["gate_scales"],
        up_proj=weights["up_q"],
        up_scale=weights["up_scales"],
        down_proj=weights["down_q"],
        down_scale=weights["down_scales"],
        physical_to_logical_map_cpu=physical_to_logical_map,
    )
    grad_lora = {name: torch.zeros_like(tensor) for name, tensor in lora.items()}
    wrapper.init_lora_weights(
        lora["gate_a"],
        lora["gate_b"],
        lora["up_a"],
        lora["up_b"],
        lora["down_a"],
        lora["down_b"],
        grad_lora["gate_a"],
        grad_lora["gate_b"],
        grad_lora["up_a"],
        grad_lora["up_b"],
        grad_lora["down_a"],
        grad_lora["down_b"],
    )
    return wrapper, grad_lora


def make_peft_lora_modules(
    lora: Dict[str, torch.Tensor],
    grad_lora: Dict[str, torch.Tensor],
    expert_num: int,
) -> Dict[int, Dict[str, Tuple[nn.Module, nn.Module]]]:
    result: Dict[int, Dict[str, Tuple[nn.Module, nn.Module]]] = {}
    for expert_idx in range(expert_num):
        result[expert_idx] = {
            "gate_proj": (
                LoraWeightModule(lora["gate_a"][expert_idx], grad_lora["gate_a"][expert_idx]),
                LoraWeightModule(lora["gate_b"][expert_idx], grad_lora["gate_b"][expert_idx]),
            ),
            "up_proj": (
                LoraWeightModule(lora["up_a"][expert_idx], grad_lora["up_a"][expert_idx]),
                LoraWeightModule(lora["up_b"][expert_idx], grad_lora["up_b"][expert_idx]),
            ),
            "down_proj": (
                LoraWeightModule(lora["down_a"][expert_idx], grad_lora["down_a"][expert_idx]),
                LoraWeightModule(lora["down_b"][expert_idx], grad_lora["down_b"][expert_idx]),
            ),
        }
    return result


def stack_peft_grad(
    peft_lora_modules: Dict[int, Dict[str, Tuple[nn.Module, nn.Module]]],
    expert_num: int,
    proj_name: str,
    lora_index: int,
) -> torch.Tensor:
    return torch.stack(
        [peft_lora_modules[expert_idx][proj_name][lora_index].weight.grad for expert_idx in range(expert_num)],
        dim=0,
    ).contiguous()


def iter_peft_lora_weights(peft_lora_modules: Dict[int, Dict[str, Tuple[nn.Module, nn.Module]]]):
    for expert_idx in sorted(peft_lora_modules):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            if proj_name not in peft_lora_modules[expert_idx]:
                continue
            for lora_index, module in enumerate(peft_lora_modules[expert_idx][proj_name]):
                yield f"expert{expert_idx}.{proj_name}.{lora_index}", module.weight


def max_peft_weight_delta(
    before: Dict[str, torch.Tensor],
    peft_lora_modules: Dict[int, Dict[str, Tuple[nn.Module, nn.Module]]],
) -> torch.Tensor:
    deltas = []
    for name, weight in iter_peft_lora_weights(peft_lora_modules):
        deltas.append((weight.detach().float() - before[name].float()).abs().max())
    return torch.stack(deltas).max()


def assert_peft_grad_alias(
    peft_lora_modules: Dict[int, Dict[str, Tuple[nn.Module, nn.Module]]],
    grad_lora: Dict[str, torch.Tensor],
    expert_num: int,
) -> None:
    checks = (
        ("gate_proj", 0, "gate_a"),
        ("gate_proj", 1, "gate_b"),
        ("up_proj", 0, "up_a"),
        ("up_proj", 1, "up_b"),
        ("down_proj", 0, "down_a"),
        ("down_proj", 1, "down_b"),
    )
    for expert_idx in range(expert_num):
        for proj_name, lora_index, grad_key in checks:
            actual_grad = peft_lora_modules[expert_idx][proj_name][lora_index].weight.grad
            expected_grad = grad_lora[grad_key][expert_idx]
            if actual_grad is None:
                raise AssertionError(f"{proj_name}[{lora_index}] expert {expert_idx} grad is None")
            if actual_grad.data_ptr() != expected_grad.data_ptr():
                raise AssertionError(
                    f"{proj_name}[{lora_index}] expert {expert_idx} grad buffer is not aliased to {grad_key}"
                )


def assert_peft_any_grad_nonzero(
    label: str,
    peft_lora_modules: Dict[int, Dict[str, Tuple[nn.Module, nn.Module]]],
) -> None:
    max_abs = 0.0
    seen = False
    for expert_loras in peft_lora_modules.values():
        for lora_a, lora_b in expert_loras.values():
            for module in (lora_a, lora_b):
                grad = module.weight.grad
                if grad is None:
                    continue
                seen = True
                max_abs = max(max_abs, grad.float().abs().max().item())
    print(f"{label}: observed_max_abs={max_abs:.6g}")
    if not seen:
        raise AssertionError(f"{label} did not observe any LoRA grad buffers")
    if max_abs == 0.0:
        raise AssertionError(f"{label} LoRA gradients are all zero")


def run_forward(
    moe,
    qlen_tensor: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    x: torch.Tensor,
    save_for_backward: bool = True,
):
    output = torch.empty((int(qlen_tensor.item()), HIDDEN_SIZE), dtype=torch.bfloat16)
    moe.forward_sft(
        qlen_tensor.data_ptr(),
        expert_ids.shape[1],
        expert_ids.data_ptr(),
        weights.data_ptr(),
        x.data_ptr(),
        output.data_ptr(),
        save_for_backward,
    )
    return output


def run_base_forward(
    cpuinfer,
    moe,
    qlen_tensor: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    x: torch.Tensor,
):
    output = torch.empty((int(qlen_tensor.item()), HIDDEN_SIZE), dtype=torch.bfloat16)
    cpuinfer.submit(
        moe.forward_task(
            qlen_tensor.data_ptr(),
            expert_ids.shape[1],
            expert_ids.data_ptr(),
            weights.data_ptr(),
            x.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpuinfer.sync()
    return output


def copy_forward_cache(moe, rank: int) -> ForwardCache:
    qlen, k, active_count, active_experts, local_counts = moe.debug_cache_summary()
    active_experts = [int(x) for x in active_experts[:active_count]]
    local_counts = [int(x) for x in local_counts]
    total_tokens = sum(local_counts[expert] for expert in active_experts)

    input_cache = torch.empty((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
    gate_cache = torch.empty((total_tokens, INTERMEDIATE_SIZE), dtype=torch.bfloat16)
    up_cache = torch.empty((total_tokens, INTERMEDIATE_SIZE), dtype=torch.bfloat16)
    intermediate_cache = torch.empty((total_tokens, INTERMEDIATE_SIZE), dtype=torch.bfloat16)
    down_cache = torch.empty((total_tokens, HIDDEN_SIZE), dtype=torch.bfloat16)
    down_lora_u_cache = torch.empty((total_tokens, rank), dtype=torch.float32)

    moe.debug_copy_forward_cache(
        input_cache.data_ptr(),
        gate_cache.data_ptr(),
        up_cache.data_ptr(),
        intermediate_cache.data_ptr(),
        down_cache.data_ptr(),
        down_lora_u_cache.data_ptr(),
    )

    assert input_cache.shape == (qlen, HIDDEN_SIZE)
    assert gate_cache.shape == (total_tokens, INTERMEDIATE_SIZE)
    assert up_cache.shape == (total_tokens, INTERMEDIATE_SIZE)
    assert intermediate_cache.shape == (total_tokens, INTERMEDIATE_SIZE)
    assert down_cache.shape == (total_tokens, HIDDEN_SIZE)
    assert down_lora_u_cache.shape == (total_tokens, rank)

    return ForwardCache(
        qlen=qlen,
        k=k,
        active_experts=active_experts,
        local_counts=local_counts,
        input=input_cache,
        gate=gate_cache,
        up=up_cache,
        intermediate=intermediate_cache,
        down=down_cache,
        down_lora_u=down_lora_u_cache,
    )


def copy_bwd_shadow(moe, expert_num: int) -> Dict[str, torch.Tensor]:
    prepared, summary_experts, summary_hidden, summary_intermediate = moe.debug_bwd_shadow_summary()
    assert prepared
    assert summary_experts == expert_num
    assert summary_hidden == HIDDEN_SIZE
    assert summary_intermediate == INTERMEDIATE_SIZE

    gate = torch.empty((expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16)
    up = torch.empty((expert_num, INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16)
    down = torch.empty((expert_num, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.bfloat16)
    moe.debug_copy_bwd_shadow(gate.data_ptr(), up.data_ptr(), down.data_ptr())
    return {"gate": gate, "up": up, "down": down}


def assert_packed_weight_contract(moe, weights: Dict[str, torch.Tensor], expert_num: int):
    (
        ready,
        summary_experts,
        summary_hidden,
        summary_intermediate,
        summary_group_size,
        gate_up_bytes,
        down_bytes,
        gate_up_scale_elems,
        down_scale_elems,
    ) = moe.debug_packed_weight_summary()
    assert ready
    assert summary_experts == expert_num
    assert summary_hidden == HIDDEN_SIZE
    assert summary_intermediate == INTERMEDIATE_SIZE
    assert summary_group_size == K_GROUP_SIZE
    assert gate_up_bytes == INTERMEDIATE_SIZE * HIDDEN_SIZE // 2
    assert down_bytes == HIDDEN_SIZE * INTERMEDIATE_SIZE // 2
    assert gate_up_scale_elems == INTERMEDIATE_SIZE * (HIDDEN_SIZE // K_GROUP_SIZE)
    assert down_scale_elems == HIDDEN_SIZE * (INTERMEDIATE_SIZE // K_GROUP_SIZE)

    gate = torch.empty((expert_num, gate_up_bytes), dtype=torch.uint8)
    up = torch.empty((expert_num, gate_up_bytes), dtype=torch.uint8)
    down = torch.empty((expert_num, down_bytes), dtype=torch.uint8)
    gate_scale = torch.empty((expert_num, gate_up_scale_elems), dtype=torch.float32)
    up_scale = torch.empty((expert_num, gate_up_scale_elems), dtype=torch.float32)
    down_scale = torch.empty((expert_num, down_scale_elems), dtype=torch.float32)
    moe.debug_copy_packed_weights(
        gate.data_ptr(),
        up.data_ptr(),
        down.data_ptr(),
        gate_scale.data_ptr(),
        up_scale.data_ptr(),
        down_scale.data_ptr(),
    )

    torch.testing.assert_close(gate, weights["gate_q"].view_as(gate), atol=0, rtol=0)
    torch.testing.assert_close(up, weights["up_q"].view_as(up), atol=0, rtol=0)
    torch.testing.assert_close(down, weights["down_q"].view_as(down), atol=0, rtol=0)
    torch.testing.assert_close(gate_scale, weights["gate_scales"].view_as(gate_scale).float(), atol=0, rtol=0)
    torch.testing.assert_close(up_scale, weights["up_scales"].view_as(up_scale).float(), atol=0, rtol=0)
    torch.testing.assert_close(down_scale, weights["down_scales"].view_as(down_scale).float(), atol=0, rtol=0)
    print(
        "packed_weight_contract_passed: "
        f"experts={expert_num}, gate_up_bytes={gate_up_bytes}, down_bytes={down_bytes}, "
        f"gate_up_scale_elems={gate_up_scale_elems}, down_scale_elems={down_scale_elems}"
    )


def assert_raises_packed_backward_retired(name: str, fn):
    expected = "K2 RAWINT4 SFT packed backward is not implemented yet; BF16 shadow path is retired"
    try:
        fn()
    except RuntimeError as exc:
        if expected not in str(exc):
            raise
        print(f"{name}: observed packed-backward guard")
        return
    raise AssertionError(f"{name} did not raise packed-backward guard")


def pack_input_by_expert(
    x: torch.Tensor, expert_ids: torch.Tensor, active_experts: List[int]
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    rows = []
    meta = []
    for expert in active_experts:
        for token_idx in range(expert_ids.shape[0]):
            for route_idx in range(expert_ids.shape[1]):
                if int(expert_ids[token_idx, route_idx]) == expert:
                    rows.append(x[token_idx])
                    meta.append((token_idx, route_idx, expert))
    if not rows:
        return torch.empty((0, x.shape[1]), dtype=x.dtype), meta
    return torch.stack(rows, dim=0).contiguous(), meta


def lora_linear_by_expert(
    x: torch.Tensor,
    active_experts: List[int],
    local_counts: List[int],
    a: torch.Tensor,
    b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    pieces = []
    offset = 0
    for expert in active_experts:
        count = local_counts[expert]
        if count == 0:
            continue
        x_part = x[offset : offset + count].float()
        a_part = a[expert].float()
        b_part = b[expert].float()
        pieces.append((x_part @ a_part.t()) @ b_part.t() * scaling)
        offset += count
    return torch.cat(pieces, dim=0) if pieces else torch.empty((0, b.shape[1]), dtype=torch.float32)


def down_lora_u_ref(cache: ForwardCache, lora: Dict[str, torch.Tensor]) -> torch.Tensor:
    pieces = []
    offset = 0
    for expert in cache.active_experts:
        count = cache.local_counts[expert]
        if count == 0:
            continue
        intermediate = cache.intermediate[offset : offset + count].float()
        pieces.append(intermediate @ lora["down_a"][expert].float().t())
        offset += count
    return torch.cat(pieces, dim=0) if pieces else torch.empty_like(cache.down_lora_u)


def merge_down_cache(cache: ForwardCache, expert_ids: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
    base_offset = {}
    cursor = 0
    for expert in cache.active_experts:
        base_offset[expert] = cursor
        cursor += cache.local_counts[expert]

    per_expert_pos = {expert: 0 for expert in cache.active_experts}
    output = torch.zeros((cache.qlen, HIDDEN_SIZE), dtype=torch.float32)
    for token_idx in range(cache.qlen):
        for route_idx in range(cache.k):
            expert = int(expert_ids[token_idx, route_idx])
            row = base_offset[expert] + per_expert_pos[expert]
            output[token_idx] += route_weights[token_idx, route_idx].item() * cache.down[row].float()
            per_expert_pos[expert] += 1
    return output


def remerge_forward_cache_cpp(moe, qlen: int) -> torch.Tensor:
    output = torch.zeros((qlen, HIDDEN_SIZE), dtype=torch.float32)
    moe.debug_remerge_forward_cache(output.data_ptr())
    return output


def grad_weights_from_down_cache(cache: ForwardCache, expert_ids: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    base_offset = {}
    cursor = 0
    for expert in cache.active_experts:
        base_offset[expert] = cursor
        cursor += cache.local_counts[expert]

    per_expert_pos = {expert: 0 for expert in cache.active_experts}
    grad_weights = torch.zeros((cache.qlen, cache.k), dtype=torch.float32)
    for token_idx in range(cache.qlen):
        for route_idx in range(cache.k):
            expert = int(expert_ids[token_idx, route_idx])
            row = base_offset[expert] + per_expert_pos[expert]
            grad_weights[token_idx, route_idx] = (grad_output[token_idx].float() * cache.down[row].float()).sum()
            per_expert_pos[expert] += 1
    return grad_weights


def backward_down_ref(
    cache: ForwardCache,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    lora: Dict[str, torch.Tensor],
    packed_weight: Dict[str, torch.Tensor],
    expert_num: int,
    scaling: float,
) -> Dict[str, torch.Tensor]:
    base_offset = {}
    cursor = 0
    for expert in cache.active_experts:
        base_offset[expert] = cursor
        cursor += cache.local_counts[expert]

    per_expert_pos = {expert: 0 for expert in cache.active_experts}
    grad_down = torch.zeros((cursor, HIDDEN_SIZE), dtype=torch.float32)
    for token_idx in range(cache.qlen):
        for route_idx in range(cache.k):
            expert = int(expert_ids[token_idx, route_idx])
            row = base_offset[expert] + per_expert_pos[expert]
            grad_down[row] += grad_output[token_idx].float() * route_weights[token_idx, route_idx].item()
            per_expert_pos[expert] += 1

    rank = lora["down_a"].shape[1]
    grad_intermediate = torch.zeros((cursor, INTERMEDIATE_SIZE), dtype=torch.float32)
    grad_down_lora_a = torch.zeros((expert_num, rank, INTERMEDIATE_SIZE), dtype=torch.float32)
    grad_down_lora_b = torch.zeros((expert_num, HIDDEN_SIZE, rank), dtype=torch.float32)

    offset = 0
    for expert in cache.active_experts:
        count = cache.local_counts[expert]
        if count == 0:
            continue
        grad_down_part = grad_down[offset : offset + count]
        intermediate_part = cache.intermediate[offset : offset + count].float()
        down_u_part = cache.down_lora_u[offset : offset + count].float()

        grad_intermediate[offset : offset + count] += grad_down_part @ packed_weight["down"][expert].float()
        grad_times_b = grad_down_part @ lora["down_b"][expert].float()
        grad_intermediate[offset : offset + count] += scaling * (grad_times_b @ lora["down_a"][expert].float())
        grad_down_lora_a[expert] += scaling * (grad_times_b.t() @ intermediate_part)
        grad_down_lora_b[expert] += scaling * (grad_down_part.t() @ down_u_part)
        offset += count

    return {
        "grad_down": grad_down.to(torch.bfloat16),
        "grad_intermediate": grad_intermediate.to(torch.bfloat16),
        "grad_down_lora_a": grad_down_lora_a.to(torch.bfloat16),
        "grad_down_lora_b": grad_down_lora_b.to(torch.bfloat16),
    }


def activation_backward_ref(cache: ForwardCache, grad_intermediate: torch.Tensor) -> Dict[str, torch.Tensor]:
    gate = cache.gate.float()
    up = cache.up.float()
    grad_inter = grad_intermediate.float()
    sigmoid = torch.sigmoid(gate)
    silu = torch.nn.functional.silu(gate)
    silu_grad = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    return {
        "grad_gate": (grad_inter * up * silu_grad).to(torch.bfloat16),
        "grad_up": (grad_inter * silu).to(torch.bfloat16),
    }


def backward_gate_up_ref(
    cache: ForwardCache,
    expert_ids: torch.Tensor,
    grad_gate: torch.Tensor,
    grad_up: torch.Tensor,
    lora: Dict[str, torch.Tensor],
    packed_weight: Dict[str, torch.Tensor],
    expert_num: int,
    scaling: float,
) -> Dict[str, torch.Tensor]:
    base_offset = {}
    cursor = 0
    for expert in cache.active_experts:
        base_offset[expert] = cursor
        cursor += cache.local_counts[expert]

    rank = lora["gate_a"].shape[1]
    grad_input = torch.zeros((cache.qlen, HIDDEN_SIZE), dtype=torch.float32)
    grad_gate_lora_a = torch.zeros((expert_num, rank, HIDDEN_SIZE), dtype=torch.float32)
    grad_gate_lora_b = torch.zeros((expert_num, INTERMEDIATE_SIZE, rank), dtype=torch.float32)
    grad_up_lora_a = torch.zeros((expert_num, rank, HIDDEN_SIZE), dtype=torch.float32)
    grad_up_lora_b = torch.zeros((expert_num, INTERMEDIATE_SIZE, rank), dtype=torch.float32)

    per_expert_pos = {expert: 0 for expert in cache.active_experts}
    for token_idx in range(cache.qlen):
        x = cache.input[token_idx].float()
        for route_idx in range(cache.k):
            expert = int(expert_ids[token_idx, route_idx])
            row = base_offset[expert] + per_expert_pos[expert]
            per_expert_pos[expert] += 1

            for prefix, grad in (("gate", grad_gate[row].float()), ("up", grad_up[row].float())):
                a = lora[f"{prefix}_a"][expert].float()
                b = lora[f"{prefix}_b"][expert].float()
                u = x @ a.t()
                grad_times_b = grad @ b

                grad_input[token_idx] += grad @ packed_weight[prefix][expert].float()
                grad_input[token_idx] += scaling * (grad_times_b @ a)

                if prefix == "gate":
                    grad_gate_lora_a[expert] += scaling * torch.outer(grad_times_b, x)
                    grad_gate_lora_b[expert] += scaling * torch.outer(grad, u)
                else:
                    grad_up_lora_a[expert] += scaling * torch.outer(grad_times_b, x)
                    grad_up_lora_b[expert] += scaling * torch.outer(grad, u)

    return {
        "grad_input": grad_input.to(torch.bfloat16),
        "grad_gate_lora_a": grad_gate_lora_a.to(torch.bfloat16),
        "grad_gate_lora_b": grad_gate_lora_b.to(torch.bfloat16),
        "grad_up_lora_a": grad_up_lora_a.to(torch.bfloat16),
        "grad_up_lora_b": grad_up_lora_b.to(torch.bfloat16),
    }


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    diff = (actual.float() - expected.float()).abs()
    print(f"{name}: max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=False)


def assert_nonzero(name: str, value: torch.Tensor):
    max_abs = value.float().abs().max().item()
    print(f"{name}: observed_max_abs={max_abs:.6g}")
    if max_abs == 0.0:
        raise AssertionError(f"{name} did not change")


def print_error_attribution(prefix: str, rows: List[Tuple[str, torch.Tensor, torch.Tensor]]):
    print(f"{prefix}:")
    for name, actual, expected in rows:
        diff = (actual.float() - expected.float()).abs()
        print(f"  {name}: max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}")


def assert_forward_cache_empty(moe, label: str = "official_backward_cache_pop"):
    try:
        moe.debug_cache_summary()
    except RuntimeError as exc:
        if "forward cache is empty" not in str(exc):
            raise
        print(f"{label}: observed empty cache")
        return
    raise AssertionError(f"{label} did not consume the forward cache")


def _run_git(args: List[str], cwd: Path, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"
    return result.stdout.strip()


def collect_git_metadata() -> Dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    diff = _run_git(["diff"], repo)
    patch_id = ""
    if diff and not diff.startswith("UNAVAILABLE:"):
        patch_id = _run_git(["patch-id", "--stable"], repo, input_text=diff)
    return {
        "repo": str(repo),
        "head": _run_git(["rev-parse", "HEAD"], repo),
        "status_short": _run_git(["status", "--short"], repo),
        "diff_stat": _run_git(["diff", "--stat"], repo),
        "patch_id": patch_id,
    }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def summarize_samples(samples: List[float]) -> Dict[str, object]:
    return {
        "samples_ms": samples,
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": percentile(samples, 50.0),
        "p90_ms": percentile(samples, 90.0),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_backward_once(
    moe,
    grad_output: torch.Tensor,
    lora: Dict[str, torch.Tensor],
    qlen: int,
    k: int,
    with_lora_grads: bool,
) -> None:
    grad_input = torch.empty((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
    grad_weights = torch.empty((qlen, k), dtype=torch.float32)
    if with_lora_grads:
        grad_gate_lora_a = torch.empty_like(lora["gate_a"])
        grad_gate_lora_b = torch.empty_like(lora["gate_b"])
        grad_up_lora_a = torch.empty_like(lora["up_a"])
        grad_up_lora_b = torch.empty_like(lora["up_b"])
        grad_down_lora_a = torch.empty_like(lora["down_a"])
        grad_down_lora_b = torch.empty_like(lora["down_b"])
        moe.backward(
            grad_output.data_ptr(),
            grad_input.data_ptr(),
            grad_gate_lora_a.data_ptr(),
            grad_gate_lora_b.data_ptr(),
            grad_up_lora_a.data_ptr(),
            grad_up_lora_b.data_ptr(),
            grad_down_lora_a.data_ptr(),
            grad_down_lora_b.data_ptr(),
            grad_weights.data_ptr(),
        )
    else:
        moe.backward(
            grad_output.data_ptr(),
            grad_input.data_ptr(),
            0,
            0,
            0,
            0,
            0,
            0,
            grad_weights.data_ptr(),
        )


def run_bench_case(name: str, warmup: int, repeat: int, fn) -> Dict[str, object]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    result = {"case": name}
    result.update(summarize_samples(samples))
    return result


def derive_2x2_metrics(results_by_case: Dict[str, Dict[str, object]], field: str) -> Dict[str, object]:
    forward_no_lora = float(results_by_case["forward_no_lora"][field])
    forward_lora = float(results_by_case["forward_lora"][field])
    forward_backward_no_lora = float(results_by_case["forward_backward_no_lora"][field])
    forward_backward_lora = float(results_by_case["forward_backward_lora"][field])
    backward_no_lora = forward_backward_no_lora - forward_no_lora
    backward_lora = forward_backward_lora - forward_lora
    return {
        "summary_field": field,
        "forward_no_lora_ms": forward_no_lora,
        "forward_lora_ms": forward_lora,
        "forward_lora_over_no_lora": forward_lora / forward_no_lora if forward_no_lora else None,
        "backward_no_lora_ms": backward_no_lora,
        "backward_lora_ms": backward_lora,
        "backward_no_lora_over_forward_no_lora": backward_no_lora / forward_no_lora if forward_no_lora else None,
        "backward_lora_over_forward_lora": backward_lora / forward_lora if forward_lora else None,
        "backward_lora_over_no_lora": backward_lora / backward_no_lora if backward_no_lora else None,
        "backward_definition": "(forward_backward - forward), compared lora/no_lora",
    }


def run_bench_2x2(args) -> None:
    torch.manual_seed(0)
    qlen = args.qlen
    k = 2
    max_len = qlen
    scaling = 1.0

    cpuinfer = make_cpuinfer(args.threads, args.tp_count)
    weights = make_weights(args.expert_num)
    lora = make_lora(args.expert_num, args.rank, scale=0.03, seed_offset=0)
    zero_lora = zero_lora_like(lora)

    moe_skip = make_moe(
        cpuinfer,
        weights,
        zero_lora,
        None,
        args.expert_num,
        max_len,
        args.rank,
        scaling,
        skip_lora=True,
    )
    moe_lora = make_moe(cpuinfer, weights, lora, None, args.expert_num, max_len, args.rank, scaling)

    x = _bf16_randn((qlen, HIDDEN_SIZE), 0.05, 300)
    expert_ids, route_weights = make_routing(qlen, args.expert_num, k)
    qlen_tensor = torch.tensor([qlen], dtype=torch.int32)
    grad_output = _bf16_randn((qlen, HIDDEN_SIZE), 0.02, 400)

    cases = [
        (
            "forward_no_lora",
            lambda: run_forward(moe_skip, qlen_tensor, expert_ids, route_weights, x, save_for_backward=False),
        ),
        (
            "forward_lora",
            lambda: run_forward(moe_lora, qlen_tensor, expert_ids, route_weights, x, save_for_backward=False),
        ),
        (
            "forward_backward_no_lora",
            lambda: (
                run_forward(moe_skip, qlen_tensor, expert_ids, route_weights, x, save_for_backward=True),
                run_backward_once(moe_skip, grad_output, lora, qlen, k, with_lora_grads=False),
            ),
        ),
        (
            "forward_backward_lora",
            lambda: (
                run_forward(moe_lora, qlen_tensor, expert_ids, route_weights, x, save_for_backward=True),
                run_backward_once(moe_lora, grad_output, lora, qlen, k, with_lora_grads=True),
            ),
        ),
    ]

    metadata = {
        "record_type": "metadata",
        "schema": "k2_sft_micro_2x2_v1",
        "git": collect_git_metadata(),
        "shape": {
            "qlen": args.qlen,
            "expert_num": args.expert_num,
            "rank": args.rank,
            "threads": args.threads,
            "tp_count": args.tp_count,
            "k": k,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
        },
        "repeat": args.bench_repeat,
        "warmup": args.bench_warmup,
        "axes": {
            "lora_axis": ["no_lora", "lora"],
            "pass_axis": ["forward", "forward_backward"],
        },
        "derived_metric_definitions": {
            "forward_lora_over_no_lora": "forward_lora / forward_no_lora",
            "backward_no_lora_over_forward_no_lora": "(forward_backward_no_lora - forward_no_lora) / "
            "forward_no_lora",
            "backward_lora_over_forward_lora": "(forward_backward_lora - forward_lora) / forward_lora",
            "backward_lora_over_no_lora": "(forward_backward_lora - forward_lora) / "
            "(forward_backward_no_lora - forward_no_lora)",
        },
        "cases": [case_name for case_name, _ in cases],
        "time_unit": "ms",
    }

    out_path = Path(args.bench_2x2_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        print(json.dumps(metadata, ensure_ascii=False))
        results_by_case = {}
        for case_name, fn in cases:
            case_result = run_bench_case(case_name, args.bench_warmup, args.bench_repeat, fn)
            results_by_case[case_name] = case_result
            record = {"record_type": "bench_case", **case_result}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(record, ensure_ascii=False))
        for field in ("p50_ms", "p90_ms", "mean_ms"):
            record = {"record_type": "derived_metrics", **derive_2x2_metrics(results_by_case, field)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(record, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="K2 SFT LoRA forward cache BF16 reference smoke.")
    parser.add_argument("--expert-num", type=int, default=2)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--qlen", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tp-count", type=int, default=1)
    parser.add_argument("--multi-layer-smoke", action="store_true")
    parser.add_argument("--multi-layer-count", type=int, default=2)
    parser.add_argument("--bench-2x2-jsonl", default=None)
    parser.add_argument("--bench-repeat", type=int, default=5)
    parser.add_argument("--bench-warmup", type=int, default=1)
    args = parser.parse_args()

    if args.expert_num < 2:
        raise ValueError("--expert-num must be >= 2 for routing order validation")
    if args.qlen < 1:
        raise ValueError("--qlen must be >= 1")
    if args.tp_count < 1:
        raise ValueError("--tp-count must be >= 1")
    if INTERMEDIATE_SIZE % args.tp_count != 0:
        raise ValueError("--tp-count must divide INTERMEDIATE_SIZE")
    if args.multi_layer_smoke and args.multi_layer_count < 2:
        raise ValueError("--multi-layer-count must be >= 2 when --multi-layer-smoke is set")
    if args.bench_repeat < 1:
        raise ValueError("--bench-repeat must be >= 1")
    if args.bench_warmup < 0:
        raise ValueError("--bench-warmup must be >= 0")
    if args.bench_2x2_jsonl:
        run_bench_2x2(args)
        return

    torch.manual_seed(0)
    qlen = args.qlen
    k = 2
    max_len = qlen
    scaling = 1.0

    cpuinfer = make_cpuinfer(args.threads)
    weights = make_weights(args.expert_num)
    lora = make_lora(args.expert_num, args.rank, scale=0.03, seed_offset=0)
    zero_lora = zero_lora_like(lora)
    no_down_lora = zero_lora_like(lora, zero_down_only=True)

    base_moe = make_base_k2_moe(cpuinfer, weights, args.expert_num, max_len)
    moe_zero = make_moe(cpuinfer, weights, zero_lora, None, args.expert_num, max_len, args.rank, scaling)
    moe_lora = make_moe(cpuinfer, weights, lora, None, args.expert_num, max_len, args.rank, scaling)
    moe_no_down_lora = make_moe(cpuinfer, weights, no_down_lora, None, args.expert_num, max_len, args.rank, scaling)
    assert_packed_weight_contract(moe_lora, weights, args.expert_num)
    packed_weight_ref = dequant_kgroup_weights(weights, args.expert_num)

    x = _bf16_randn((qlen, HIDDEN_SIZE), 0.05, 300)
    expert_ids, route_weights = make_routing(qlen, args.expert_num, k)
    qlen_tensor = torch.tensor([qlen], dtype=torch.int32)

    base_output = run_base_forward(cpuinfer, base_moe, qlen_tensor, expert_ids, route_weights, x)
    output_zero_no_cache = run_forward(moe_zero, qlen_tensor, expert_ids, route_weights, x, save_for_backward=False)
    assert_close("zero_lora_no_cache_vs_k2_base", output_zero_no_cache, base_output, atol=5e-3, rtol=5e-3)

    output_zero = run_forward(moe_zero, qlen_tensor, expert_ids, route_weights, x, save_for_backward=True)
    cache_zero = copy_forward_cache(moe_zero, args.rank)
    assert_close("zero_lora_cache_vs_no_cache", output_zero, output_zero_no_cache, atol=0.0, rtol=0.0)

    output_lora_no_cache = run_forward(moe_lora, qlen_tensor, expert_ids, route_weights, x, save_for_backward=False)
    output_lora = run_forward(moe_lora, qlen_tensor, expert_ids, route_weights, x, save_for_backward=True)
    cache_lora = copy_forward_cache(moe_lora, args.rank)
    assert_close("lora_cache_vs_no_cache", output_lora, output_lora_no_cache, atol=0.0, rtol=0.0)

    run_forward(moe_no_down_lora, qlen_tensor, expert_ids, route_weights, x, save_for_backward=True)
    cache_no_down_lora = copy_forward_cache(moe_no_down_lora, args.rank)

    expected_active, expected_counts, expected_meta = expected_routing_meta(expert_ids, args.expert_num)
    assert cache_lora.qlen == qlen
    assert cache_lora.k == k
    assert cache_lora.active_experts == expected_active
    assert [cache_lora.local_counts[i] for i in expected_active] == [expected_counts[i] for i in expected_active]
    assert_close("input_cache", cache_lora.input, x, atol=0.0, rtol=0.0)

    packed_x, meta = pack_input_by_expert(x, expert_ids, cache_lora.active_experts)
    assert meta == expected_meta

    gate_delta = lora_linear_by_expert(
        packed_x, cache_lora.active_experts, cache_lora.local_counts, lora["gate_a"], lora["gate_b"], scaling
    )
    up_delta = lora_linear_by_expert(
        packed_x, cache_lora.active_experts, cache_lora.local_counts, lora["up_a"], lora["up_b"], scaling
    )
    actual_gate_delta = cache_lora.gate.float() - cache_zero.gate.float()
    actual_up_delta = cache_lora.up.float() - cache_zero.up.float()
    assert_nonzero("gate_lora_delta_presence", actual_gate_delta)
    assert_nonzero("up_lora_delta_presence", actual_up_delta)
    assert_close("gate_lora_delta", actual_gate_delta, gate_delta, atol=0.03, rtol=0.08)
    assert_close("up_lora_delta", actual_up_delta, up_delta, atol=0.03, rtol=0.08)

    intermediate_ref = (torch.nn.functional.silu(cache_lora.gate.float()) * cache_lora.up.float()).to(torch.bfloat16)
    assert_close("intermediate_cache", cache_lora.intermediate, intermediate_ref, atol=0.03, rtol=0.08)

    down_u_ref = down_lora_u_ref(cache_lora, lora)
    assert_close("down_lora_u_cache", cache_lora.down_lora_u, down_u_ref, atol=0.05, rtol=0.08)

    down_delta = lora_linear_by_expert(
        cache_lora.intermediate,
        cache_lora.active_experts,
        cache_lora.local_counts,
        lora["down_a"],
        lora["down_b"],
        scaling,
    )
    actual_down_delta = cache_lora.down.float() - cache_no_down_lora.down.float()
    assert_nonzero("down_lora_delta_presence", actual_down_delta)
    assert_close("down_lora_delta", actual_down_delta, down_delta, atol=0.08, rtol=0.12)

    merged_from_cache = remerge_forward_cache_cpp(moe_lora, qlen)
    assert_close(
        "cpp_remerged_output_from_down_cache",
        output_lora,
        merged_from_cache.to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )

    print(
        "forward_cache_contract_passed: "
        f"qlen={qlen}, experts={args.expert_num}, active={cache_lora.active_experts}, "
        f"total_cached_tokens={cache_lora.gate.shape[0]}"
    )

    grad_output = _bf16_randn((qlen, HIDDEN_SIZE), 0.02, 400)
    sample_grad_input = torch.empty((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
    sample_grad_weights = torch.empty((qlen, k), dtype=torch.float32)
    moe_lora.debug_backward_sample(
        grad_output.data_ptr(),
        sample_grad_input.data_ptr(),
        sample_grad_weights.data_ptr(),
    )
    grad_weights_ref = grad_weights_from_down_cache(cache_lora, expert_ids, grad_output)
    assert_close(
        "backward_sample_grad_input_zero",
        sample_grad_input,
        torch.zeros_like(sample_grad_input),
        atol=0.0,
        rtol=0.0,
    )
    assert_close("backward_sample_grad_weights", sample_grad_weights, grad_weights_ref, atol=1e-4, rtol=1e-4)

    total_tokens = cache_lora.down.shape[0]
    sample_grad_down = torch.empty((total_tokens, HIDDEN_SIZE), dtype=torch.bfloat16)
    sample_grad_intermediate = torch.empty((total_tokens, INTERMEDIATE_SIZE), dtype=torch.bfloat16)
    sample_grad_down_lora_a = torch.empty_like(lora["down_a"])
    sample_grad_down_lora_b = torch.empty_like(lora["down_b"])
    moe_lora.debug_backward_down_sample(
        grad_output.data_ptr(),
        sample_grad_down.data_ptr(),
        sample_grad_intermediate.data_ptr(),
        sample_grad_down_lora_a.data_ptr(),
        sample_grad_down_lora_b.data_ptr(),
    )
    down_ref = backward_down_ref(
        cache_lora,
        expert_ids,
        route_weights,
        grad_output,
        lora,
        packed_weight_ref,
        args.expert_num,
        scaling,
    )
    assert_close("backward_down_grad_down", sample_grad_down, down_ref["grad_down"], atol=0.0, rtol=0.0)
    assert_close(
        "backward_down_grad_intermediate",
        sample_grad_intermediate,
        down_ref["grad_intermediate"],
        atol=0.02,
        rtol=0.08,
    )
    assert_close("backward_down_lora_a", sample_grad_down_lora_a, down_ref["grad_down_lora_a"], atol=0.01, rtol=0.08)
    assert_close("backward_down_lora_b", sample_grad_down_lora_b, down_ref["grad_down_lora_b"], atol=0.01, rtol=0.08)

    sample_grad_gate = torch.empty_like(cache_lora.gate)
    sample_grad_up = torch.empty_like(cache_lora.up)
    moe_lora.debug_backward_activation_sample(
        grad_output.data_ptr(),
        sample_grad_intermediate.data_ptr(),
        sample_grad_gate.data_ptr(),
        sample_grad_up.data_ptr(),
    )
    activation_ref = activation_backward_ref(cache_lora, down_ref["grad_intermediate"])
    assert_close(
        "backward_activation_grad_intermediate",
        sample_grad_intermediate,
        down_ref["grad_intermediate"],
        atol=0.02,
        rtol=0.08,
    )
    assert_close("backward_activation_grad_gate", sample_grad_gate, activation_ref["grad_gate"], atol=0.01, rtol=0.08)
    assert_close("backward_activation_grad_up", sample_grad_up, activation_ref["grad_up"], atol=0.01, rtol=0.08)

    sample_gate_up_grad_input = torch.empty((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
    sample_grad_gate_lora_a = torch.empty_like(lora["gate_a"])
    sample_grad_gate_lora_b = torch.empty_like(lora["gate_b"])
    sample_grad_up_lora_a = torch.empty_like(lora["up_a"])
    sample_grad_up_lora_b = torch.empty_like(lora["up_b"])
    moe_lora.debug_backward_gate_up_sample(
        grad_output.data_ptr(),
        sample_gate_up_grad_input.data_ptr(),
        sample_grad_gate_lora_a.data_ptr(),
        sample_grad_gate_lora_b.data_ptr(),
        sample_grad_up_lora_a.data_ptr(),
        sample_grad_up_lora_b.data_ptr(),
    )
    gate_up_ref = backward_gate_up_ref(
        cache_lora,
        expert_ids,
        activation_ref["grad_gate"],
        activation_ref["grad_up"],
        lora,
        packed_weight_ref,
        args.expert_num,
        scaling,
    )
    assert_close(
        "backward_gate_up_grad_input",
        sample_gate_up_grad_input,
        gate_up_ref["grad_input"],
        atol=0.02,
        rtol=0.08,
    )
    assert_close(
        "backward_gate_lora_a",
        sample_grad_gate_lora_a,
        gate_up_ref["grad_gate_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "backward_gate_lora_b",
        sample_grad_gate_lora_b,
        gate_up_ref["grad_gate_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close("backward_up_lora_a", sample_grad_up_lora_a, gate_up_ref["grad_up_lora_a"], atol=0.01, rtol=0.08)
    assert_close("backward_up_lora_b", sample_grad_up_lora_b, gate_up_ref["grad_up_lora_b"], atol=0.01, rtol=0.08)

    official_grad_input = torch.empty((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
    official_grad_gate_lora_a = torch.empty_like(lora["gate_a"])
    official_grad_gate_lora_b = torch.empty_like(lora["gate_b"])
    official_grad_up_lora_a = torch.empty_like(lora["up_a"])
    official_grad_up_lora_b = torch.empty_like(lora["up_b"])
    official_grad_down_lora_a = torch.empty_like(lora["down_a"])
    official_grad_down_lora_b = torch.empty_like(lora["down_b"])
    official_grad_weights = torch.empty((qlen, k), dtype=torch.float32)
    moe_lora.backward(
        grad_output.data_ptr(),
        official_grad_input.data_ptr(),
        official_grad_gate_lora_a.data_ptr(),
        official_grad_gate_lora_b.data_ptr(),
        official_grad_up_lora_a.data_ptr(),
        official_grad_up_lora_b.data_ptr(),
        official_grad_down_lora_a.data_ptr(),
        official_grad_down_lora_b.data_ptr(),
        official_grad_weights.data_ptr(),
    )
    assert_close(
        "official_backward_grad_input",
        official_grad_input,
        gate_up_ref["grad_input"],
        atol=0.02,
        rtol=0.08,
    )
    assert_close(
        "official_backward_grad_weights",
        official_grad_weights,
        grad_weights_ref,
        atol=1e-4,
        rtol=1e-4,
    )
    assert_close(
        "official_backward_gate_lora_a",
        official_grad_gate_lora_a,
        gate_up_ref["grad_gate_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "official_backward_gate_lora_b",
        official_grad_gate_lora_b,
        gate_up_ref["grad_gate_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "official_backward_up_lora_a",
        official_grad_up_lora_a,
        gate_up_ref["grad_up_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "official_backward_up_lora_b",
        official_grad_up_lora_b,
        gate_up_ref["grad_up_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "official_backward_down_lora_a",
        official_grad_down_lora_a,
        down_ref["grad_down_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "official_backward_down_lora_b",
        official_grad_down_lora_b,
        down_ref["grad_down_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_forward_cache_empty(moe_lora)

    run_forward(moe_lora, qlen_tensor, expert_ids, route_weights, x)
    task_grad_input = torch.empty((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
    task_grad_gate_lora_a = torch.empty_like(lora["gate_a"])
    task_grad_gate_lora_b = torch.empty_like(lora["gate_b"])
    task_grad_up_lora_a = torch.empty_like(lora["up_a"])
    task_grad_up_lora_b = torch.empty_like(lora["up_b"])
    task_grad_down_lora_a = torch.empty_like(lora["down_a"])
    task_grad_down_lora_b = torch.empty_like(lora["down_b"])
    task_grad_weights = torch.empty((qlen, k), dtype=torch.float32)
    cpuinfer.submit(
        moe_lora.backward_task(
            grad_output.data_ptr(),
            task_grad_input.data_ptr(),
            task_grad_gate_lora_a.data_ptr(),
            task_grad_gate_lora_b.data_ptr(),
            task_grad_up_lora_a.data_ptr(),
            task_grad_up_lora_b.data_ptr(),
            task_grad_down_lora_a.data_ptr(),
            task_grad_down_lora_b.data_ptr(),
            task_grad_weights.data_ptr(),
        )
    )
    cpuinfer.sync()
    assert_close("official_backward_task_grad_input", task_grad_input, gate_up_ref["grad_input"], atol=0.02, rtol=0.08)
    assert_close("official_backward_task_grad_weights", task_grad_weights, grad_weights_ref, atol=1e-4, rtol=1e-4)
    assert_close("official_backward_task_gate_lora_a", task_grad_gate_lora_a, gate_up_ref["grad_gate_lora_a"], atol=0.01, rtol=0.08)
    assert_close("official_backward_task_gate_lora_b", task_grad_gate_lora_b, gate_up_ref["grad_gate_lora_b"], atol=0.01, rtol=0.08)
    assert_close("official_backward_task_up_lora_a", task_grad_up_lora_a, gate_up_ref["grad_up_lora_a"], atol=0.01, rtol=0.08)
    assert_close("official_backward_task_up_lora_b", task_grad_up_lora_b, gate_up_ref["grad_up_lora_b"], atol=0.01, rtol=0.08)
    assert_close("official_backward_task_down_lora_a", task_grad_down_lora_a, down_ref["grad_down_lora_a"], atol=0.01, rtol=0.08)
    assert_close("official_backward_task_down_lora_b", task_grad_down_lora_b, down_ref["grad_down_lora_b"], atol=0.01, rtol=0.08)
    assert_forward_cache_empty(moe_lora)

    wrapper, _ = make_wrapper(weights, lora, None, args.expert_num, max_len, args.rank, scaling, args.threads)
    wrapper_output = wrapper.forward(x, expert_ids, route_weights, save_for_backward=True)
    assert_close("wrapper_forward_output", wrapper_output, output_lora, atol=0.03, rtol=0.08)
    if wrapper._cache_depth != 1:
        raise AssertionError(f"wrapper forward cache depth mismatch: expected 1, got {wrapper._cache_depth}")

    wrapper_grad_input, wrapper_grad_weights = wrapper.backward(grad_output)
    assert_close("wrapper_backward_grad_input", wrapper_grad_input, gate_up_ref["grad_input"], atol=0.02, rtol=0.08)
    assert_close("wrapper_backward_grad_weights", wrapper_grad_weights, grad_weights_ref, atol=1e-4, rtol=1e-4)
    assert_close("wrapper_backward_gate_lora_a", wrapper.grad_gate_lora_a, gate_up_ref["grad_gate_lora_a"], atol=0.01, rtol=0.08)
    assert_close("wrapper_backward_gate_lora_b", wrapper.grad_gate_lora_b, gate_up_ref["grad_gate_lora_b"], atol=0.01, rtol=0.08)
    assert_close("wrapper_backward_up_lora_a", wrapper.grad_up_lora_a, gate_up_ref["grad_up_lora_a"], atol=0.01, rtol=0.08)
    assert_close("wrapper_backward_up_lora_b", wrapper.grad_up_lora_b, gate_up_ref["grad_up_lora_b"], atol=0.01, rtol=0.08)
    assert_close("wrapper_backward_down_lora_a", wrapper.grad_down_lora_a, down_ref["grad_down_lora_a"], atol=0.01, rtol=0.08)
    assert_close("wrapper_backward_down_lora_b", wrapper.grad_down_lora_b, down_ref["grad_down_lora_b"], atol=0.01, rtol=0.08)
    if wrapper._cache_depth != 0:
        raise AssertionError(f"wrapper backward cache depth mismatch: expected 0, got {wrapper._cache_depth}")
    assert_forward_cache_empty(wrapper.moe, "wrapper_backward_cache_pop")

    from kt_kernel.sft.arch import MOEArchConfig
    from kt_kernel.sft.layer import KTMoELayerWrapper

    layer_wrapper, layer_grad_lora = make_wrapper(
        weights, lora, None, args.expert_num, max_len, args.rank, scaling, args.threads
    )
    moe_config = MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=args.expert_num,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts_per_tok=k,
        has_shared_experts=False,
        router_type="linear",
    )
    original_moe = DummyOriginalMoE(args.expert_num, expert_ids, route_weights)
    kt_layer = KTMoELayerWrapper(
        original_moe=original_moe,
        wrapper=layer_wrapper,
        lora_params=None,
        moe_config=moe_config,
        hidden_size=HIDDEN_SIZE,
        layer_idx=0,
    )
    kt_layer.train()
    peft_lora_modules = make_peft_lora_modules(lora, layer_grad_lora, args.expert_num)
    kt_layer._peft_lora_modules = peft_lora_modules
    for expert_loras in peft_lora_modules.values():
        for lora_a, lora_b in expert_loras.values():
            lora_a.weight.grad = None
            lora_b.weight.grad = None

    layer_input = x.view(1, qlen, HIDDEN_SIZE).clone().detach().requires_grad_(True)
    layer_output = kt_layer(layer_input)
    assert_peft_grad_alias(peft_lora_modules, layer_grad_lora, args.expert_num)
    assert_close("layer_autograd_forward_output", layer_output.view(qlen, HIDDEN_SIZE), output_lora, atol=0.03, rtol=0.08)
    if layer_wrapper._cache_depth != 1:
        raise AssertionError(f"layer wrapper forward cache depth mismatch: expected 1, got {layer_wrapper._cache_depth}")

    layer_loss = (layer_output.view(qlen, HIDDEN_SIZE).float() * grad_output.float()).sum()
    layer_loss.backward()

    assert_close("layer_autograd_grad_input", layer_input.grad.view(qlen, HIDDEN_SIZE), gate_up_ref["grad_input"], atol=0.02, rtol=0.08)
    assert_peft_grad_alias(peft_lora_modules, layer_grad_lora, args.expert_num)
    assert_close(
        "layer_autograd_gate_lora_a_grad",
        stack_peft_grad(peft_lora_modules, args.expert_num, "gate_proj", 0),
        gate_up_ref["grad_gate_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "layer_autograd_gate_lora_b_grad",
        stack_peft_grad(peft_lora_modules, args.expert_num, "gate_proj", 1),
        gate_up_ref["grad_gate_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "layer_autograd_up_lora_a_grad",
        stack_peft_grad(peft_lora_modules, args.expert_num, "up_proj", 0),
        gate_up_ref["grad_up_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "layer_autograd_up_lora_b_grad",
        stack_peft_grad(peft_lora_modules, args.expert_num, "up_proj", 1),
        gate_up_ref["grad_up_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "layer_autograd_down_lora_a_grad",
        stack_peft_grad(peft_lora_modules, args.expert_num, "down_proj", 0),
        down_ref["grad_down_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "layer_autograd_down_lora_b_grad",
        stack_peft_grad(peft_lora_modules, args.expert_num, "down_proj", 1),
        down_ref["grad_down_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    if layer_wrapper._cache_depth != 0:
        raise AssertionError(f"layer wrapper backward cache depth mismatch: expected 0, got {layer_wrapper._cache_depth}")
    assert_forward_cache_empty(layer_wrapper.moe, "layer_autograd_cache_pop")

    from kt_kernel.sft.config import KTConfig
    from kt_kernel.sft.lora import kt_adapt_peft_lora
    from kt_kernel.sft.wrapper import wrap_moe_layers_with_kt_wrapper

    fake_model = FakeKGroupModel(
        expert_num=args.expert_num,
        expert_ids=expert_ids,
        route_weights=route_weights,
        weights=weights,
        shadow=None,
        lora=lora,
        max_len=max_len,
    )
    kt_config = KTConfig(
        kt_backend="AMXINT4_KGroup",
        kt_num_threads=args.threads,
        kt_tp_enabled=False,
        kt_model_max_length=max_len,
        kt_lora_rank=args.rank,
        kt_lora_alpha=scaling * args.rank,
        kt_max_cache_depth=1,
        kt_group_size=K_GROUP_SIZE,
        kt_zero_point=False,
        kt_share_backward_bb=False,
        kt_expert_checkpoint_path="/tmp/kgroup-shadow-should-be-ignored",
    )
    wrapped_layers = wrap_moe_layers_with_kt_wrapper(fake_model, kt_config)
    fake_model._kt_wrappers = wrapped_layers
    if len(wrapped_layers) != 1:
        raise AssertionError(f"wrap_moe_layers_with_kt_wrapper expected 1 wrapper, got {len(wrapped_layers)}")

    wrapped_layer = wrapped_layers[0]
    if wrapped_layer.wrapper.method != "AMXINT4_KGroup_SFT":
        raise AssertionError(f"wrapped method mismatch: {wrapped_layer.wrapper.method}")
    if wrapped_layer.wrapper.group_size != K_GROUP_SIZE or wrapped_layer.wrapper.zero_point:
        raise AssertionError(
            f"KGroup quant config mismatch: group_size={wrapped_layer.wrapper.group_size}, "
            f"zero_point={wrapped_layer.wrapper.zero_point}"
        )
    if not getattr(wrapped_layer.wrapper, "_weights_loaded", False):
        raise AssertionError("wrapped KGroup wrapper did not load weights")
    if any(
        getattr(wrapped_layer.wrapper, attr, None) is not None
        for attr in ("gate_bwd_shadow", "up_bwd_shadow", "down_bwd_shadow")
    ):
        raise AssertionError("KGroup wrapper unexpectedly retained BF16 shadow tensors")

    kt_adapt_peft_lora(fake_model)
    if not getattr(wrapped_layer.wrapper, "_lora_initialized", False):
        raise AssertionError("kt_adapt_peft_lora did not initialize wrapper LoRA weights")
    if wrapped_layer._peft_lora_modules is None:
        raise AssertionError("kt_adapt_peft_lora did not attach PEFT LoRA modules")

    fake_model.zero_grad(set_to_none=True)
    wrapped_input = x.view(1, qlen, HIDDEN_SIZE).clone().detach().requires_grad_(True)
    wrapped_output = wrapped_layer(wrapped_input)
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, {
        "gate_a": wrapped_layer.wrapper.grad_gate_lora_a,
        "gate_b": wrapped_layer.wrapper.grad_gate_lora_b,
        "up_a": wrapped_layer.wrapper.grad_up_lora_a,
        "up_b": wrapped_layer.wrapper.grad_up_lora_b,
        "down_a": wrapped_layer.wrapper.grad_down_lora_a,
        "down_b": wrapped_layer.wrapper.grad_down_lora_b,
    }, args.expert_num)
    assert_close("wrap_adapt_forward_output", wrapped_output.view(qlen, HIDDEN_SIZE), output_lora, atol=0.03, rtol=0.08)

    wrapped_loss = (wrapped_output.view(qlen, HIDDEN_SIZE).float() * grad_output.float()).sum()
    wrapped_loss.backward()

    assert_close("wrap_adapt_grad_input", wrapped_input.grad.view(qlen, HIDDEN_SIZE), gate_up_ref["grad_input"], atol=0.02, rtol=0.08)
    assert_close(
        "wrap_adapt_gate_lora_a_grad",
        stack_peft_grad(wrapped_layer._peft_lora_modules, args.expert_num, "gate_proj", 0),
        gate_up_ref["grad_gate_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "wrap_adapt_gate_lora_b_grad",
        stack_peft_grad(wrapped_layer._peft_lora_modules, args.expert_num, "gate_proj", 1),
        gate_up_ref["grad_gate_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "wrap_adapt_up_lora_a_grad",
        stack_peft_grad(wrapped_layer._peft_lora_modules, args.expert_num, "up_proj", 0),
        gate_up_ref["grad_up_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "wrap_adapt_up_lora_b_grad",
        stack_peft_grad(wrapped_layer._peft_lora_modules, args.expert_num, "up_proj", 1),
        gate_up_ref["grad_up_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "wrap_adapt_down_lora_a_grad",
        stack_peft_grad(wrapped_layer._peft_lora_modules, args.expert_num, "down_proj", 0),
        down_ref["grad_down_lora_a"],
        atol=0.01,
        rtol=0.08,
    )
    assert_close(
        "wrap_adapt_down_lora_b_grad",
        stack_peft_grad(wrapped_layer._peft_lora_modules, args.expert_num, "down_proj", 1),
        down_ref["grad_down_lora_b"],
        atol=0.01,
        rtol=0.08,
    )
    if wrapped_layer.wrapper._cache_depth != 0:
        raise AssertionError(
            f"wrap/adapt wrapper backward cache depth mismatch: expected 0, got {wrapped_layer.wrapper._cache_depth}"
        )
    assert_forward_cache_empty(wrapped_layer.wrapper.moe, "wrap_adapt_cache_pop")
    pre_step_output = wrapped_output.detach().view(qlen, HIDDEN_SIZE).clone()

    from kt_kernel.sft.lora import update_kt_lora_pointers

    optimizer_params = [param for param in fake_model.parameters() if param.requires_grad]
    if not optimizer_params:
        raise AssertionError("wrap/adapt optimizer smoke found no trainable LoRA parameters")
    before_step_weights = {
        name: weight.detach().clone()
        for name, weight in iter_peft_lora_weights(wrapped_layer._peft_lora_modules)
    }
    optimizer = torch.optim.SGD(optimizer_params, lr=64.0)
    optimizer.step()
    step_delta = max_peft_weight_delta(before_step_weights, wrapped_layer._peft_lora_modules)
    assert_nonzero("wrap_adapt_optimizer_step_lora_delta", step_delta)

    update_kt_lora_pointers(fake_model)
    if not wrapped_layer._lora_pointers_dirty:
        raise AssertionError("update_kt_lora_pointers did not mark wrapped layer dirty")

    optimizer.zero_grad(set_to_none=True)
    post_step_input = x.view(1, qlen, HIDDEN_SIZE).clone().detach().requires_grad_(True)
    post_step_output = wrapped_layer(post_step_input)
    if wrapped_layer._lora_pointers_dirty:
        raise AssertionError("wrapped layer did not consume dirty LoRA pointer flag on next forward")
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, {
        "gate_a": wrapped_layer.wrapper.grad_gate_lora_a,
        "gate_b": wrapped_layer.wrapper.grad_gate_lora_b,
        "up_a": wrapped_layer.wrapper.grad_up_lora_a,
        "up_b": wrapped_layer.wrapper.grad_up_lora_b,
        "down_a": wrapped_layer.wrapper.grad_down_lora_a,
        "down_b": wrapped_layer.wrapper.grad_down_lora_b,
    }, args.expert_num)
    post_step_output_delta = (post_step_output.detach().view(qlen, HIDDEN_SIZE).float() - pre_step_output.float()).abs().max()
    assert_nonzero("wrap_adapt_post_step_forward_delta", post_step_output_delta)

    post_step_loss = (post_step_output.view(qlen, HIDDEN_SIZE).float() * grad_output.float()).sum()
    post_step_loss.backward()
    if post_step_input.grad is None:
        raise AssertionError("wrap/adapt post-step backward did not produce input grad")
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, {
        "gate_a": wrapped_layer.wrapper.grad_gate_lora_a,
        "gate_b": wrapped_layer.wrapper.grad_gate_lora_b,
        "up_a": wrapped_layer.wrapper.grad_up_lora_a,
        "up_b": wrapped_layer.wrapper.grad_up_lora_b,
        "down_a": wrapped_layer.wrapper.grad_down_lora_a,
        "down_b": wrapped_layer.wrapper.grad_down_lora_b,
    }, args.expert_num)
    if wrapped_layer.wrapper._cache_depth != 0:
        raise AssertionError(
            f"wrap/adapt post-step wrapper backward cache depth mismatch: expected 0, got {wrapped_layer.wrapper._cache_depth}"
        )
    assert_forward_cache_empty(wrapped_layer.wrapper.moe, "wrap_adapt_post_step_cache_pop")

    bad_tp_model = FakeKGroupModel(
        expert_num=args.expert_num,
        expert_ids=expert_ids,
        route_weights=route_weights,
        weights=weights,
        shadow=None,
        lora=lora,
        max_len=max_len,
    )
    bad_tp_config = KTConfig(
        kt_backend="AMXINT4_KGroup",
        kt_num_threads=args.threads * 2,
        kt_tp_enabled=True,
        kt_threadpool_count=3,
        kt_model_max_length=max_len,
        kt_lora_rank=args.rank,
        kt_lora_alpha=scaling * args.rank,
        kt_max_cache_depth=1,
        kt_group_size=K_GROUP_SIZE,
        kt_zero_point=False,
        kt_share_backward_bb=False,
    )
    try:
        wrap_moe_layers_with_kt_wrapper(bad_tp_model, bad_tp_config)
    except RuntimeError as exc:
        if "intermediate_size divisible by kt_threadpool_count" not in str(exc):
            raise AssertionError(f"unexpected KGroup TP shape error: {exc}") from exc
    else:
        raise AssertionError("KGroup invalid TP wrapper construction unexpectedly succeeded")

    if args.multi_layer_smoke:
        multi_model = FakeKGroupModel(
            expert_num=args.expert_num,
            expert_ids=expert_ids,
            route_weights=route_weights,
            weights=weights,
            shadow=None,
            lora=lora,
            max_len=max_len,
            num_layers=args.multi_layer_count,
        )
        multi_config = KTConfig(
            kt_backend="AMXINT4_KGroup",
            kt_num_threads=args.threads,
            kt_tp_enabled=False,
            kt_model_max_length=max_len,
            kt_lora_rank=args.rank,
            kt_lora_alpha=scaling * args.rank,
            kt_max_cache_depth=1,
            kt_group_size=K_GROUP_SIZE,
            kt_zero_point=False,
            kt_share_backward_bb=False,
        )
        multi_layers = wrap_moe_layers_with_kt_wrapper(multi_model, multi_config)
        multi_model._kt_wrappers = multi_layers
        if len(multi_layers) != args.multi_layer_count:
            raise AssertionError(
                f"multi-layer wrap expected {args.multi_layer_count} wrappers, got {len(multi_layers)}"
            )

        repack_calls = {"submit": 0, "wait": 0}

        def unexpected_submit_repack():
            repack_calls["submit"] += 1
            raise AssertionError("K2 multi-layer smoke should not submit async repack with share_backward_bb=False")

        def unexpected_wait_repack():
            repack_calls["wait"] += 1
            raise AssertionError("K2 multi-layer smoke should not wait async repack with share_backward_bb=False")

        for layer_idx, layer in enumerate(multi_layers):
            wrapper_impl = layer.wrapper
            expected_next = None if layer_idx == 0 else multi_layers[layer_idx - 1].wrapper
            actual_next = getattr(wrapper_impl, "_next_backward_wrapper", None)
            if actual_next is not expected_next:
                raise AssertionError(f"multi-layer wrapper link mismatch at layer {layer_idx}")
            if getattr(wrapper_impl, "share_backward_bb", None):
                raise AssertionError("K2 multi-layer smoke must keep share_backward_bb disabled")
            wrapper_impl.submit_backward_repack = unexpected_submit_repack
            wrapper_impl.wait_backward_repack = unexpected_wait_repack

        kt_adapt_peft_lora(multi_model)
        update_kt_lora_pointers(multi_model)
        if not all(layer._lora_pointers_dirty for layer in multi_layers):
            raise AssertionError("update_kt_lora_pointers did not mark all multi-layer wrappers dirty")

        multi_model.train()
        multi_model.zero_grad(set_to_none=True)
        multi_input = x.view(1, qlen, HIDDEN_SIZE).clone().detach().requires_grad_(True)
        multi_output = multi_model(multi_input)
        if tuple(multi_output.shape) != (1, qlen, HIDDEN_SIZE):
            raise AssertionError(f"multi-layer output shape mismatch: {tuple(multi_output.shape)}")
        if any(layer._lora_pointers_dirty for layer in multi_layers):
            raise AssertionError("multi-layer forward did not consume all dirty LoRA pointer flags")

        multi_loss = (multi_output.float() * grad_output.view(1, qlen, HIDDEN_SIZE).float()).sum()
        multi_loss.backward()
        if multi_input.grad is None:
            raise AssertionError("multi-layer backward did not produce input grad")
        assert_nonzero("multi_layer_input_grad", multi_input.grad)

        for layer_idx, layer in enumerate(multi_layers):
            layer_grad_lora = {
                "gate_a": layer.wrapper.grad_gate_lora_a,
                "gate_b": layer.wrapper.grad_gate_lora_b,
                "up_a": layer.wrapper.grad_up_lora_a,
                "up_b": layer.wrapper.grad_up_lora_b,
                "down_a": layer.wrapper.grad_down_lora_a,
                "down_b": layer.wrapper.grad_down_lora_b,
            }
            assert_peft_grad_alias(layer._peft_lora_modules, layer_grad_lora, args.expert_num)
            assert_peft_any_grad_nonzero(f"multi_layer{layer_idx}_lora_grad", layer._peft_lora_modules)
            if layer.wrapper._cache_depth != 0:
                raise AssertionError(
                    f"multi-layer wrapper {layer_idx} cache depth mismatch: expected 0, got {layer.wrapper._cache_depth}"
                )
            assert_forward_cache_empty(layer.wrapper.moe, f"multi_layer{layer_idx}_cache_pop")

        if repack_calls["submit"] != 0 or repack_calls["wait"] != 0:
            raise AssertionError(f"unexpected async repack calls observed: {repack_calls}")
        print(f"multi_layer_link_smoke: layers={args.multi_layer_count}, async_repack_calls={repack_calls}")

    if args.tp_count > 1:
        cpuinfer_tp = make_cpuinfer(args.threads, args.tp_count)
        moe_tp = make_moe(cpuinfer_tp, weights, lora, None, args.expert_num, max_len, args.rank, scaling)
        tp_output = run_forward(moe_tp, qlen_tensor, expert_ids, route_weights, x)
        assert_close(f"tp{args.tp_count}_forward_output", tp_output, output_lora, atol=0.03, rtol=0.08)

        tp_grad_input = torch.zeros((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
        tp_grad_gate_lora_a = torch.zeros_like(lora["gate_a"])
        tp_grad_gate_lora_b = torch.zeros_like(lora["gate_b"])
        tp_grad_up_lora_a = torch.zeros_like(lora["up_a"])
        tp_grad_up_lora_b = torch.zeros_like(lora["up_b"])
        tp_grad_down_lora_a = torch.zeros_like(lora["down_a"])
        tp_grad_down_lora_b = torch.zeros_like(lora["down_b"])
        tp_grad_weights = torch.zeros((qlen, k), dtype=torch.float32)
        moe_tp.backward(
            grad_output.data_ptr(),
            tp_grad_input.data_ptr(),
            tp_grad_gate_lora_a.data_ptr(),
            tp_grad_gate_lora_b.data_ptr(),
            tp_grad_up_lora_a.data_ptr(),
            tp_grad_up_lora_b.data_ptr(),
            tp_grad_down_lora_a.data_ptr(),
            tp_grad_down_lora_b.data_ptr(),
            tp_grad_weights.data_ptr(),
        )
        assert_close(f"tp{args.tp_count}_backward_grad_input", tp_grad_input, gate_up_ref["grad_input"], atol=0.02, rtol=0.08)
        assert_close(f"tp{args.tp_count}_backward_grad_weights", tp_grad_weights, grad_weights_ref, atol=0.002, rtol=0.02)
        assert_close(f"tp{args.tp_count}_backward_gate_lora_a", tp_grad_gate_lora_a, gate_up_ref["grad_gate_lora_a"], atol=0.01, rtol=0.08)
        assert_close(f"tp{args.tp_count}_backward_gate_lora_b", tp_grad_gate_lora_b, gate_up_ref["grad_gate_lora_b"], atol=0.01, rtol=0.08)
        assert_close(f"tp{args.tp_count}_backward_up_lora_a", tp_grad_up_lora_a, gate_up_ref["grad_up_lora_a"], atol=0.01, rtol=0.08)
        assert_close(f"tp{args.tp_count}_backward_up_lora_b", tp_grad_up_lora_b, gate_up_ref["grad_up_lora_b"], atol=0.01, rtol=0.08)
        assert_close(f"tp{args.tp_count}_backward_down_lora_a", tp_grad_down_lora_a, down_ref["grad_down_lora_a"], atol=0.01, rtol=0.08)
        assert_close(f"tp{args.tp_count}_backward_down_lora_b", tp_grad_down_lora_b, down_ref["grad_down_lora_b"], atol=0.01, rtol=0.08)

        tp_wrapper, _ = make_wrapper(
            weights, lora, None, args.expert_num, max_len, args.rank, scaling, args.threads, tp_count=args.tp_count
        )
        tp_wrapper_output = tp_wrapper.forward(x, expert_ids, route_weights, save_for_backward=True)
        assert_close(f"tp{args.tp_count}_wrapper_forward_output", tp_wrapper_output, output_lora, atol=0.03, rtol=0.08)
        tp_wrapper_grad_input, tp_wrapper_grad_weights = tp_wrapper.backward(grad_output)
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_grad_input",
            tp_wrapper_grad_input,
            gate_up_ref["grad_input"],
            atol=0.02,
            rtol=0.08,
        )
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_grad_weights",
            tp_wrapper_grad_weights,
            grad_weights_ref,
            atol=0.002,
            rtol=0.02,
        )
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_gate_lora_a",
            tp_wrapper.grad_gate_lora_a,
            gate_up_ref["grad_gate_lora_a"],
            atol=0.01,
            rtol=0.08,
        )
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_gate_lora_b",
            tp_wrapper.grad_gate_lora_b,
            gate_up_ref["grad_gate_lora_b"],
            atol=0.01,
            rtol=0.08,
        )
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_up_lora_a",
            tp_wrapper.grad_up_lora_a,
            gate_up_ref["grad_up_lora_a"],
            atol=0.01,
            rtol=0.08,
        )
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_up_lora_b",
            tp_wrapper.grad_up_lora_b,
            gate_up_ref["grad_up_lora_b"],
            atol=0.01,
            rtol=0.08,
        )
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_down_lora_a",
            tp_wrapper.grad_down_lora_a,
            down_ref["grad_down_lora_a"],
            atol=0.01,
            rtol=0.08,
        )
        assert_close(
            f"tp{args.tp_count}_wrapper_backward_down_lora_b",
            tp_wrapper.grad_down_lora_b,
            down_ref["grad_down_lora_b"],
            atol=0.01,
            rtol=0.08,
        )
        if tp_wrapper._cache_depth != 0:
            raise AssertionError(f"tp{args.tp_count} wrapper cache depth mismatch: expected 0, got {tp_wrapper._cache_depth}")
        assert_forward_cache_empty(tp_wrapper.moe, f"tp{args.tp_count}_wrapper_backward_cache_pop")

    print(
        "K2 SFT LoRA forward cache reference passed: "
        f"active_experts={cache_lora.active_experts}, local_counts={cache_lora.local_counts[:args.expert_num]}"
    )


if __name__ == "__main__":
    main()
