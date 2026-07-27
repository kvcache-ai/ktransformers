#!/usr/bin/env python
"""Small executable INT8 SFT numerical and optimizer-window contract test."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="default")

try:
    import torch
    import torch.nn.functional as F

    import kt_kernel

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    _IMPORT_ERROR = None
except (ImportError, AttributeError) as error:
    torch = None
    F = None
    kt_kernel_ext = None
    _IMPORT_ERROR = str(error)


EXPERTS = 3
TOP_K = 1
HIDDEN = 256
INTERMEDIATE = 256
RANK = 8
ALPHA = 8.0
QLEN = 4
MAX_QLEN = 32

GRAD_SHAPES = {
    "gate_lora_a": (EXPERTS, RANK, HIDDEN),
    "gate_lora_b": (EXPERTS, INTERMEDIATE, RANK),
    "up_lora_a": (EXPERTS, RANK, HIDDEN),
    "up_lora_b": (EXPERTS, INTERMEDIATE, RANK),
    "down_lora_a": (EXPERTS, RANK, INTERMEDIATE),
    "down_lora_b": (EXPERTS, HIDDEN, RANK),
}


@dataclass
class Batch:
    expert: int
    inputs: "torch.Tensor"
    route_weights: "torch.Tensor"
    grad_output: "torch.Tensor"


def _environment_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _make_cpu_infer():
    tp_count = _environment_int("KT_INT8_SFT_TEST_TP_COUNT", 1)
    threads_per_tp = _environment_int("KT_INT8_SFT_TEST_THREADS_PER_TP", 8)
    configured_map = os.environ.get("KT_INT8_SFT_TEST_NUMA_MAP")
    numa_map = (
        [int(value) for value in configured_map.split(",")]
        if configured_map
        else [0] * tp_count
    )
    if len(numa_map) != tp_count:
        raise ValueError(
            "KT_INT8_SFT_TEST_NUMA_MAP must contain one NUMA id per TP partition: "
            f"tp_count={tp_count}, numa_map={numa_map}"
        )

    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = tp_count
    worker_config.subpool_numa_map = numa_map
    worker_config.subpool_thread_count = [threads_per_tp] * tp_count
    return kt_kernel_ext.CPUInfer(worker_config), tp_count, numa_map


def _randn(shape, generator, scale):
    return (
        (torch.randn(shape, generator=generator, dtype=torch.float32) * scale)
        .to(torch.bfloat16)
        .contiguous()
    )


def _make_weights():
    generator = torch.Generator(device="cpu").manual_seed(20260728)
    base = {
        "gate": _randn((EXPERTS, INTERMEDIATE, HIDDEN), generator, 0.08),
        "up": _randn((EXPERTS, INTERMEDIATE, HIDDEN), generator, 0.08),
        "down": _randn((EXPERTS, HIDDEN, INTERMEDIATE), generator, 0.08),
    }
    lora = {name: _randn(shape, generator, 0.04) for name, shape in GRAD_SHAPES.items()}
    return base, lora


def _make_batch(expert: int, seed: int) -> Batch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _randn((QLEN, HIDDEN), generator, 0.20)
    grad_output = _randn((QLEN, HIDDEN), generator, 0.10)
    route_weights = (
        torch.linspace(0.55, 0.85, QLEN, dtype=torch.float32)
        .reshape(QLEN, TOP_K)
        .contiguous()
    )
    return Batch(
        expert=expert,
        inputs=inputs,
        route_weights=route_weights,
        grad_output=grad_output,
    )


def _reference(base, lora, batch: Batch):
    inputs = batch.inputs.float().requires_grad_(True)
    route_weights = batch.route_weights.float().requires_grad_(True)
    parameters = {
        name: value.float().detach().requires_grad_(True)
        for name, value in lora.items()
    }
    expert = batch.expert
    scaling = ALPHA / RANK

    gate = F.linear(inputs, base["gate"][expert].float())
    gate = gate + scaling * F.linear(
        F.linear(inputs, parameters["gate_lora_a"][expert]),
        parameters["gate_lora_b"][expert],
    )
    up = F.linear(inputs, base["up"][expert].float())
    up = up + scaling * F.linear(
        F.linear(inputs, parameters["up_lora_a"][expert]),
        parameters["up_lora_b"][expert],
    )
    intermediate = F.silu(gate) * up
    output = F.linear(intermediate, base["down"][expert].float())
    output = output + scaling * F.linear(
        F.linear(intermediate, parameters["down_lora_a"][expert]),
        parameters["down_lora_b"][expert],
    )
    output = output * route_weights

    targets = (inputs, route_weights, *(parameters[name] for name in GRAD_SHAPES))
    gradients = torch.autograd.grad(
        output,
        targets,
        grad_outputs=batch.grad_output.float(),
        retain_graph=False,
        create_graph=False,
    )
    return (
        output.detach(),
        gradients[0].detach(),
        gradients[1].detach(),
        {name: gradient.detach() for name, gradient in zip(GRAD_SHAPES, gradients[2:])},
    )


def _make_backend(base, lora):
    cpu_infer, tp_count, numa_map = _make_cpu_infer()
    physical_to_logical_map = torch.arange(EXPERTS, dtype=torch.int64).contiguous()

    config = kt_kernel_ext.moe.MOESFTConfig(EXPERTS, TOP_K, HIDDEN, INTERMEDIATE)
    config.max_len = MAX_QLEN
    config.max_cache_depth = 1
    config.layer_idx = 0
    config.lora_rank = RANK
    config.lora_alpha = ALPHA
    config.full_weight_grad = False
    config.authoritative_optimizer_grads = True
    config.share_backward_bb = False
    config.share_cache_pool = False
    config.physical_to_logical_map = physical_to_logical_map.data_ptr()
    config.gate_proj = base["gate"].data_ptr()
    config.up_proj = base["up"].data_ptr()
    config.down_proj = base["down"].data_ptr()
    config.gate_lora_a = lora["gate_lora_a"].data_ptr()
    config.gate_lora_b = lora["gate_lora_b"].data_ptr()
    config.up_lora_a = lora["up_lora_a"].data_ptr()
    config.up_lora_b = lora["up_lora_b"].data_ptr()
    config.down_lora_a = lora["down_lora_a"].data_ptr()
    config.down_lora_b = lora["down_lora_b"].data_ptr()
    config.pool = cpu_infer.backend_

    moe = kt_kernel_ext.moe.AMXInt8_SFT_MOE(config)
    moe.load_weights()
    keepalive = (cpu_infer, physical_to_logical_map, base, lora)
    return moe, keepalive, tp_count, numa_map


def _make_grad_buffers():
    # A nonzero sentinel proves that the first authoritative backward initializes
    # the complete six-buffer set before publishing any gradients.
    return {
        name: torch.full(shape, 7.0, dtype=torch.bfloat16).contiguous()
        for name, shape in GRAD_SHAPES.items()
    }


def _run_cpp_batch(
    moe, grad_buffers, batch: Batch, *, accumulate: bool, grad_scale: float
):
    qlen = torch.tensor([QLEN], dtype=torch.int32)
    expert_ids = torch.full((QLEN, TOP_K), batch.expert, dtype=torch.int64).contiguous()
    output = torch.empty((QLEN, HIDDEN), dtype=torch.bfloat16).contiguous()
    grad_input = torch.empty_like(output)
    grad_weights = torch.empty((QLEN, TOP_K), dtype=torch.float32).contiguous()

    moe.forward_sft(
        qlen.data_ptr(),
        TOP_K,
        expert_ids.data_ptr(),
        batch.route_weights.data_ptr(),
        batch.inputs.data_ptr(),
        output.data_ptr(),
        True,
    )
    moe.backward(
        batch.grad_output.data_ptr(),
        grad_input.data_ptr(),
        grad_buffers["gate_lora_a"].data_ptr(),
        grad_buffers["gate_lora_b"].data_ptr(),
        grad_buffers["up_lora_a"].data_ptr(),
        grad_buffers["up_lora_b"].data_ptr(),
        grad_buffers["down_lora_a"].data_ptr(),
        grad_buffers["down_lora_b"].data_ptr(),
        grad_weights.data_ptr(),
        0,
        0,
        0,
        accumulate,
        grad_scale,
    )
    return output.float(), grad_input.float(), grad_weights.float()


def _metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    expected_norm = float(torch.linalg.vector_norm(expected))
    actual_norm = float(torch.linalg.vector_norm(actual))
    relative_l2 = float(torch.linalg.vector_norm(difference)) / max(
        expected_norm, 1.0e-12
    )
    cosine = float(torch.dot(actual.flatten(), expected.flatten())) / max(
        actual_norm * expected_norm, 1.0e-20
    )
    return relative_l2, cosine


def _assert_tensor_close(name, actual, expected, *, relative_l2_limit, cosine_limit):
    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
    assert torch.isfinite(expected).all(), (
        f"{name} reference contains non-finite values"
    )
    assert torch.count_nonzero(expected), (
        f"{name} reference unexpectedly has zero signal"
    )
    relative_l2, cosine = _metrics(actual, expected)
    assert relative_l2 <= relative_l2_limit, (
        f"{name} relative L2 {relative_l2:.6f} exceeds {relative_l2_limit:.6f}"
    )
    assert cosine >= cosine_limit, (
        f"{name} cosine {cosine:.6f} is below {cosine_limit:.6f}"
    )
    return relative_l2, cosine


def _expected_window(*reference_grads, scale):
    return {
        name: sum(
            (gradients[name] for gradients in reference_grads),
            torch.zeros_like(reference_grads[0][name]),
        )
        * scale
        for name in GRAD_SHAPES
    }


def _assert_window(stage, grad_buffers, expected, active_experts):
    relative_l2_limit = float(os.environ.get("KT_INT8_SFT_TEST_GRAD_REL_L2", "0.15"))
    cosine_limit = float(os.environ.get("KT_INT8_SFT_TEST_GRAD_COSINE", "0.985"))
    stage_summary = {}
    active_experts = set(active_experts)

    for name in GRAD_SHAPES:
        actual = grad_buffers[name].float()
        relative_l2, cosine = _assert_tensor_close(
            f"{stage}.{name}",
            actual,
            expected[name],
            relative_l2_limit=relative_l2_limit,
            cosine_limit=cosine_limit,
        )
        for expert in range(EXPERTS):
            if expert not in active_experts:
                assert torch.count_nonzero(actual[expert]) == 0, (
                    f"{stage}.{name}: stale gradient remains for inactive expert {expert}"
                )
        stage_summary[name] = {
            "l2": round(float(torch.linalg.vector_norm(actual)), 7),
            "sum": round(float(actual.sum()), 7),
            "relative_l2": round(relative_l2, 7),
            "cosine": round(cosine, 7),
        }
    return stage_summary


def _run_contract():
    torch.manual_seed(20260728)
    torch.set_num_threads(1)
    base, lora = _make_weights()
    moe, keepalive, tp_count, numa_map = _make_backend(base, lora)
    grad_buffers = _make_grad_buffers()
    grad_scale = float(os.environ.get("KT_INT8_SFT_TEST_GRAD_SCALE", "0.5"))
    if not math.isfinite(grad_scale) or grad_scale <= 0:
        raise ValueError(
            f"KT_INT8_SFT_TEST_GRAD_SCALE must be finite and positive, got {grad_scale}"
        )

    batches = [
        _make_batch(0, 1001),
        _make_batch(1, 1002),
        _make_batch(2, 1003),
        _make_batch(0, 1004),
        _make_batch(0, 1005),
    ]
    references = [_reference(base, lora, batch) for batch in batches]
    summary = {}

    output, grad_input, grad_weights = _run_cpp_batch(
        moe,
        grad_buffers,
        batches[0],
        accumulate=False,
        grad_scale=grad_scale,
    )
    _assert_tensor_close(
        "window1_mb1.forward",
        output,
        references[0][0],
        relative_l2_limit=0.12,
        cosine_limit=0.99,
    )
    _assert_tensor_close(
        "window1_mb1.grad_input",
        grad_input,
        references[0][1],
        relative_l2_limit=0.18,
        cosine_limit=0.98,
    )
    _assert_tensor_close(
        "window1_mb1.grad_weights",
        grad_weights,
        references[0][2],
        relative_l2_limit=0.12,
        cosine_limit=0.99,
    )

    _run_cpp_batch(
        moe, grad_buffers, batches[1], accumulate=True, grad_scale=grad_scale
    )
    expected = _expected_window(references[0][3], references[1][3], scale=grad_scale)
    summary["gas_accumulate"] = _assert_window(
        "gas_accumulate", grad_buffers, expected, {0, 1}
    )

    _run_cpp_batch(
        moe, grad_buffers, batches[2], accumulate=False, grad_scale=grad_scale
    )
    expected = _expected_window(references[2][3], scale=grad_scale)
    summary["switch_expert"] = _assert_window(
        "switch_expert", grad_buffers, expected, {2}
    )

    _run_cpp_batch(
        moe, grad_buffers, batches[3], accumulate=False, grad_scale=grad_scale
    )
    expected = _expected_window(references[3][3], scale=grad_scale)
    summary["return_expert"] = _assert_window(
        "return_expert", grad_buffers, expected, {0}
    )

    _run_cpp_batch(
        moe, grad_buffers, batches[4], accumulate=False, grad_scale=grad_scale
    )
    expected = _expected_window(references[4][3], scale=grad_scale)
    summary["same_expert_overwrite"] = _assert_window(
        "same_expert_overwrite", grad_buffers, expected, {0}
    )

    result = {
        "tp_count": tp_count,
        "numa_map": numa_map,
        "grad_scale": grad_scale,
        "summary": summary,
    }
    print("KT_INT8_SFT_SUMMARY=" + json.dumps(result, sort_keys=True))
    del moe
    del keepalive
    return result


@pytest.mark.cpu
def test_int8_sft_numerical_and_optimizer_windows():
    if _IMPORT_ERROR is not None:
        pytest.skip(f"kt-kernel extension unavailable: {_IMPORT_ERROR}")
    if not hasattr(kt_kernel_ext.moe, "AMXInt8_SFT_MOE"):
        pytest.skip("AMXInt8_SFT_MOE is unavailable in this build")
    _run_contract()


@pytest.mark.cpu
def test_int8_async_load_error_reaches_python(tmp_path):
    if _IMPORT_ERROR is not None:
        pytest.skip(f"kt-kernel extension unavailable: {_IMPORT_ERROR}")
    if not hasattr(kt_kernel_ext.moe, "AMXInt8_SFT_MOE"):
        pytest.skip("AMXInt8_SFT_MOE is unavailable in this build")

    cpu_infer, tp_count, _ = _make_cpu_infer()
    physical_to_logical_map = torch.arange(1, dtype=torch.int64).contiguous()
    config = kt_kernel_ext.moe.MOESFTConfig(1, 1, HIDDEN, INTERMEDIATE)
    config.max_len = MAX_QLEN
    config.max_cache_depth = 1
    config.layer_idx = 0
    config.lora_rank = 0
    config.load = True
    missing_root = tmp_path / "missing-int8-root"
    config.path = str(missing_root)
    config.share_backward_bb = True
    config.physical_to_logical_map = physical_to_logical_map.data_ptr()
    config.pool = cpu_infer.backend_

    with pytest.raises(RuntimeError, match="construction failed: Path not found"):
        kt_kernel_ext.moe.AMXInt8_SFT_MOE(config)

    for tp_index in range(tp_count):
        (missing_root / "_layer_0" / f"_numa_{tp_index}").mkdir(parents=True)
    moe = kt_kernel_ext.moe.AMXInt8_SFT_MOE(config)
    cpu_infer.submit(moe.load_weights_task())
    with pytest.raises(RuntimeError, match="missing weight file"):
        cpu_infer.sync()

    # The worker must decrement pending and clear the delivered exception so
    # the queue remains usable after Python handles the failure.
    cpu_infer.sync()


if __name__ == "__main__":
    if _IMPORT_ERROR is not None:
        raise SystemExit(f"kt-kernel extension unavailable: {_IMPORT_ERROR}")
    _run_contract()
