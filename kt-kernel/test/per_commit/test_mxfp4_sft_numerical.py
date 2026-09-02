#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Executable C++ contract test for native MXFP4 routed-expert LoRA SFT.

Pytest may collect this file in a source-only environment and skip it when no
extension has been built.  The kt-kernel CPU suite executes the file directly;
that path deliberately fails when ``MXFP4_SFT_MOE`` is absent so release
validation cannot silently pass against a stale wheel.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci
from mxfp4_sft_reference import (
    LORA_NAMES,
    MXFP4ExpertWeights,
    MXFP4Projection,
    base_storage_hash,
    pack_e2m1_codes,
    relative_l2_and_cosine,
    run_routed_reference,
    ue8m0_to_bf16,
)

register_cpu_ci(est_time=120, suite="default")

try:
    import kt_kernel

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    _IMPORT_ERROR = None
except (ImportError, AttributeError) as error:
    kt_kernel_ext = None
    _IMPORT_ERROR = str(error)


EXPERTS = 4
TOP_K = 2
HIDDEN = 256
INTERMEDIATE = 256
RANK = 8
ALPHA = 16.0
SWIGLU_LIMIT = 10.0
QLEN = 5
MAX_QLEN = 32

GRAD_SHAPES = {
    "gate_lora_a": (EXPERTS, RANK, HIDDEN),
    "gate_lora_b": (EXPERTS, INTERMEDIATE, RANK),
    "up_lora_a": (EXPERTS, RANK, HIDDEN),
    "up_lora_b": (EXPERTS, INTERMEDIATE, RANK),
    "down_lora_a": (EXPERTS, RANK, INTERMEDIATE),
    "down_lora_b": (EXPERTS, HIDDEN, RANK),
}


@dataclass(frozen=True)
class Batch:
    inputs: torch.Tensor
    expert_ids: torch.Tensor
    route_weights: torch.Tensor
    grad_output: torch.Tensor

    @property
    def active_experts(self) -> set[int]:
        return {int(value) for value in self.expert_ids.flatten()}


def _make_cpu_infer(tp_count: int):
    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = tp_count
    worker_config.subpool_numa_map = [0] * tp_count
    worker_config.subpool_thread_count = [4] * tp_count
    return kt_kernel_ext.CPUInfer(worker_config)


def _make_projection(shape, generator):
    experts, output_features, input_features = shape
    codes = torch.randint(
        0,
        16,
        (experts, output_features, input_features),
        generator=generator,
    )
    packed = pack_e2m1_codes(codes)
    encoded_scales = torch.randint(
        120,
        123,
        (experts, output_features, input_features // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    return packed, ue8m0_to_bf16(encoded_scales)


def _make_weights():
    generator = torch.Generator(device="cpu").manual_seed(20260902)
    gate, gate_scale = _make_projection((EXPERTS, INTERMEDIATE, HIDDEN), generator)
    up, up_scale = _make_projection((EXPERTS, INTERMEDIATE, HIDDEN), generator)
    down, down_scale = _make_projection((EXPERTS, HIDDEN, INTERMEDIATE), generator)
    base = {
        "gate": gate.contiguous(),
        "up": up.contiguous(),
        "down": down.contiguous(),
        "gate_scale": gate_scale.contiguous(),
        "up_scale": up_scale.contiguous(),
        "down_scale": down_scale.contiguous(),
    }
    # Force both sides of the V4 clamp in the extension-backed comparison.
    # 0x77 decodes to two +6 values and 0xaa to two -1 values.
    base["gate"][0, 0].fill_(0x77)
    base["up"][0, 0].fill_(0x77)
    base["gate"][0, 1].fill_(0xAA)
    base["up"][0, 1].fill_(0x77)
    base["gate_scale"][0, :2].fill_(0.0625)
    base["up_scale"][0, :2].fill_(0.0625)
    # Exercise legal UE8M0 code 0 (2^-127, BF16 bits 0x0040) through
    # forward scale expansion, transpose-free dX, and compact-scale validation.
    base["down_scale"].view(torch.int16)[3, -1, -1] = 0x0040

    def randn(shape, scale):
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .mul_(scale)
            .to(torch.bfloat16)
            .contiguous()
        )

    lora = {name: randn(shape, 0.04) for name, shape in GRAD_SHAPES.items()}
    experts = [
        MXFP4ExpertWeights(
            gate=MXFP4Projection(base["gate"][idx], base["gate_scale"][idx]),
            up=MXFP4Projection(base["up"][idx], base["up_scale"][idx]),
            down=MXFP4Projection(base["down"][idx], base["down_scale"][idx]),
        )
        for idx in range(EXPERTS)
    ]
    return base, lora, experts


def _make_batch(active_pair: tuple[int, int], seed: int) -> Batch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = (
        torch.randn((QLEN, HIDDEN), generator=generator, dtype=torch.float32)
        .mul_(0.20)
        .to(torch.bfloat16)
        .contiguous()
    )
    inputs[0].fill_(1.0)
    inputs[1].fill_(-1.0)
    grad_output = (
        torch.randn((QLEN, HIDDEN), generator=generator, dtype=torch.float32)
        .mul_(0.10)
        .to(torch.bfloat16)
        .contiguous()
    )
    first, second = active_pair
    expert_ids = torch.tensor(
        [
            [first, second] if token % 2 == 0 else [second, first]
            for token in range(QLEN)
        ],
        dtype=torch.int64,
    ).contiguous()
    route_weights = (
        torch.stack(
            (
                torch.linspace(0.55, 0.75, QLEN),
                torch.linspace(0.45, 0.25, QLEN),
            ),
            dim=1,
        )
        .to(torch.float32)
        .contiguous()
    )
    return Batch(inputs, expert_ids, route_weights, grad_output)


def _reference(experts, lora, batch):
    return run_routed_reference(
        batch.inputs,
        batch.expert_ids,
        batch.route_weights,
        experts,
        lora,
        batch.grad_output,
        lora_scaling=ALPHA / RANK,
        swiglu_limit=SWIGLU_LIMIT,
        transpose_free_backward=False,
    )


def _make_backend(base, lora, tp_count, physical_to_logical_map=None):
    cpu_infer = _make_cpu_infer(tp_count)
    if physical_to_logical_map is None:
        physical_to_logical_map = torch.arange(EXPERTS, dtype=torch.int64).contiguous()
    config = kt_kernel_ext.moe.MOESFTConfig(EXPERTS, TOP_K, HIDDEN, INTERMEDIATE)
    config.max_len = MAX_QLEN
    config.max_cache_depth = 2
    config.layer_idx = 0
    config.lora_rank = RANK
    config.lora_alpha = ALPHA
    config.lora_dropout = 0.0
    config.full_weight_grad = False
    config.authoritative_optimizer_grads = True
    config.share_backward_bb = False
    config.share_cache_pool = False
    config.physical_to_logical_map = physical_to_logical_map.data_ptr()
    config.gate_proj = base["gate"].data_ptr()
    config.up_proj = base["up"].data_ptr()
    config.down_proj = base["down"].data_ptr()
    config.gate_scale = base["gate_scale"].data_ptr()
    config.up_scale = base["up_scale"].data_ptr()
    config.down_scale = base["down_scale"].data_ptr()
    config.gate_lora_a = lora["gate_lora_a"].data_ptr()
    config.gate_lora_b = lora["gate_lora_b"].data_ptr()
    config.up_lora_a = lora["up_lora_a"].data_ptr()
    config.up_lora_b = lora["up_lora_b"].data_ptr()
    config.down_lora_a = lora["down_lora_a"].data_ptr()
    config.down_lora_b = lora["down_lora_b"].data_ptr()
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.swiglu_limit = SWIGLU_LIMIT
    config.swiglu_alpha = 0.0
    config.pool = cpu_infer.backend_

    moe = kt_kernel_ext.moe.MXFP4_SFT_MOE(config)
    moe.load_weights()
    keepalive = (cpu_infer, physical_to_logical_map, base, lora)
    return moe, keepalive


def _make_grad_buffers():
    # A nonzero sentinel proves the first authoritative write initializes all
    # six buffers, including experts absent from the first microbatch.
    return {
        name: torch.full(shape, 7.0, dtype=torch.bfloat16).contiguous()
        for name, shape in GRAD_SHAPES.items()
    }


def _forward(moe, batch):
    qlen = torch.tensor([QLEN], dtype=torch.int32)
    output = torch.empty((QLEN, HIDDEN), dtype=torch.bfloat16).contiguous()
    moe.forward_sft(
        qlen.data_ptr(),
        TOP_K,
        batch.expert_ids.data_ptr(),
        batch.route_weights.data_ptr(),
        batch.inputs.data_ptr(),
        output.data_ptr(),
        True,
    )
    return output.float()


def _backward(moe, grad_buffers, batch, *, accumulate, grad_scale):
    grad_input = torch.empty((QLEN, HIDDEN), dtype=torch.bfloat16).contiguous()
    grad_weights = torch.empty((QLEN, TOP_K), dtype=torch.float32).contiguous()
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
        bool(accumulate),
        float(grad_scale),
    )
    return grad_input.float(), grad_weights.float()


def _forward_backward(moe, grad_buffers, batch, *, accumulate, grad_scale):
    output = _forward(moe, batch)
    grad_input, grad_weights = _backward(
        moe,
        grad_buffers,
        batch,
        accumulate=accumulate,
        grad_scale=grad_scale,
    )
    return output, grad_input, grad_weights


def _assert_frozen_base_fail_fast(moe, grad_buffers, batch):
    _forward(moe, batch)
    grad_input = torch.empty((QLEN, HIDDEN), dtype=torch.bfloat16).contiguous()
    grad_weights = torch.empty((QLEN, TOP_K), dtype=torch.float32).contiguous()
    forbidden_base_grad = torch.empty(1, dtype=torch.bfloat16)
    # std::invalid_argument is surfaced by pybind11 as ValueError.
    with pytest.raises(ValueError, match="base|frozen|gradient"):
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
            forbidden_base_grad.data_ptr(),
            0,
            0,
            False,
            1.0,
        )
    with pytest.raises(RuntimeError, match="frozen|base"):
        moe.set_base_weight_pointers(
            forbidden_base_grad.data_ptr(),
            forbidden_base_grad.data_ptr(),
            forbidden_base_grad.data_ptr(),
        )
    with pytest.raises(RuntimeError, match="frozen|base|transpose|backward"):
        moe.prepare_and_save_bwd(
            forbidden_base_grad.data_ptr(),
            forbidden_base_grad.data_ptr(),
            forbidden_base_grad.data_ptr(),
            "/tmp/kt-mxfp4-sft-forbidden-repack",
        )
    with pytest.raises(RuntimeError, match="immutable|already loaded"):
        moe.load_weights()


def _assert_close(name, actual, expected, *, relative_l2_limit, cosine_limit):
    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
    assert torch.isfinite(expected).all(), (
        f"{name} reference contains non-finite values"
    )
    assert torch.count_nonzero(expected), f"{name} reference unexpectedly has no signal"
    relative_l2, cosine = relative_l2_and_cosine(actual, expected)
    assert relative_l2 <= relative_l2_limit, (
        f"{name} relative L2 {relative_l2:.6f} exceeds {relative_l2_limit:.6f}"
    )
    assert cosine >= cosine_limit, (
        f"{name} cosine {cosine:.6f} is below {cosine_limit:.6f}"
    )
    return relative_l2, cosine


def _expected_window(reference_gradients, grad_scale):
    expected = {}
    for name in LORA_NAMES:
        total = torch.zeros_like(reference_gradients[0][name])
        for gradients in reference_gradients:
            total.add_(gradients[name])
        expected[name] = total * grad_scale
    return expected


def _assert_window(stage, grad_buffers, expected, active_experts):
    summary = {}
    for name in LORA_NAMES:
        actual = grad_buffers[name].float()
        relative_l2, cosine = _assert_close(
            f"{stage}.{name}",
            actual,
            expected[name],
            relative_l2_limit=0.05,
            cosine_limit=0.995,
        )
        for expert in range(EXPERTS):
            if expert not in active_experts:
                assert torch.count_nonzero(actual[expert]) == 0, (
                    f"{stage}.{name}: stale gradient remains for expert {expert}"
                )
        summary[name] = {
            "relative_l2": round(relative_l2, 7),
            "cosine": round(cosine, 7),
        }
    return summary


def _run_sft_contract(tp_count):
    base, lora, experts = _make_weights()
    base_hash_before = base_storage_hash(experts)
    moe, keepalive = _make_backend(base, lora, tp_count)
    grad_buffers = _make_grad_buffers()
    grad_scale = 0.5
    active_pairs = ((0, 1), (1, 2), (2, 3), (0, 3)) * 2
    batches = [
        _make_batch(pair, 3000 + index) for index, pair in enumerate(active_pairs)
    ]
    references = [_reference(experts, lora, batch) for batch in batches]

    output, grad_input, grad_weights = _forward_backward(
        moe,
        grad_buffers,
        batches[0],
        accumulate=False,
        grad_scale=grad_scale,
    )
    _assert_close(
        "gas1.forward",
        output,
        references[0][0],
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    _assert_close(
        "gas1.dX",
        grad_input,
        references[0][1],
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    _assert_close(
        "gas1.dRoute",
        grad_weights,
        references[0][2],
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    summary = {
        "gas1": _assert_window(
            "gas1",
            grad_buffers,
            _expected_window([references[0][3]], grad_scale),
            batches[0].active_experts,
        )
    }

    _forward_backward(
        moe,
        grad_buffers,
        batches[1],
        accumulate=True,
        grad_scale=grad_scale,
    )
    summary["gas2"] = _assert_window(
        "gas2",
        grad_buffers,
        _expected_window([references[0][3], references[1][3]], grad_scale),
        batches[0].active_experts | batches[1].active_experts,
    )
    for index in range(2, 8):
        _forward_backward(
            moe,
            grad_buffers,
            batches[index],
            accumulate=True,
            grad_scale=grad_scale,
        )
    summary["gas8"] = _assert_window(
        "gas8",
        grad_buffers,
        _expected_window([reference[3] for reference in references], grad_scale),
        set().union(*(batch.active_experts for batch in batches)),
    )

    # A new optimizer window with only experts 2/3 must overwrite those rows
    # and lazily clear stale rows 0/1 from the previous window.
    new_window = _make_batch((2, 3), 4001)
    new_reference = _reference(experts, lora, new_window)
    _forward_backward(
        moe,
        grad_buffers,
        new_window,
        accumulate=False,
        grad_scale=grad_scale,
    )
    summary["lazy_clear"] = _assert_window(
        "lazy_clear",
        grad_buffers,
        _expected_window([new_reference[3]], grad_scale),
        {2, 3},
    )

    # A second saved forward before backward is the checkpoint-recompute
    # contract: it replaces the pending cache and therefore has exactly one
    # matching backward.  Validate the replacement, then run the first batch
    # again as a normal forward/backward pair.
    first = _make_batch((0, 1), 5001)
    second = _make_batch((1, 3), 5002)
    first_ref = _reference(experts, lora, first)
    second_ref = _reference(experts, lora, second)
    _forward(moe, first)
    _forward(moe, second)
    second_dx, second_droute = _backward(
        moe, grad_buffers, second, accumulate=False, grad_scale=grad_scale
    )
    _forward(moe, first)
    first_dx, first_droute = _backward(
        moe, grad_buffers, first, accumulate=True, grad_scale=grad_scale
    )
    _assert_close(
        "lifo.second.dX",
        second_dx,
        second_ref[1],
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    _assert_close(
        "lifo.second.dRoute",
        second_droute,
        second_ref[2],
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    _assert_close(
        "lifo.first.dX",
        first_dx,
        first_ref[1],
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    _assert_close(
        "lifo.first.dRoute",
        first_droute,
        first_ref[2],
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    summary["cache_replacement"] = _assert_window(
        "cache_replacement",
        grad_buffers,
        _expected_window([second_ref[3], first_ref[3]], grad_scale),
        {0, 1, 3},
    )

    _assert_frozen_base_fail_fast(moe, grad_buffers, _make_batch((0, 1), 7001))
    assert base_storage_hash(experts) == base_hash_before
    result = {
        "tp_count": tp_count,
        "base_sha256": base_hash_before,
        "summary": summary,
        "output": output.clone(),
        "grad_input": grad_input.clone(),
        "grad_weights": grad_weights.clone(),
        "grads": {
            name: tensor.float().clone() for name, tensor in grad_buffers.items()
        },
    }
    del moe
    del keepalive
    return result


def _run_inference_regression():
    base, _, experts = _make_weights()
    before_hash = base_storage_hash(experts)
    cpu_infer = _make_cpu_infer(1)
    physical_to_logical_map = torch.arange(EXPERTS, dtype=torch.int64).contiguous()
    config = kt_kernel_ext.moe.MOEConfig(EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, 0)
    config.max_len = MAX_QLEN
    config.layer_idx = 0
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.swiglu_limit = SWIGLU_LIMIT
    config.swiglu_alpha = 0.0
    config.pool = cpu_infer.backend_
    config.gate_proj = base["gate"].data_ptr()
    config.up_proj = base["up"].data_ptr()
    config.down_proj = base["down"].data_ptr()
    config.gate_scale = base["gate_scale"].data_ptr()
    config.up_scale = base["up_scale"].data_ptr()
    config.down_scale = base["down_scale"].data_ptr()
    moe = kt_kernel_ext.moe.AMXFP4_KGroup_MOE(config)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpu_infer.sync()

    batch = _make_batch((0, 2), 6001)
    qlen = torch.tensor([QLEN], dtype=torch.int32)
    output = torch.empty((QLEN, HIDDEN), dtype=torch.bfloat16).contiguous()
    cpu_infer.submit(
        moe.forward_task(
            qlen.data_ptr(),
            TOP_K,
            batch.expert_ids.data_ptr(),
            batch.route_weights.data_ptr(),
            batch.inputs.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()

    zero_lora = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in GRAD_SHAPES.items()
    }
    reference = run_routed_reference(
        batch.inputs,
        batch.expert_ids,
        batch.route_weights,
        experts,
        zero_lora,
        batch.grad_output,
        lora_scaling=ALPHA / RANK,
        swiglu_limit=SWIGLU_LIMIT,
        transpose_free_backward=False,
    )[0]
    relative_l2, cosine = _assert_close(
        "inference.forward",
        output.float(),
        reference,
        relative_l2_limit=0.03,
        cosine_limit=0.995,
    )
    assert base_storage_hash(experts) == before_hash
    return {"relative_l2": relative_l2, "cosine": cosine}


def _require_extension(*, allow_source_only_skip):
    if _IMPORT_ERROR is not None:
        if allow_source_only_skip:
            pytest.skip(f"kt-kernel extension unavailable: {_IMPORT_ERROR}")
        raise RuntimeError(f"kt-kernel extension unavailable: {_IMPORT_ERROR}")
    if not hasattr(kt_kernel_ext.moe, "MXFP4_SFT_MOE"):
        if allow_source_only_skip:
            pytest.skip("built extension does not contain MXFP4_SFT_MOE")
        raise AssertionError("built extension is missing required MXFP4_SFT_MOE")
    if not hasattr(kt_kernel_ext.moe, "AMXFP4_KGroup_MOE"):
        if allow_source_only_skip:
            pytest.skip("built extension does not contain AMXFP4_KGroup_MOE")
        raise AssertionError("built extension is missing existing AMXFP4_KGroup_MOE")


@pytest.mark.cpu
def test_mxfp4_sft_tp1_tp2_numerical_lifecycle_and_inference_regression():
    _require_extension(allow_source_only_skip=True)
    base, lora, _ = _make_weights()
    invalid_map = torch.tensor([0, 0, 2, 3], dtype=torch.int64).contiguous()
    with pytest.raises(RuntimeError, match="must be a permutation"):
        _make_backend(base, lora, 1, invalid_map)
    tp1 = _run_sft_contract(1)
    tp2 = _run_sft_contract(2)
    for name in ("output", "grad_input", "grad_weights"):
        _assert_close(
            f"tp1_vs_tp2.{name}",
            tp2[name],
            tp1[name],
            relative_l2_limit=0.01,
            cosine_limit=0.999,
        )
    for name in LORA_NAMES:
        _assert_close(
            f"tp1_vs_tp2.{name}",
            tp2["grads"][name],
            tp1["grads"][name],
            relative_l2_limit=0.01,
            cosine_limit=0.999,
        )
    inference = _run_inference_regression()
    print(
        "KT_MXFP4_SFT_SUMMARY="
        + json.dumps(
            {
                "tp1": {
                    key: value
                    for key, value in tp1.items()
                    if key not in {"output", "grad_input", "grad_weights", "grads"}
                },
                "tp2": {
                    key: value
                    for key, value in tp2.items()
                    if key not in {"output", "grad_input", "grad_weights", "grads"}
                },
                "inference": inference,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _require_extension(allow_source_only_skip=False)
    test_mxfp4_sft_tp1_tp2_numerical_lifecycle_and_inference_regression()
