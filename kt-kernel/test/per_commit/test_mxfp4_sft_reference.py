#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python numerical contract for native MXFP4 routed-expert LoRA."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci
from mxfp4_sft_reference import (
    E2M1_VALUES,
    LORA_NAMES,
    MXFP4ExpertWeights,
    MXFP4Projection,
    base_storage_hash,
    dequantize_mxfp4,
    pack_e2m1_codes,
    relative_l2_and_cosine,
    run_routed_reference,
    transpose_free_mxfp4_dx,
    ue8m0_to_bf16,
    unpack_e2m1,
    v4_swiglu,
)

register_cpu_ci(est_time=20, suite="default")


def _make_projection(
    output_features: int,
    input_features: int,
    generator: torch.Generator,
) -> MXFP4Projection:
    codes = torch.randint(
        0,
        16,
        (output_features, input_features),
        dtype=torch.int64,
        generator=generator,
    )
    # Keep the synthetic MLP well-scaled while exercising several UE8M0 values.
    encoded_scales = torch.randint(
        120,
        123,
        (output_features, input_features // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    return MXFP4Projection(
        packed=pack_e2m1_codes(codes),
        scales=ue8m0_to_bf16(encoded_scales),
    )


def _make_problem(*, expert_count=4, hidden=64, intermediate=64, rank=8):
    generator = torch.Generator(device="cpu").manual_seed(20260902)
    experts = [
        MXFP4ExpertWeights(
            gate=_make_projection(intermediate, hidden, generator),
            up=_make_projection(intermediate, hidden, generator),
            down=_make_projection(hidden, intermediate, generator),
        )
        for _ in range(expert_count)
    ]

    def randn(shape, scale=0.04):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    lora = {
        "gate_lora_a": randn((expert_count, rank, hidden)),
        "gate_lora_b": randn((expert_count, intermediate, rank)),
        "up_lora_a": randn((expert_count, rank, hidden)),
        "up_lora_b": randn((expert_count, intermediate, rank)),
        "down_lora_a": randn((expert_count, rank, intermediate)),
        "down_lora_b": randn((expert_count, hidden, rank)),
    }
    inputs = randn((5, hidden), scale=0.20)
    grad_output = randn((5, hidden), scale=0.10)
    expert_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]], dtype=torch.int64
    )
    route_weights = torch.tensor(
        [[0.75, 0.25], [0.60, 0.40], [0.55, 0.45], [0.70, 0.30], [0.65, 0.35]],
        dtype=torch.float32,
    )
    return experts, lora, inputs, expert_ids, route_weights, grad_output


def test_all_e2m1_codepoints_and_nibble_order():
    codes = torch.arange(16, dtype=torch.int64).repeat(2).reshape(1, 32)
    packed = pack_e2m1_codes(codes)
    decoded = unpack_e2m1(packed)

    torch.testing.assert_close(decoded[0, :16], E2M1_VALUES, rtol=0, atol=0)
    assert not torch.signbit(decoded[0, 0])
    assert torch.signbit(decoded[0, 8])
    assert int(packed[0, 0]) == 0x10
    assert int(packed[0, 7]) == 0xFE


def test_ue8m0_to_bf16_is_an_exact_exponent_bit_mapping():
    encoded = torch.tensor([0, 1, 120, 126, 127, 128, 254, 255], dtype=torch.uint8)
    converted = ue8m0_to_bf16(encoded)
    expected_bits = (encoded.to(torch.int32) << 7).to(torch.int16)
    expected_bits[0] = 0x0040

    assert torch.equal(converted.view(torch.int16), expected_bits)
    # Native FTZ/DAZ can flush this BF16 subnormal during float conversion.
    assert int(converted.view(torch.int16)[0]) == 0x0040
    assert float(converted[6]) == 2.0**127
    assert torch.isinf(converted[7])


@pytest.mark.parametrize("tokens", [1, 3, 4, 17])
def test_transpose_free_dx_matches_dense_with_row_tail_and_strides(tokens):
    generator = torch.Generator(device="cpu").manual_seed(1000 + tokens)
    n, k = 37, 64
    codes = torch.randint(0, 16, (n, k), generator=generator)
    packed_contiguous = pack_e2m1_codes(codes)
    scale_encoded = torch.randint(
        120, 129, (n, k // 32), dtype=torch.uint8, generator=generator
    )

    # Slice padded storage to exercise non-trivial row strides and a non-tile N tail.
    packed_storage = torch.zeros((n, packed_contiguous.shape[1] + 5), dtype=torch.uint8)
    packed_storage[:, : packed_contiguous.shape[1]] = packed_contiguous
    packed = packed_storage[:, : packed_contiguous.shape[1]]
    scale_storage = torch.ones((n, scale_encoded.shape[1] + 3), dtype=torch.bfloat16)
    scale_storage[:, : scale_encoded.shape[1]] = ue8m0_to_bf16(scale_encoded)
    scales = scale_storage[:, : scale_encoded.shape[1]]
    assert not packed.is_contiguous()
    assert not scales.is_contiguous()

    grad_output = torch.randn((tokens, n), generator=generator)
    actual = transpose_free_mxfp4_dx(grad_output, packed, scales)
    expected = grad_output @ dequantize_mxfp4(packed, scales)

    torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-6)


def test_mxfp4_layout_validation_fails_closed():
    packed = torch.zeros((3, 32), dtype=torch.uint8)
    scales = torch.ones((3, 2), dtype=torch.bfloat16)
    grad_output = torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="group_size=32"):
        transpose_free_mxfp4_dx(grad_output, packed, scales, group_size=16)
    with pytest.raises(ValueError, match="scales have shape"):
        transpose_free_mxfp4_dx(grad_output, packed, scales[:, :1])
    with pytest.raises(ValueError, match="grad_output must have shape"):
        transpose_free_mxfp4_dx(torch.ones((2, 4)), packed, scales)
    with pytest.raises(ValueError, match="rank-2 uint8"):
        dequantize_mxfp4(packed.to(torch.int8), scales)


def test_v4_clamp_backward_matches_pytorch_including_exact_boundaries():
    limit = 10.0
    gate = torch.tensor([-20.0, 9.0, 10.0, 11.0], requires_grad=True)
    up = torch.tensor([-11.0, -10.0, 10.0, 11.0], requires_grad=True)
    output = v4_swiglu(gate, up, limit)
    grad_gate, grad_up = torch.autograd.grad(output.sum(), (gate, up))

    clamped_gate = torch.clamp(gate.detach(), max=limit)
    silu = F.silu(clamped_gate)
    sigmoid = torch.sigmoid(clamped_gate)
    silu_derivative = sigmoid * (1.0 + clamped_gate * (1.0 - sigmoid))
    expected_gate = (
        torch.clamp(up.detach(), min=-limit, max=limit)
        * silu_derivative
        * (gate.detach() <= limit)
    )
    expected_up = silu * ((up.detach() >= -limit) & (up.detach() <= limit))
    torch.testing.assert_close(grad_gate, expected_gate)
    torch.testing.assert_close(grad_up, expected_up)
    assert grad_gate[-1] == 0
    assert grad_up[0] == 0 and grad_up[-1] == 0
    assert grad_gate[2] != 0 and grad_up[1] != 0 and grad_up[2] != 0


def test_lora_delta_is_applied_before_v4_clamp():
    base_gate = torch.tensor([9.75])
    lora_delta = torch.tensor([1.0])
    up = torch.tensor([2.0])
    correct = v4_swiglu(base_gate + lora_delta, up, 10.0)
    incorrect_post_clamp = F.silu(torch.clamp(base_gate, max=10.0) + lora_delta) * up

    torch.testing.assert_close(correct, F.silu(torch.tensor([10.0])) * up)
    assert not torch.allclose(correct, incorrect_post_clamp)


def test_single_expert_forward_dx_droute_and_six_lora_gradients():
    generator = torch.Generator(device="cpu").manual_seed(77)
    hidden = intermediate = 64
    rank = 8
    expert = MXFP4ExpertWeights(
        gate=_make_projection(intermediate, hidden, generator),
        up=_make_projection(intermediate, hidden, generator),
        down=_make_projection(hidden, intermediate, generator),
    )

    def randn(shape, scale):
        return torch.randn(shape, generator=generator) * scale

    lora = {
        "gate_lora_a": randn((1, rank, hidden), 0.04),
        "gate_lora_b": randn((1, intermediate, rank), 0.04),
        "up_lora_a": randn((1, rank, hidden), 0.04),
        "up_lora_b": randn((1, intermediate, rank), 0.04),
        "down_lora_a": randn((1, rank, intermediate), 0.04),
        "down_lora_b": randn((1, hidden, rank), 0.04),
    }
    inputs = randn((3, hidden), 0.2)
    expert_ids = torch.zeros((3, 1), dtype=torch.int64)
    routes = torch.tensor([[0.6], [0.7], [0.8]], dtype=torch.float32)
    grad_output = randn((3, hidden), 0.1)
    dense = run_routed_reference(
        inputs,
        expert_ids,
        routes,
        [expert],
        lora,
        grad_output,
        lora_scaling=2.0,
        swiglu_limit=10.0,
        transpose_free_backward=False,
    )
    transpose_free = run_routed_reference(
        inputs,
        expert_ids,
        routes,
        [expert],
        lora,
        grad_output,
        lora_scaling=2.0,
        swiglu_limit=10.0,
        transpose_free_backward=True,
    )

    for actual, expected in zip(transpose_free[:3], dense[:3]):
        torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-6)
        assert torch.count_nonzero(actual) > 0
    for name in LORA_NAMES:
        torch.testing.assert_close(
            transpose_free[3][name], dense[3][name], rtol=2.0e-6, atol=2.0e-6
        )
        assert torch.count_nonzero(transpose_free[3][name]) > 0


def test_routed_topk_forward_and_all_gradients_match_dense_oracle():
    experts, lora, inputs, expert_ids, routes, grad_output = _make_problem()
    before_hash = base_storage_hash(experts)
    dense = run_routed_reference(
        inputs,
        expert_ids,
        routes,
        experts,
        lora,
        grad_output,
        lora_scaling=2.0,
        swiglu_limit=10.0,
        transpose_free_backward=False,
    )
    transpose_free = run_routed_reference(
        inputs,
        expert_ids,
        routes,
        experts,
        lora,
        grad_output,
        lora_scaling=2.0,
        swiglu_limit=10.0,
        transpose_free_backward=True,
    )
    after_hash = base_storage_hash(experts)

    for name, actual, expected in (
        ("forward", transpose_free[0], dense[0]),
        ("dX", transpose_free[1], dense[1]),
        ("dRoute", transpose_free[2], dense[2]),
    ):
        relative_l2, cosine = relative_l2_and_cosine(actual, expected)
        assert relative_l2 <= 2.0e-6, f"{name}: relative L2={relative_l2}"
        assert cosine >= 0.999999, f"{name}: cosine={cosine}"
    for name in LORA_NAMES:
        actual = transpose_free[3][name]
        expected = dense[3][name]
        relative_l2, cosine = relative_l2_and_cosine(actual, expected)
        assert relative_l2 <= 2.0e-6, f"{name}: relative L2={relative_l2}"
        assert cosine >= 0.999999, f"{name}: cosine={cosine}"
        assert torch.count_nonzero(actual[:3]) > 0, (
            f"{name}: active expert gradient is zero"
        )
        assert torch.count_nonzero(actual[3]) == 0, (
            f"{name}: inactive expert gradient is nonzero"
        )
    assert torch.count_nonzero(transpose_free[2]) == transpose_free[2].numel()
    assert before_hash == after_hash


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
