# SPDX-License-Identifier: Apache-2.0
"""RAWINT4 group-32 SFT numerical and scratch-alignment regression."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "rawint4_reference_fixture", Path(__file__).with_name("test_sft_int8_numerical.py")
)
reference = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reference
_spec.loader.exec_module(reference)

reference.register_cpu_ci(est_time=30, suite="default")


def _quantize(base):
    torch = reference.torch
    packed, scales, dequantized = {}, {}, {}
    for name, weight in base.items():
        blocks = weight.float().reshape(*weight.shape[:-1], -1, 32)
        scale = (blocks.abs().amax(-1).clamp_min(1e-6) / 7).to(torch.bfloat16)
        quant = (
            (blocks / scale.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
        )
        dequantized[name] = (
            (quant.float() * scale.float().unsqueeze(-1))
            .reshape(weight.shape)
            .to(torch.bfloat16)
        )
        # Compressed-tensors encodes signed values with a +8 nibble offset.
        quant = (quant.reshape(weight.shape) + 8).to(torch.uint8)
        packed[name] = (quant[..., 0::2] | (quant[..., 1::2] << 4)).contiguous()
        scales[name] = scale.contiguous()
    return packed, scales, dequantized


def _run_contract(experts, qlen):
    import kt_kernel

    torch = reference.torch
    torch.set_num_threads(1)
    torch.manual_seed(20260728)
    reference.EXPERTS = experts
    reference.QLEN = qlen
    reference.GRAD_SHAPES = {
        name: (experts, *shape[1:]) for name, shape in reference.GRAD_SHAPES.items()
    }
    base, lora = reference._make_weights()
    packed, scales, dequantized = _quantize(base)
    frozen = packed | {f"{name}_scale": value for name, value in scales.items()}
    originals = {name: value.clone() for name, value in frozen.items()}
    cpu, tp, numa = reference._make_cpu_infer()
    mapping = torch.arange(experts, dtype=torch.int64)
    config = kt_kernel.kt_kernel_ext.moe.MOESFTConfig(
        experts, reference.TOP_K, reference.HIDDEN, reference.INTERMEDIATE
    )
    config.max_len = reference.MAX_QLEN
    config.max_cache_depth = 1
    config.layer_idx = 0
    config.lora_rank = reference.RANK
    config.lora_alpha = reference.ALPHA
    config.full_weight_grad = False
    config.authoritative_optimizer_grads = True
    config.share_backward_bb = False
    config.share_cache_pool = False
    config.physical_to_logical_map = mapping.data_ptr()
    config.pool = cpu.backend_
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    for name in ("gate", "up", "down"):
        setattr(config, f"{name}_proj", packed[name].data_ptr())
        setattr(config, f"{name}_scale", scales[name].data_ptr())
    for name, parameter in lora.items():
        setattr(config, name, parameter.data_ptr())
    moe = kt_kernel.kt_kernel_ext.moe.AMXInt4_KGroup_SFT_MOE(config)
    moe.load_weights()
    gradients = reference._make_grad_buffers()
    batches = [
        reference._make_batch(expert, 100 + index)
        for index, expert in enumerate((0, 1, 2, 0, 0))
    ]
    references = [reference._reference(dequantized, lora, batch) for batch in batches]
    summary = {}
    for index, (batch, expected) in enumerate(zip(batches, references)):
        output, inputs, routes = reference._run_cpp_batch(
            moe, gradients, batch, accumulate=index == 1, grad_scale=0.5
        )
        for name, got, golden in (
            ("forward", output, expected[0]),
            ("input", inputs, expected[1]),
            ("router", routes, expected[2]),
        ):
            summary[f"{index}.{name}"] = reference._assert_tensor_close(
                name, got, golden, relative_l2_limit=0.15, cosine_limit=0.985
            )
        expected_grads = reference._expected_window(
            *([references[0][3], expected[3]] if index == 1 else [expected[3]]),
            scale=0.5,
        )
        summary[f"{index}.parameters"] = reference._assert_window(
            str(index),
            gradients,
            expected_grads,
            {0, 1} if index == 1 else {batch.expert},
        )
    for name, value in frozen.items():
        assert torch.equal(originals[name], value), f"Frozen base changed: {name}"
    print(
        "KT_RAWINT4_SFT_SUMMARY="
        + json.dumps(
            dict(
                cpu_variant=kt_kernel.__cpu_variant__,
                tp=tp,
                numa=numa,
                experts=experts,
                qlen=qlen,
                summary=summary,
                frozen_base_unchanged=True,
            )
        )
    )


@pytest.mark.parametrize("experts", [3, 4])
@pytest.mark.parametrize("qlen", [4, 17])
def test_rawint4_sft_numerical_and_pool_alignment(experts, qlen):
    if reference._IMPORT_ERROR is not None:
        pytest.skip(reference._IMPORT_ERROR)
    if not hasattr(reference.kt_kernel_ext.moe, "AMXInt4_KGroup_SFT_MOE"):
        pytest.skip("RAWINT4 SFT requires an AMX or AVX512-BF16 build")
    # Isolate native faults so an alignment regression fails a test, not the suite.
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--experts",
            str(experts),
            "--qlen",
            str(qlen),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KT_RAWINT4_SFT_SUMMARY=" in result.stdout


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, choices=(3, 4), required=True)
    parser.add_argument("--qlen", type=int, choices=(4, 17), required=True)
    args = parser.parse_args()
    if reference._IMPORT_ERROR is not None:
        raise SystemExit(reference._IMPORT_ERROR)
    _run_contract(args.experts, args.qlen)
