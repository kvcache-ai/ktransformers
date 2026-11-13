#!/usr/bin/env python
"""Compare native block-FP8 LoRA layouts without changing quantized weights.

--matrix --dump/--compare also compare independently built baseline/candidate
extensions. The shared lifecycle harness checks forward, dX, router gradients,
six LoRA gradients, AdamW updates, inactive experts, and output guard regions.
"""

import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch

if __package__:
    from . import test_sft_bf16_recompute_buffers as lifecycle
else:
    import test_sft_bf16_recompute_buffers as lifecycle
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="default")
ext = lifecycle.ext


def _fp8_layer(cpu, *, shared, rank, top_k, dropout, layer, max_len, authoritative=True):
    e, h, i = lifecycle.E, lifecycle.H, lifecycle.I
    generator = torch.Generator().manual_seed(1729 + layer)
    weights = {}
    for projection, shape in (("gate", (e, i, h)), ("up", (e, i, h)), ("down", (e, h, i))):
        weights[f"{projection}_proj"] = lifecycle._random(shape, generator).to(torch.float8_e4m3fn)
        # Nonunit, distinct scales across BOTH block axes exercise FP8 reduction
        # and TP slicing. Comparing unit scales would miss stale reduce_buf views.
        scales = 0.25 + torch.rand((e, shape[1] // 128, shape[2] // 128), generator=generator)
        weights[f"{projection}_scale"] = scales.contiguous()
    for name, shape in zip(
        lifecycle.NAMES, ((e, rank, h), (e, i, rank), (e, rank, h), (e, i, rank), (e, rank, i), (e, h, rank))
    ):
        weights[name] = lifecycle._random(shape, generator)
    mapping = torch.arange(e, dtype=torch.int64)
    config = ext.moe.MOESFTConfig(e, top_k, h, i)
    config.max_len = max_len
    config.max_cache_depth = 1
    config.layer_idx = layer
    config.lora_rank = rank
    config.lora_alpha = 2.0 * rank
    config.lora_dropout = dropout
    config.full_weight_grad = False
    config.authoritative_optimizer_grads = authoritative
    config.share_backward_bb = True
    config.share_cache_pool = shared
    config.quant_config.bits = 8
    config.quant_config.group_size = 128
    config.quant_config.zero_point = False
    config.physical_to_logical_map = mapping.data_ptr()
    for name, tensor in weights.items():
        setattr(config, name, tensor.data_ptr())
    config.pool = cpu.backend_
    moe = ext.moe.AMXFP8_SFT_MOE(config)
    moe.load_weights()
    grads = {name: torch.full_like(weights[name], 7.0 if authoritative else 0.0) for name in lifecycle.NAMES}
    return moe, weights, grads, mapping


def run_case(*case, authoritative=True, mixed=None):
    frozen = []

    def make_layer(cpu, **kwargs):
        use_bf16 = mixed is not None and (kwargs["layer"] % 2 == (0 if mixed == "bf16-first" else 1))
        factory = lifecycle._layer if use_bf16 else _fp8_layer
        result = factory(cpu, **kwargs)
        for name, tensor in result[1].items():
            if name not in lifecycle.NAMES:
                # Compare raw bytes, including FP8 payloads and FP32 scales.
                frozen.append((name, tensor, tensor.view(torch.uint8).clone()))
        return result

    values = lifecycle.run_case(*case, authoritative=authoritative, layer_factory=make_layer)
    for name, tensor, before in frozen:
        assert torch.equal(tensor.view(torch.uint8), before), f"{name}: frozen base/scales changed"
    return values


@pytest.mark.cpu
@pytest.mark.parametrize("case", lifecycle.CASES)
def test_fp8_recompute_buffers(case):
    if ext is None or not hasattr(ext.moe, "AMXFP8_SFT_MOE"):
        pytest.skip("FP8 SFT kernel unavailable")
    run_case(*case)


@pytest.mark.cpu
@pytest.mark.parametrize("mode", ("enabled", "legacy", "fp8-first", "bf16-first"))
def test_fp8_recompute_workspace_policy_and_growth(mode):
    if ext is None or not hasattr(ext.moe, "AMXFP8_SFT_MOE"):
        pytest.skip("FP8 SFT kernel unavailable")
    # Fresh arenas make mixed-format growth observable regardless of test order.
    subprocess.run([sys.executable, __file__, "--check-workspace-policy", mode], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--dump")
    parser.add_argument("--compare")
    parser.add_argument("--check-workspace-policy", choices=("enabled", "legacy", "fp8-first", "bf16-first"))
    args = parser.parse_args()
    if args.check_workspace_policy:
        mode = args.check_workspace_policy
        enabled = mode != "legacy"
        mixed = mode if mode.endswith("first") else None
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "pools.log"
            os.environ["SFT_POOL_LOG"] = "1"
            os.environ["SFT_POOL_LOG_FILE"] = str(log)
            run_case(2, 4, 8, 8, 4, 0.1, authoritative=enabled, mixed=mixed)
            assert ("recompute_work" in log.read_text()) == enabled
        print(f"PASS workspace policy {mode}", flush=True)
        sys.exit(0)
    cases = (
        list(itertools.product((1, 2), (1, 4), (1, 8), (8, 16, 32), (1, 4), (0.0, 0.1)))
        if args.matrix
        else lifecycle.CASES
    )
    reference = torch.load(args.compare, weights_only=True) if args.compare else None
    result = {}
    for case in cases:
        values = run_case(*case)
        key = str(case)
        if reference is not None:
            lifecycle._check(values, reference[key], case[0] == case[1] == 1)
        result[key] = values
        print(f"PASS {case}", flush=True)
    if args.dump:
        torch.save(result, args.dump)
