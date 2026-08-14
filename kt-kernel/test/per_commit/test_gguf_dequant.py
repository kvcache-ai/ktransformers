#!/usr/bin/env python
# coding=utf-8
"""GGUF strip dequantization tests: kt::gguf::dequant_rows_bf16 vs ggml.

For every supported GGML type the C++ strip dequant must be BIT-IDENTICAL to
`ggml_internal_get_type_traits(type).to_float` followed by
`ggml_fp32_to_bf16` (round-to-nearest-even). Covers full rows, tail rows,
non-multiple-of-N_BLOCK row counts, block-aligned and non-aligned column
slices, and the generic ggml fallback for unsupported types.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="default")

import torch
import kt_kernel

kt_kernel_ext = kt_kernel.kt_kernel_ext
# ggml_table_f32_f16 is all zeros until ggml_init() runs; the quantized-type
# reference below (utils.to_float) reads it, so initialize before any tests.
kt_kernel_ext.utils.ggml_init()
utils = kt_kernel_ext.utils
moe = kt_kernel_ext.moe
from kt_kernel.utils.loader import GGMLQuantizationType

# (type, k, nrows) — k and nrows cover the realistic model shapes plus
# tail/strip-boundary cases. Q4_0/Q8_K ride the generic ggml fallback.
CASES = [
    (GGMLQuantizationType.Q4_K, 7168, 5),
    (GGMLQuantizationType.Q4_K, 7168, 80),  # non-multiple of N_BLOCK=64
    (GGMLQuantizationType.Q4_K, 256, 64),
    (GGMLQuantizationType.Q5_K, 1024, 3),
    (GGMLQuantizationType.Q5_K, 512, 65),
    (GGMLQuantizationType.Q6_K, 7168, 2),
    (GGMLQuantizationType.Q6_K, 768, 7),
    (GGMLQuantizationType.Q8_0, 7168, 4),
    (GGMLQuantizationType.Q8_0, 512, 80),
    (GGMLQuantizationType.F16, 7168, 3),
    (GGMLQuantizationType.BF16, 7168, 3),
    (GGMLQuantizationType.F32, 7168, 3),
    (GGMLQuantizationType.Q4_0, 1024, 5),  # fallback path
    (GGMLQuantizationType.Q8_K, 1024, 5),  # fallback path (f32 super-scale)
]

# Column-slice cases per type: (col_begin, col_end)
SLICES = [
    (0, 0),  # sentinel: full columns
    (256, 512),  # block-aligned
    (100, 200),  # non-aligned -> generic path
]


def _ggml_type(t):
    """Wrap an int into the pybind ggml_type enum (kt_kernel_ext.kvcache)."""
    return kt_kernel_ext.kvcache.ggml_type(int(t))


def quantize(f32: torch.Tensor, ggml_type: GGMLQuantizationType) -> torch.Tensor:
    return utils.from_float(f32.data_ptr(), f32.numel(), _ggml_type(ggml_type))


def reference_bf16(raw: torch.Tensor, ggml_type: GGMLQuantizationType, n: int) -> torch.Tensor:
    # Passthrough types: convert directly (fp16->fp32 and bf16->fp32 are exact;
    # the f32->bf16 tail is RNE in both torch and ggml for normal values).
    if ggml_type == GGMLQuantizationType.F16:
        u16 = raw.view(torch.uint16)[:n].clone().view(torch.float16).float()
        return u16.to(torch.bfloat16)
    if ggml_type == GGMLQuantizationType.BF16:
        return raw.view(torch.uint16)[:n].clone().view(torch.bfloat16).float().to(torch.bfloat16)
    if ggml_type == GGMLQuantizationType.F32:
        return raw.view(torch.float32)[:n].clone().to(torch.bfloat16)
    # Quantized types: ggml to_float -> RNE bf16.
    f32 = utils.to_float(raw.data_ptr(), n, _ggml_type(ggml_type))
    return f32.to(torch.bfloat16)


def dequant(raw: torch.Tensor, ggml_type: GGMLQuantizationType, k, r0, r1, c0, c1) -> torch.Tensor:
    return moe.dequant_rows_bf16(raw.data_ptr(), int(ggml_type), k, r0, r1, c0, c1)


def check_type(ggml_type, k, nrows, seed):
    torch.manual_seed(seed)
    # Realistic weight distribution: small magnitudes, some outliers.
    f32 = torch.randn(nrows, k, dtype=torch.float32) * 0.02
    # ensure some values that stress the fp16 scale rounding (both signs, wide range)
    if nrows > 0 and k >= 256:
        f32[0, :256] = torch.randn(256, dtype=torch.float32) * 3.0
    raw = quantize(f32, ggml_type)

    # --- full columns ---
    got = dequant(raw, ggml_type, k, 0, nrows, 0, k)
    ref = reference_bf16(raw, ggml_type, nrows * k).view(nrows, k)
    assert got.shape == (nrows, k), f"{ggml_type.name}: shape {got.shape} != {(nrows, k)}"
    assert torch.equal(got, ref), f"{ggml_type.name}: full-column dequant not bit-exact with ggml"

    # --- row sub-ranges (tail rows, non-multiple-of-64) ---
    ref_full = ref  # keep the full reference; slices below must not rebind it
    for (r0, r1) in [(1, 3), (nrows - 2, nrows), (0, min(nrows, 63)), (min(nrows, 15), nrows)]:
        r1 = min(r1, nrows)  # never read past the tensor (e.g. nrows=2 vs [1,3))
        if r0 >= r1:
            continue
        got = dequant(raw, ggml_type, k, r0, r1, 0, k)
        ref = ref_full[r0:r1, :]
        assert torch.equal(got, ref), f"{ggml_type.name}: rows [{r0},{r1}) not bit-exact"

    # --- column slices ---
    dcol = k // 2
    for (c0, c1) in [(0, dcol), (dcol, k), (256, 512), (100, 200)]:
        c0 = max(0, min(c0, k))
        c1 = max(c0, min(c1, k))
        if c0 >= c1:
            continue
        got = dequant(raw, ggml_type, k, 0, nrows, c0, c1)
        ref = ref_full[:, c0:c1]
        assert got.shape == (nrows, c1 - c0), f"{ggml_type.name}: slice shape {got.shape}"
        assert torch.equal(got, ref), f"{ggml_type.name}: cols [{c0},{c1}) not bit-exact"

    # --- row + column combined (down-projection NUMA slice shape) ---
    if k >= 512:
        c0, c1 = 256, min(768, k)  # block-aligned window within the row
        r0, r1 = 1, min(3, nrows)
        if c1 - c0 >= 256 and r0 < r1:
            got = dequant(raw, ggml_type, k, r0, r1, c0, c1)
            ref = ref_full[r0:r1, c0:c1]
            assert torch.equal(got, ref), f"{ggml_type.name}: combined rows+cols not bit-exact"


def test_gguf_dequant_bit_exact():
    for i, (ggml_type, k, nrows) in enumerate(CASES):
        check_type(ggml_type, k, nrows, seed=1000 + i)


def test_gguf_dequant_empty_ranges():
    """Empty / degenerate ranges must be a no-op, not a crash."""
    raw = torch.zeros(1, dtype=torch.uint8)
    out = moe.dequant_rows_bf16(raw.data_ptr(), int(GGMLQuantizationType.Q4_K), 256, 3, 3, 0, 256)
    assert out.shape == (0, 256)
    out = moe.dequant_rows_bf16(raw.data_ptr(), int(GGMLQuantizationType.Q4_K), 256, 0, 4, 5, 5)
    assert out.shape == (4, 0)


def run_all_tests():
    test_gguf_dequant_bit_exact()
    test_gguf_dequant_empty_ranges()
    print("✓ all gguf dequant tests passed")


if __name__ == "__main__":
    run_all_tests()