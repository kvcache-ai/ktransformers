#!/usr/bin/env python
# coding=utf-8
"""RV64 (CPU-only) tests for the Linear / MLP operators.

Complements test_ops_rv64.py: these exercise the llamafile_sgemm_rv64
dispatch paths (F16 / BF16 / Q8_0 / Q4_0 via tinyBLAS RVV templates and the
kt_ggml_compat extern "C" quantize_row_* symbols) through the Linear/MLP
classes that were wired up in ext_bindings.cpp for the RV bring-up.

Adapted from examples/test_linear.py / test_mlp.py with:
  - no CUDA (weights generated directly on CPU)
  - dtype parameterization (the x86 examples only test F16)
  - pytest + standalone runner, CI registration

Run (on the RV box):
    cd kt-kernel
    PYTHONPATH=$PWD:$PWD/test LD_LIBRARY_PATH=$PWD/build:/usr/local/kt-libs \
        python3 -m pytest test/per_commit/test_linear_mlp_rv64.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="default")

torch.manual_seed(42)

STRIDE = 32
GROUP_MAX_LEN = 1024
CPUINFER_THREADS = 48

GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0, GGML_BF16 = 0, 1, 2, 8, 30

_ggml = kt_kernel_ext.kvcache.ggml_type
_GGML_MAP = {
    GGML_F32: _ggml.FP32,
    GGML_F16: _ggml.FP16,
    GGML_Q4_0: _ggml.Q4_0,
    GGML_Q8_0: _ggml.Q8_0,
    GGML_BF16: _ggml.BF16,
}

TORCH_DTYPE_MAP = {
    GGML_F16: torch.float16,
    GGML_BF16: torch.bfloat16,
    GGML_F32: torch.float32,
}


def _make_weight(shape, dtype):
    """CPU-only weight tensor for the given ggml dtype.

    Quantized types (Q8_0/Q4_0) are produced via kt_kernel_ext.utils.from_float
    because the Linear/MLP operators read the weight buffer as ALREADY-quantized
    ggml blocks (type_size/blck_size layout).  Returns (buffer, ref) where ref
    is the dequantized float32 tensor for the torch reference (or the original
    fp16/bf16/f32 tensor for float dtypes).
    """
    w = torch.randn(shape, dtype=torch.float32) / 10.0
    if dtype in TORCH_DTYPE_MAP:
        return w.to(TORCH_DTYPE_MAP[dtype]).contiguous(), w
    # quantize via utils (Q8_0/Q4_0): returns a flat uint8 tensor of ggml blocks
    q = kt_kernel_ext.utils.from_float(w.contiguous().data_ptr(), w.numel(), _GGML_MAP[dtype])
    dq = kt_kernel_ext.utils.to_float(q.data_ptr(), w.numel(), _GGML_MAP[dtype])
    return q.contiguous(), dq.reshape(shape).contiguous()


def _mlp_ref(input_f32, gate, up, down):
    g = torch.mm(input_f32, gate.float().t())
    u = torch.mm(input_f32, up.float().t())
    inter = torch.nn.functional.silu(g) * u
    return torch.mm(inter, down.float().t())


# ---------------------------------------------------------------------------
# Linear across dtypes — llamafile_sgemm_rv64 proj branches
#
# NOTE on BF16: hidden_type must equal the ACTIVATION dtype (what the caller
# passes), while proj_type is the weight dtype.  Known issue (RV, recorded in
# RV_TEST_PLAN.md): hidden=FP16 + BF16 weights returns garbage (~0 values) —
# the activation conversion via the fp16 row helper appears uninitialized in
# this link configuration.  F32 activations cover the same BF16 sgemm branch
# and pass cleanly.
# ---------------------------------------------------------------------------

LINEAR_DTYPES = [
    pytest.param(GGML_F16, GGML_F16, 0.02, id="f16"),
    pytest.param(GGML_BF16, GGML_F32, 0.05, id="bf16-f32in"),
    pytest.param(GGML_Q8_0, GGML_F16, 0.03, id="q8_0"),
    pytest.param(GGML_Q4_0, GGML_F16, 0.10, id="q4_0"),
]


@pytest.mark.cpu
@pytest.mark.parametrize("dtype,hidden,tol", LINEAR_DTYPES)
def test_linear_rv64_dtype(dtype, hidden, tol):
    input_size, output_size = 2048, 1024
    qlen = 32
    iters = 3

    proj, proj_ref = _make_weight((output_size, input_size), dtype)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_THREADS)
    config = kt_kernel_ext.linear.LinearConfig(
        input_size, output_size, STRIDE, GROUP_MAX_LEN,
        proj.data_ptr(), _GGML_MAP[dtype], _GGML_MAP[hidden],
    )
    linear = kt_kernel_ext.linear.Linear(config)
    cpuinfer.submit(linear.warm_up())
    cpuinfer.sync()

    with torch.inference_mode():
        in_t = TORCH_DTYPE_MAP.get(hidden, torch.float16)
        for i in range(iters):
            x = (torch.randn((qlen, input_size)) / 10.0).to(in_t).contiguous()
            y = torch.empty((qlen, output_size), dtype=in_t).contiguous()
            cpuinfer.submit(linear.forward(qlen, x.data_ptr(), y.data_ptr()))
            cpuinfer.sync()

            ref = torch.mm(x.float(), proj_ref.float().t())
            diff = torch.mean(torch.abs(y.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)
            print(f"[linear {dtype} iter {i}] diff = {diff:.6f}")
            assert diff < tol, f"linear dtype={dtype}: diff={diff:.6f} >= {tol}"


# ---------------------------------------------------------------------------
# MLP across dtypes — gate/up/down sgemm + SwiGLU, incl. quantized vec_dot
# ---------------------------------------------------------------------------

MLP_DTYPES = [
    pytest.param(GGML_F16, GGML_F16, 0.05, id="f16"),
    pytest.param(GGML_BF16, GGML_F32, 0.05, id="bf16-f32in"),
    pytest.param(GGML_Q8_0, GGML_F16, 0.06, id="q8_0"),
    pytest.param(GGML_Q4_0, GGML_F16, 0.15, id="q4_0"),
]


@pytest.mark.cpu
@pytest.mark.parametrize("dtype,hidden,tol", MLP_DTYPES)
def test_mlp_rv64_dtype(dtype, hidden, tol):
    hidden_size, intermediate_size = 1024, 512
    qlen = 32
    iters = 3

    gate, gate_ref = _make_weight((intermediate_size, hidden_size), dtype)
    up, up_ref = _make_weight((intermediate_size, hidden_size), dtype)
    down, down_ref = _make_weight((hidden_size, intermediate_size), dtype)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_THREADS)
    config = kt_kernel_ext.mlp.MLPConfig(
        hidden_size, intermediate_size, STRIDE, GROUP_MAX_LEN,
        gate.data_ptr(), up.data_ptr(), down.data_ptr(),
        _GGML_MAP[dtype], _GGML_MAP[dtype], _GGML_MAP[dtype], _GGML_MAP[hidden],
    )
    mlp = kt_kernel_ext.mlp.MLP(config)
    cpuinfer.submit(mlp.warm_up())
    cpuinfer.sync()

    with torch.inference_mode():
        in_t = TORCH_DTYPE_MAP.get(hidden, torch.float16)
        for i in range(iters):
            x = (torch.randn((qlen, hidden_size)) / 100.0).to(in_t).contiguous()
            y = torch.empty((qlen, hidden_size), dtype=in_t).contiguous()
            cpuinfer.submit(mlp.forward(qlen, x.data_ptr(), y.data_ptr()))
            cpuinfer.sync()

            ref = _mlp_ref(x.float(), gate_ref, up_ref, down_ref)
            diff = torch.mean(torch.abs(y.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)
            print(f"[mlp {dtype} iter {i}] diff = {diff:.6f}")
            assert diff < tol, f"mlp dtype={dtype}: diff={diff:.6f} >= {tol}"


# ---------------------------------------------------------------------------
# Long-prefill shape (qlen > group_max_len exercises the looped forward path)
# ---------------------------------------------------------------------------

@pytest.mark.cpu
def test_mlp_rv64_long_qlen_f16():
    hidden_size, intermediate_size = 1024, 512
    qlen = GROUP_MAX_LEN + 64  # forces Linear/MLP::forward chunked recursion

    gate, gate_ref = _make_weight((intermediate_size, hidden_size), GGML_F16)
    up, up_ref = _make_weight((intermediate_size, hidden_size), GGML_F16)
    down, down_ref = _make_weight((hidden_size, intermediate_size), GGML_F16)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_THREADS)
    config = kt_kernel_ext.mlp.MLPConfig(
        hidden_size, intermediate_size, STRIDE, GROUP_MAX_LEN,
        gate.data_ptr(), up.data_ptr(), down.data_ptr(),
        _GGML_MAP[GGML_F16], _GGML_MAP[GGML_F16], _GGML_MAP[GGML_F16], _GGML_MAP[GGML_F16],
    )
    mlp = kt_kernel_ext.mlp.MLP(config)
    cpuinfer.submit(mlp.warm_up())
    cpuinfer.sync()

    with torch.inference_mode():
        x = (torch.randn((qlen, hidden_size)) / 100.0).to(torch.float16).contiguous()
        y = torch.empty((qlen, hidden_size), dtype=torch.float16).contiguous()
        cpuinfer.submit(mlp.forward(qlen, x.data_ptr(), y.data_ptr()))
        cpuinfer.sync()

        ref = _mlp_ref(x.float(), gate_ref, up_ref, down_ref)
        diff = torch.mean(torch.abs(y.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)
        print(f"[mlp long qlen={qlen}] diff = {diff:.6f}")
        assert diff < 0.05, f"mlp long-qlen: diff={diff:.6f}"


if __name__ == "__main__":
    print("=" * 60)
    print("RV64 CPU-only Linear/MLP tests (llamafile_sgemm_rv64 paths)")
    print("=" * 60)
    failures = 0
    for which, params, cases in [
        (test_linear_rv64_dtype, ("dtype", "hidden", "tol"), LINEAR_DTYPES),
        (test_mlp_rv64_dtype, ("dtype", "hidden", "tol"), MLP_DTYPES),
    ]:
        for dtype, hidden, tol in cases:
            try:
                which(dtype, hidden, tol)
                print(f"PASS {which.__name__} {dtype} hidden={hidden}")
            except Exception as e:
                failures += 1
                print(f"FAIL {which.__name__} {dtype} hidden={hidden}: {e}")
                import traceback
                traceback.print_exc()
    try:
        test_mlp_rv64_long_qlen_f16()
        print("PASS mlp long-qlen")
    except Exception as e:
        failures += 1
        print(f"FAIL mlp long-qlen: {e}")
        import traceback
        traceback.print_exc()
    sys.exit(1 if failures else 0)
