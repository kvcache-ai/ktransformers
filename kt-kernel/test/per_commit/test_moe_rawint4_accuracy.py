#!/usr/bin/env python
# coding=utf-8
"""RAWINT4 MoE accuracy tests for KT-Kernel x86 backends."""

import importlib.util
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

register_cpu_ci(est_time=120, suite="default")

import pytest
import torch
import kt_kernel_ext

KT_KERNEL_ROOT = Path(__file__).resolve().parents[2]

expert_num = 8
hidden_size = 256
intermediate_size = 512
num_experts_per_tok = 2
max_len = 512
group_size = 128
validation_iter = 3
CPUINFER_PARAM = 16


def load_amx_utils():
    pkg_root = KT_KERNEL_ROOT / "python"
    utils_root = pkg_root / "utils"

    if "kt_kernel" not in sys.modules:
        kt_kernel_pkg = types.ModuleType("kt_kernel")
        kt_kernel_pkg.__path__ = [str(pkg_root)]
        kt_kernel_pkg.kt_kernel_ext = kt_kernel_ext
        sys.modules["kt_kernel"] = kt_kernel_pkg

    if "kt_kernel_ext" not in sys.modules:
        sys.modules["kt_kernel_ext"] = kt_kernel_ext

    if "kt_kernel.utils" not in sys.modules:
        utils_pkg = types.ModuleType("kt_kernel.utils")
        utils_pkg.__path__ = [str(utils_root)]
        sys.modules["kt_kernel.utils"] = utils_pkg

    module_specs = [
        ("kt_kernel.experts_base", pkg_root / "experts_base.py"),
        ("kt_kernel.utils.loader", utils_root / "loader.py"),
        ("kt_kernel.utils.amx", utils_root / "amx.py"),
    ]
    for module_name, module_path in module_specs:
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules["kt_kernel.utils.amx"]


def rawint4_quantize(weight_bf16, quant_group_size=group_size):
    """Quantize [N, K] BF16 weight to RAWINT4 layout."""
    n, k = weight_bf16.shape
    assert k % 2 == 0
    assert k % quant_group_size == 0

    weight_fp32 = weight_bf16.float()
    qweight = torch.zeros((n, k // 2), dtype=torch.uint8)
    scales = torch.zeros((n, k // quant_group_size), dtype=torch.bfloat16)

    for ni in range(n):
        for g in range(k // quant_group_size):
            k_start = g * quant_group_size
            k_end = k_start + quant_group_size
            block = weight_fp32[ni, k_start:k_end]
            amax = block.abs().max().item()
            scale = amax / 7.0 if amax > 0 else 1.0
            scales[ni, g] = scale

            for kk in range(k_start, k_end, 2):
                q0 = int(round(weight_fp32[ni, kk].item() / scale)) + 8
                q1 = int(round(weight_fp32[ni, kk + 1].item() / scale)) + 8
                q0 = max(0, min(15, q0))
                q1 = max(0, min(15, q1))
                qweight[ni, kk // 2] = (q1 << 4) | q0

    return qweight, scales


def rawint4_dequantize(qweight, scales, out_features, in_features, quant_group_size=group_size):
    """Dequantize RAWINT4 qweight/scales back to fp32 [N, K]."""
    result = torch.zeros((out_features, in_features), dtype=torch.float32)
    for ni in range(out_features):
        for g in range(in_features // quant_group_size):
            scale = scales[ni, g].float().item()
            k_start = g * quant_group_size
            k_end = k_start + quant_group_size
            for kk in range(k_start, k_end, 2):
                packed = int(qweight[ni, kk // 2].item())
                result[ni, kk] = ((packed & 0x0F) - 8) * scale
                result[ni, kk + 1] = (((packed >> 4) & 0x0F) - 8) * scale
    return result


def pack_rawint4_uint8_as_int32(qweight):
    """Pack byte RAWINT4 layout into compressed-tensors int32 storage."""
    assert qweight.dtype == torch.uint8
    assert qweight.shape[1] % 4 == 0
    return qweight.contiguous().view(torch.int32).contiguous()


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input_data, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input_data, gate_proj.t())
    up_buf = torch.mm(input_data, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input_data, expert_ids, weights, gate_proj, up_proj, down_proj):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input_data[idxs // expert_ids.shape[1]]
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens = sorted_tokens[start_idx:end_idx]
        out = mlp_torch(tokens, gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(out)
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (new_x.view(*expert_ids.shape, -1).float().mul_(weights.unsqueeze(-1)).sum(1)).to(new_x.dtype)


def available_backends():
    backends = []
    if hasattr(kt_kernel_ext.moe, "AVX2RawInt4_MOE"):
        backends.append(("AVX2RawInt4_MOE", kt_kernel_ext.moe.AVX2RawInt4_MOE, 0.12))

    if hasattr(kt_kernel_ext.moe, "AVXVNNI256RawInt4_MOE"):
        has_avx_vnni = False
        try:
            with open("/proc/cpuinfo", "r") as f:
                has_avx_vnni = any(("avx_vnni" in line or "avxvnni" in line) for line in f if line.startswith("flags"))
        except OSError:
            has_avx_vnni = False
        if has_avx_vnni:
            backends.append(("AVXVNNI256RawInt4_MOE", kt_kernel_ext.moe.AVXVNNI256RawInt4_MOE, 0.20))
    return backends


def run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen, quant_group_size=group_size):
    physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
    cpu_infer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    with torch.inference_mode():
        gate_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        up_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        down_bf16 = (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )

        gate_qw_list, gate_scale_list = [], []
        up_qw_list, up_scale_list = [], []
        down_qw_list, down_scale_list = [], []

        for e in range(expert_num):
            qw, sc = rawint4_quantize(gate_bf16[e], quant_group_size)
            gate_qw_list.append(qw)
            gate_scale_list.append(sc)

            qw, sc = rawint4_quantize(up_bf16[e], quant_group_size)
            up_qw_list.append(qw)
            up_scale_list.append(sc)

            qw, sc = rawint4_quantize(down_bf16[e], quant_group_size)
            down_qw_list.append(qw)
            down_scale_list.append(sc)

        gate_qw = torch.stack(gate_qw_list).contiguous()
        gate_scales = torch.stack(gate_scale_list).contiguous()
        up_qw = torch.stack(up_qw_list).contiguous()
        up_scales = torch.stack(up_scale_list).contiguous()
        down_qw = torch.stack(down_qw_list).contiguous()
        down_scales = torch.stack(down_scale_list).contiguous()

        gate_deq = torch.stack(
            [
                rawint4_dequantize(gate_qw_list[e], gate_scale_list[e], intermediate_size, hidden_size, quant_group_size)
                for e in range(expert_num)
            ]
        )
        up_deq = torch.stack(
            [
                rawint4_dequantize(up_qw_list[e], up_scale_list[e], intermediate_size, hidden_size, quant_group_size)
                for e in range(expert_num)
            ]
        )
        down_deq = torch.stack(
            [
                rawint4_dequantize(down_qw_list[e], down_scale_list[e], hidden_size, intermediate_size, quant_group_size)
                for e in range(expert_num)
            ]
        )

        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max(max_len, qlen)
        config.gate_proj = gate_qw.data_ptr()
        config.up_proj = up_qw.data_ptr()
        config.down_proj = down_qw.data_ptr()
        config.gate_scale = gate_scales.data_ptr()
        config.up_scale = up_scales.data_ptr()
        config.down_scale = down_scales.data_ptr()
        config.quant_config.bits = 4
        config.quant_config.group_size = quant_group_size
        config.quant_config.zero_point = False
        config.pool = cpu_infer.backend_

        moe = backend_cls(config)
        cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        cpu_infer.sync()

        print(f"\n--- {backend_name} (qlen={qlen}) ---")
        for i in range(validation_iter):
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)
            cpu_infer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            cpu_infer.sync()

            ref_output = moe_torch(input_data.float(), expert_ids, weights, gate_deq, up_deq, down_deq).to(
                torch.bfloat16
            )
            diff = torch.mean(torch.abs(output.float() - ref_output.float())) / (
                torch.mean(torch.abs(ref_output.float())) + 1e-8
            )
            print(f"  Iteration {i}: diff = {diff.item():.6f}")
            assert diff < threshold, f"{backend_name} accuracy test failed: diff={diff.item():.6f} >= {threshold}"


def test_rawint4_accuracy():
    backends = available_backends()
    if not backends:
        print("Skipping RAWINT4 accuracy tests: no x86 RAWINT4 backend available")
        return

    for backend_name, backend_cls, threshold in backends:
        run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen=1)
        run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen=16)


def test_amxint4_kgroup_accuracy():
    if not hasattr(kt_kernel_ext.moe, "AMXInt4_KGroup_MOE"):
        pytest.skip("AMXInt4_KGroup_MOE is not available")

    backend_cls = kt_kernel_ext.moe.AMXInt4_KGroup_MOE
    run_backend_accuracy_test("AMXInt4_KGroup_MOE", backend_cls, 0.20, qlen=1, quant_group_size=32)
    run_backend_accuracy_test("AMXInt4_KGroup_MOE", backend_cls, 0.20, qlen=32, quant_group_size=32)
    run_backend_accuracy_test("AMXInt4_KGroup_MOE", backend_cls, 0.20, qlen=128, quant_group_size=32)


def test_amxint4_kgroup_blocked_accuracy():
    if not hasattr(kt_kernel_ext.moe, "AMXInt4_KGroupBlocked_MOE"):
        pytest.skip("AMXInt4_KGroupBlocked_MOE is not available")

    backend_cls = kt_kernel_ext.moe.AMXInt4_KGroupBlocked_MOE
    run_backend_accuracy_test("AMXInt4_KGroupBlocked_MOE", backend_cls, 0.20, qlen=1, quant_group_size=32)
    run_backend_accuracy_test("AMXInt4_KGroupBlocked_MOE", backend_cls, 0.20, qlen=32, quant_group_size=32)
    run_backend_accuracy_test("AMXInt4_KGroupBlocked_MOE", backend_cls, 0.20, qlen=128, quant_group_size=32)
    run_backend_accuracy_test("AMXInt4_KGroupBlocked_MOE", backend_cls, 0.20, qlen=512, quant_group_size=32)


def _make_writer_inputs(quant_group_size=32):
    torch.manual_seed(1234)
    gate_qw = torch.randint(0, 256, (expert_num, intermediate_size, hidden_size // 2), dtype=torch.uint8).contiguous()
    up_qw = torch.randint(0, 256, (expert_num, intermediate_size, hidden_size // 2), dtype=torch.uint8).contiguous()
    down_qw = torch.randint(0, 256, (expert_num, hidden_size, intermediate_size // 2), dtype=torch.uint8).contiguous()
    gate_scales = torch.rand((expert_num, intermediate_size, hidden_size // quant_group_size), dtype=torch.float32).to(
        torch.bfloat16
    ).contiguous()
    up_scales = torch.rand((expert_num, intermediate_size, hidden_size // quant_group_size), dtype=torch.float32).to(
        torch.bfloat16
    ).contiguous()
    down_scales = torch.rand((expert_num, hidden_size, intermediate_size // quant_group_size), dtype=torch.float32).to(
        torch.bfloat16
    ).contiguous()
    return gate_qw, up_qw, down_qw, gate_scales, up_scales, down_scales


def _load_writer_backend(backend_cls, tensors, quant_group_size=32):
    gate_qw, up_qw, down_qw, gate_scales, up_scales, down_scales = tensors
    cpu_infer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    config.max_len = max_len
    config.gate_proj = gate_qw.data_ptr()
    config.up_proj = up_qw.data_ptr()
    config.down_proj = down_qw.data_ptr()
    config.gate_scale = gate_scales.data_ptr()
    config.up_scale = up_scales.data_ptr()
    config.down_scale = down_scales.data_ptr()
    config.quant_config.bits = 4
    config.quant_config.group_size = quant_group_size
    config.quant_config.zero_point = False
    config.pool = cpu_infer.backend_

    moe = backend_cls(config)
    physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
    cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpu_infer.sync()
    return cpu_infer, moe


def _export_writer_buffers(cpu_infer, moe, gpu_tp_count, quant_group_size=32, gpu_experts=2):
    per_mat_weight_bytes = intermediate_size * hidden_size // 2
    per_mat_scale_elems = intermediate_size * (hidden_size // quant_group_size)
    weight_bytes_per_expert_per_tp = per_mat_weight_bytes // gpu_tp_count
    scale_elems_per_expert_per_tp = per_mat_scale_elems // gpu_tp_count

    w13_weight_bufs = [torch.empty(2 * gpu_experts * weight_bytes_per_expert_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w13_scale_bufs = [torch.empty(2 * gpu_experts * scale_elems_per_expert_per_tp, dtype=torch.bfloat16) for _ in range(gpu_tp_count)]
    w2_weight_bufs = [torch.empty(gpu_experts * weight_bytes_per_expert_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w2_scale_bufs = [torch.empty(gpu_experts * scale_elems_per_expert_per_tp, dtype=torch.bfloat16) for _ in range(gpu_tp_count)]

    for expert_id in range(gpu_experts):
        w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs = [], [], [], []
        for tp_idx in range(gpu_tp_count):
            w13_weight_ptrs.append(w13_weight_bufs[tp_idx].data_ptr() + expert_id * 2 * weight_bytes_per_expert_per_tp)
            w13_scale_ptrs.append(
                w13_scale_bufs[tp_idx].data_ptr() + expert_id * 2 * scale_elems_per_expert_per_tp * 2
            )
            w2_weight_ptrs.append(w2_weight_bufs[tp_idx].data_ptr() + expert_id * weight_bytes_per_expert_per_tp)
            w2_scale_ptrs.append(w2_scale_bufs[tp_idx].data_ptr() + expert_id * scale_elems_per_expert_per_tp * 2)

        cpu_infer.submit(
            moe.write_weight_scale_to_buffer_task(
                gpu_tp_count=gpu_tp_count,
                expert_id=expert_id,
                w13_weight_ptrs=w13_weight_ptrs,
                w13_scale_ptrs=w13_scale_ptrs,
                w2_weight_ptrs=w2_weight_ptrs,
                w2_scale_ptrs=w2_scale_ptrs,
            )
        )
        cpu_infer.sync()

    return w13_weight_bufs, w13_scale_bufs, w2_weight_bufs, w2_scale_bufs


def test_amxint4_kgroup_blocked_write_buffer_matches_rowmajor_backend():
    if not hasattr(kt_kernel_ext.moe, "AMXInt4_KGroup_MOE") or not hasattr(
        kt_kernel_ext.moe, "AMXInt4_KGroupBlocked_MOE"
    ):
        pytest.skip("AMX RAWINT4 row-major and blocked backends are both required")

    tensors = _make_writer_inputs(quant_group_size=32)
    old_cpu, old_moe = _load_writer_backend(kt_kernel_ext.moe.AMXInt4_KGroup_MOE, tensors, quant_group_size=32)
    blocked_cpu, blocked_moe = _load_writer_backend(
        kt_kernel_ext.moe.AMXInt4_KGroupBlocked_MOE, tensors, quant_group_size=32
    )

    for gpu_tp_count in (1, 2, 4):
        old_buffers = _export_writer_buffers(old_cpu, old_moe, gpu_tp_count, quant_group_size=32)
        blocked_buffers = _export_writer_buffers(blocked_cpu, blocked_moe, gpu_tp_count, quant_group_size=32)
        for old_group, blocked_group in zip(old_buffers, blocked_buffers):
            for old_buf, blocked_buf in zip(old_group, blocked_group):
                assert torch.equal(blocked_buf, old_buf)


def test_compressed_loader_normalizes_int32_pack_quantized_weights():
    load_amx_utils()
    loader_mod = sys.modules["kt_kernel.utils.loader"]

    weight_bf16 = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(torch.bfloat16)
    qweight, scales = rawint4_quantize(weight_bf16)
    packed_int32 = pack_rawint4_uint8_as_int32(qweight)
    weight_shape = torch.tensor([intermediate_size, hidden_size], dtype=torch.int32)

    normalized = loader_mod.CompressedSafeTensorLoader._normalize_rawint4_weight(
        packed_int32, scales, weight_shape, "test.weight_packed"
    )

    assert normalized.dtype == torch.uint8
    assert normalized.shape == qweight.shape
    assert torch.equal(normalized, qweight)


def test_compressed_loader_accepts_uint8_rawint4_weights():
    load_amx_utils()
    loader_mod = sys.modules["kt_kernel.utils.loader"]

    weight_bf16 = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(torch.bfloat16)
    qweight, scales = rawint4_quantize(weight_bf16)
    weight_shape = torch.tensor([intermediate_size, hidden_size], dtype=torch.int32)

    normalized = loader_mod.CompressedSafeTensorLoader._normalize_rawint4_weight(
        qweight, scales, weight_shape, "test.weight_packed"
    )

    assert normalized.dtype == torch.uint8
    assert normalized.shape == qweight.shape
    assert torch.equal(normalized, qweight)


def test_compressed_loader_ignores_invalid_weight_shape_metadata():
    load_amx_utils()
    loader_mod = sys.modules["kt_kernel.utils.loader"]

    weight_bf16 = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(torch.bfloat16)
    qweight, scales = rawint4_quantize(weight_bf16)
    packed_int32 = pack_rawint4_uint8_as_int32(qweight)
    invalid_shape = torch.tensor([-1752796263, -1707567530], dtype=torch.int32)

    normalized = loader_mod.CompressedSafeTensorLoader._normalize_rawint4_weight(
        packed_int32, scales, invalid_shape, "test.weight_packed"
    )

    assert normalized.dtype == torch.uint8
    assert normalized.shape == qweight.shape
    assert torch.equal(normalized, qweight)


def test_compressed_loader_ignores_odd_weight_shape_metadata():
    load_amx_utils()
    loader_mod = sys.modules["kt_kernel.utils.loader"]

    weight_bf16 = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(torch.bfloat16)
    qweight, scales = rawint4_quantize(weight_bf16)
    packed_int32 = pack_rawint4_uint8_as_int32(qweight)
    invalid_shape = torch.tensor([241597647, 1216029047], dtype=torch.int32)

    normalized = loader_mod.CompressedSafeTensorLoader._normalize_rawint4_weight(
        packed_int32, scales, invalid_shape, "test.weight_packed"
    )

    assert normalized.dtype == torch.uint8
    assert normalized.shape == qweight.shape
    assert torch.equal(normalized, qweight)


def test_rawint4_backend_selection_falls_back_to_avx2_for_large_group_size(monkeypatch):
    amx_utils = load_amx_utils()
    fake_amx_backend = object()
    fake_avx2_backend = object()
    fake_avxvnni_backend = object()

    monkeypatch.setattr(amx_utils, "AMXInt4_KGroup_MOE", fake_amx_backend)
    monkeypatch.setattr(amx_utils, "AVX2RawInt4_MOE", fake_avx2_backend)
    monkeypatch.setattr(amx_utils, "AVXVNNI256RawInt4_MOE", fake_avxvnni_backend)
    monkeypatch.setattr(amx_utils, "_HAS_RAWINT4_SUPPORT", False)
    monkeypatch.setattr(amx_utils, "_HAS_AVX2_RAWINT4_SUPPORT", True)
    monkeypatch.setattr(amx_utils, "_HAS_AVXVNNI256_RAW_INT4_SUPPORT", True)
    monkeypatch.setattr(amx_utils, "_HOST_HAS_AVX_VNNI", True)
    monkeypatch.delenv("KT_RAWINT4_BACKEND", raising=False)

    assert amx_utils._select_rawint4_backend(512) is fake_avx2_backend
    assert amx_utils._select_rawint4_backend(128) is fake_avxvnni_backend


def test_rawint4_backend_selection_accepts_forced_blocked_amx(monkeypatch):
    amx_utils = load_amx_utils()
    fake_blocked_backend = object()

    monkeypatch.setattr(amx_utils, "AMXInt4_KGroupBlocked_MOE", fake_blocked_backend)
    monkeypatch.setattr(amx_utils, "_HAS_RAWINT4_BLOCKED_SUPPORT", True)
    monkeypatch.setenv("KT_RAWINT4_BACKEND", "amx_blocked")

    assert amx_utils._select_rawint4_backend(32) is fake_blocked_backend


def test_rawint4_backend_selection_rejects_forced_avxvnni_with_large_group_size(monkeypatch):
    amx_utils = load_amx_utils()

    monkeypatch.setattr(amx_utils, "_HAS_AVXVNNI256_RAW_INT4_SUPPORT", True)
    monkeypatch.setattr(amx_utils, "_HOST_HAS_AVX_VNNI", True)
    monkeypatch.setenv("KT_RAWINT4_BACKEND", "avxvnni")

    with pytest.raises(RuntimeError, match="group_size=512 is unsupported"):
        amx_utils._select_rawint4_backend(512)


if __name__ == "__main__":
    print("=" * 60)
    print("RAWINT4 MoE Accuracy Test")
    print("=" * 60)
    test_rawint4_accuracy()
    print("PASSED")
