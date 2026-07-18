#!/usr/bin/env python
# coding=utf-8
"""RAWINT4 MoE load-path equivalence tests for KT-Kernel x86 backends.

Each test compares forward outputs of two backend instances that hold the same
weight bytes loaded through different paths (per-expert pointers vs flat
buffers, permuted vs identity expert maps, int32 vs uint8 packing, one vs two
worker subpools). Unless noted the comparison is bitwise: both instances run
identical kernels on identical bytes, so any difference is a load-path bug.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default")

import torch
import kt_kernel_ext

expert_num = 8
hidden_size = 256
intermediate_size = 512
num_experts_per_tok = 2
max_len = 128
group_size = 32


def has_avx_vnni():
    try:
        with open("/proc/cpuinfo", "r") as f:
            return any(("avx_vnni" in line or "avxvnni" in line) for line in f if line.startswith("flags"))
    except OSError:
        return False


def available_backends():
    backends = []
    if hasattr(kt_kernel_ext.moe, "AVX2RawInt4_MOE"):
        backends.append(("AVX2RawInt4_MOE", kt_kernel_ext.moe.AVX2RawInt4_MOE))
    if hasattr(kt_kernel_ext.moe, "AVXVNNI256RawInt4_MOE") and has_avx_vnni():
        backends.append(("AVXVNNI256RawInt4_MOE", kt_kernel_ext.moe.AVXVNNI256RawInt4_MOE))
    return backends


def avxvnni_backend():
    for name, backend_cls in available_backends():
        if name == "AVXVNNI256RawInt4_MOE":
            return backend_cls
    return None


def avx2_backend():
    for name, backend_cls in available_backends():
        if name == "AVX2RawInt4_MOE":
            return backend_cls
    return None


def make_cpu_infer(subpool_count):
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = subpool_count
    config.subpool_numa_map = [0] * subpool_count
    config.subpool_thread_count = [4] * subpool_count
    return kt_kernel_ext.CPUInfer(config)


def identity_map():
    return torch.tensor(range(expert_num), dtype=torch.int64).contiguous()


def make_rawint4_weights(seed):
    """Random packed RAWINT4 weights and bf16 scales for all three projections."""
    gen = torch.Generator().manual_seed(seed)

    def qweight(n, k):
        return torch.randint(0, 256, (expert_num, n, k // 2), dtype=torch.uint8, generator=gen).contiguous()

    def scales(n, k):
        s = torch.rand((expert_num, n, k // group_size), generator=gen) * 0.02 + 0.001
        return s.to(torch.bfloat16).contiguous()

    return {
        "gate_qw": qweight(intermediate_size, hidden_size),
        "up_qw": qweight(intermediate_size, hidden_size),
        "down_qw": qweight(hidden_size, intermediate_size),
        "gate_sc": scales(intermediate_size, hidden_size),
        "up_sc": scales(intermediate_size, hidden_size),
        "down_sc": scales(hidden_size, intermediate_size),
    }


def base_config(pool_backend):
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    config.max_len = max_len
    config.quant_config.bits = 4
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False
    config.pool = pool_backend
    return config


def build_flat_moe(backend_cls, cpu_infer, w, p2l_map):
    config = base_config(cpu_infer.backend_)
    config.gate_proj = w["gate_qw"].data_ptr()
    config.up_proj = w["up_qw"].data_ptr()
    config.down_proj = w["down_qw"].data_ptr()
    config.gate_scale = w["gate_sc"].data_ptr()
    config.up_scale = w["up_sc"].data_ptr()
    config.down_scale = w["down_sc"].data_ptr()
    moe = backend_cls(config)
    cpu_infer.submit(moe.load_weights_task(p2l_map.data_ptr()))
    cpu_infer.sync()
    return moe


def build_per_expert_moe(backend_cls, cpu_infer, w, p2l_map):
    """Load through per-expert pointers, each expert in its own storage.

    Returns the MoE instance and the per-expert tensors, which the caller must
    keep alive while the instance is used (the AVX2 backend serves weights
    directly from these buffers).
    """
    holders = []
    config = base_config(cpu_infer.backend_)
    for weight_key, scale_key, proj_attr, scale_attr in (
        ("gate_qw", "gate_sc", "gate_projs", "gate_scales"),
        ("up_qw", "up_sc", "up_projs", "up_scales"),
        ("down_qw", "down_sc", "down_projs", "down_scales"),
    ):
        weights = [w[weight_key][e].clone().contiguous() for e in range(expert_num)]
        scales = [w[scale_key][e].clone().contiguous() for e in range(expert_num)]
        holders.extend(weights)
        holders.extend(scales)
        setattr(config, proj_attr, [[t.data_ptr() for t in weights]])
        setattr(config, scale_attr, [[t.data_ptr() for t in scales]])
    moe = backend_cls(config)
    cpu_infer.submit(moe.load_weights_task(p2l_map.data_ptr()))
    cpu_infer.sync()
    return moe, holders


def run_forward(cpu_infer, moe, qlen, seed, expert_ids=None):
    gen = torch.Generator().manual_seed(seed)
    if expert_ids is None:
        expert_ids = torch.stack(
            [torch.randperm(expert_num, generator=gen)[:num_experts_per_tok] for _ in range(qlen)]
        ).contiguous()
    weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32, generator=gen).contiguous()
    input_data = (
        (torch.randn((qlen, hidden_size), dtype=torch.float32, generator=gen) / 100.0).to(torch.bfloat16).contiguous()
    )
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

    assert torch.isfinite(output.float()).all()
    assert output.float().abs().sum() > 0
    return output


def check_per_expert_matches_flat(backend_name, backend_cls, subpool_count):
    w = make_rawint4_weights(seed=1234)
    p2l_map = identity_map()
    cpu_infer = make_cpu_infer(subpool_count)
    moe_flat = build_flat_moe(backend_cls, cpu_infer, w, p2l_map)
    moe_pe, holders = build_per_expert_moe(backend_cls, cpu_infer, w, p2l_map)

    for qlen in (1, 16):
        out_flat = run_forward(cpu_infer, moe_flat, qlen, seed=qlen)
        out_pe = run_forward(cpu_infer, moe_pe, qlen, seed=qlen)
        assert torch.equal(out_pe, out_flat), (
            f"{backend_name}: per-expert load differs from flat load " f"(qlen={qlen}, subpools={subpool_count})"
        )


def test_avxvnni_per_expert_load_matches_flat():
    backend_cls = avxvnni_backend()
    if backend_cls is None:
        print("Skipping: AVXVNNI256RawInt4_MOE not available")
        return
    check_per_expert_matches_flat("AVXVNNI256RawInt4_MOE", backend_cls, subpool_count=1)


def test_avxvnni_per_expert_load_matches_flat_two_subpools():
    backend_cls = avxvnni_backend()
    if backend_cls is None:
        print("Skipping: AVXVNNI256RawInt4_MOE not available")
        return
    check_per_expert_matches_flat("AVXVNNI256RawInt4_MOE", backend_cls, subpool_count=2)


def test_avx2_per_expert_load_matches_flat():
    backend_cls = avx2_backend()
    if backend_cls is None:
        print("Skipping: AVX2RawInt4_MOE not available")
        return
    # The AVX2 per-expert (direct pointer) mode requires a single subpool.
    check_per_expert_matches_flat("AVX2RawInt4_MOE", backend_cls, subpool_count=1)


def test_per_expert_load_respects_physical_to_logical_map():
    backends = available_backends()
    if not backends:
        print("Skipping: no x86 RAWINT4 backend available")
        return

    perm = torch.randperm(expert_num, generator=torch.Generator().manual_seed(7)).contiguous()
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(expert_num, dtype=torch.int64)

    for backend_name, backend_cls in backends:
        w = make_rawint4_weights(seed=99)
        cpu_infer = make_cpu_infer(1)
        moe_id, holders_id = build_per_expert_moe(backend_cls, cpu_infer, w, identity_map())
        moe_perm, holders_perm = build_per_expert_moe(backend_cls, cpu_infer, w, perm)

        for qlen in (1, 16):
            gen = torch.Generator().manual_seed(qlen)
            logical_ids = torch.stack(
                [torch.randperm(expert_num, generator=gen)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            # Physical slot p holds logical expert perm[p], so routing to
            # inverse[logical] must reproduce the identity-mapped run.
            physical_ids = inverse[logical_ids].contiguous()
            out_id = run_forward(cpu_infer, moe_id, qlen, seed=qlen + 1000, expert_ids=logical_ids)
            out_perm = run_forward(cpu_infer, moe_perm, qlen, seed=qlen + 1000, expert_ids=physical_ids)
            assert torch.equal(out_perm, out_id), f"{backend_name}: permuted expert map differs (qlen={qlen})"


def int32_pack_from_uint8(qweight):
    """Repack byte-packed RAWINT4 into compressed-tensors int32 storage.

    Goes through the logical nibble order (low nibble = even k), eight int4
    values per int32 word, lowest bits first.
    """
    nib_lo = (qweight & 0x0F).to(torch.int64)
    nib_hi = (qweight >> 4).to(torch.int64)
    nibbles = torch.stack((nib_lo, nib_hi), dim=-1).reshape(qweight.shape[0], qweight.shape[1], -1)
    grouped = nibbles.reshape(qweight.shape[0], qweight.shape[1], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    words = (grouped << shifts).sum(dim=-1)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return words.contiguous()


def test_int32_pack_quantized_layout_matches_uint8():
    backend_cls = avx2_backend()
    if backend_cls is None:
        print("Skipping: AVX2RawInt4_MOE not available")
        return

    w = make_rawint4_weights(seed=4321)
    w32 = dict(w)
    for key in ("gate_qw", "up_qw", "down_qw"):
        words = int32_pack_from_uint8(w[key])
        # The identity behind the loader's int32 -> uint8 view (PR #2075).
        assert torch.equal(words.view(torch.uint8).reshape(w[key].shape), w[key])
        w32[key] = words

    p2l_map = identity_map()
    cpu_infer = make_cpu_infer(1)
    moe_u8 = build_flat_moe(backend_cls, cpu_infer, w, p2l_map)
    moe_i32 = build_flat_moe(backend_cls, cpu_infer, w32, p2l_map)
    for qlen in (1, 16):
        out_u8 = run_forward(cpu_infer, moe_u8, qlen, seed=qlen)
        out_i32 = run_forward(cpu_infer, moe_i32, qlen, seed=qlen)
        assert torch.equal(out_i32, out_u8), f"int32-packed weights differ from uint8 (qlen={qlen})"


def test_flat_load_two_subpools_matches_single():
    backends = available_backends()
    if not backends:
        print("Skipping: no x86 RAWINT4 backend available")
        return

    for backend_name, backend_cls in backends:
        w = make_rawint4_weights(seed=777)
        p2l_map = identity_map()
        cpu_infer_1 = make_cpu_infer(1)
        moe_1 = build_flat_moe(backend_cls, cpu_infer_1, w, p2l_map)
        cpu_infer_2 = make_cpu_infer(2)
        moe_2 = build_flat_moe(backend_cls, cpu_infer_2, w, p2l_map)

        for qlen in (1, 16):
            out_1 = run_forward(cpu_infer_1, moe_1, qlen, seed=qlen)
            out_2 = run_forward(cpu_infer_2, moe_2, qlen, seed=qlen)
            # The split down projection sums its two partials in a different
            # order than the single-part run, so allow rounding differences.
            diff = torch.mean(torch.abs(out_2.float() - out_1.float())) / (torch.mean(torch.abs(out_1.float())) + 1e-8)
            assert (
                diff.item() < 5e-3
            ), f"{backend_name}: subpool split changed results (qlen={qlen}, diff={diff.item():.6f})"


if __name__ == "__main__":
    print("=" * 60)
    print("RAWINT4 MoE Load Equivalence Test")
    print("=" * 60)
    test_avxvnni_per_expert_load_matches_flat()
    test_avxvnni_per_expert_load_matches_flat_two_subpools()
    test_avx2_per_expert_load_matches_flat()
    test_per_expert_load_respects_physical_to_logical_map()
    test_int32_pack_quantized_layout_matches_uint8()
    test_flat_load_two_subpools_matches_single()
    print("PASSED")
