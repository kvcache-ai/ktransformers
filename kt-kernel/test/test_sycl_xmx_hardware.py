"""Opt-in integration checks: source oneAPI and set KT_TEST_SYCL_XMX=1."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("KT_TEST_SYCL_XMX") != "1", reason="requires Intel XMX GPU and SYCL extension"
)


@pytest.fixture
def setup(monkeypatch):
    import torch
    from kt_kernel import kt_kernel_ext as ext

    monkeypatch.setenv("KT_SYCL_XMX_MODE", "int8")
    cpu = ext.CPUInfer(2)
    mask = torch.zeros(4, dtype=torch.bool)
    config = ext.moe.MOEConfig(4, 2, 128, 128, mask.data_ptr())
    config.pool = cpu.backend_
    config.max_len = 17
    config.quant_config.bits = 4
    config.quant_config.group_size = 128
    config.quant_config.zero_point = False
    qw = torch.full((4, 16, 128), -2004318072, dtype=torch.int32)  # all GPTQ nibbles are 8, hence zero
    scales = torch.ones(4, 1, 128)
    for name in ("gate", "up", "down"):
        setattr(config, name + "_projs", [[t.data_ptr() for t in qw]])
        setattr(config, name + "_scales", [[t.data_ptr() for t in scales]])
    return ext, torch, cpu, config, mask, qw, scales


@pytest.mark.parametrize("mode", ["fp16", "int8"])
@pytest.mark.parametrize("contiguous", [False, True])
def test_zero_weights_and_incremental_output(setup, monkeypatch, mode, contiguous):
    ext, torch, cpu, config, mask, qw, scales = setup
    monkeypatch.setenv("KT_SYCL_XMX_MODE", mode)
    if contiguous:
        for name in ("gate", "up", "down"):
            setattr(config, name + "_projs", [])
            setattr(config, name + "_scales", [])
            setattr(config, name + "_proj", qw.data_ptr())
            setattr(config, name + "_scale", scales.data_ptr())
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    cpu.submit(moe.load_weights_task())
    cpu.sync()
    for rows in (1, 17):
        qlen = torch.tensor([rows], dtype=torch.int32)
        x = torch.zeros(rows, 128, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1]] * rows, dtype=torch.int64)
        weights = torch.full((rows, 2), 0.5)
        out = torch.ones_like(x)
        cpu.submit(
            moe.forward_task(
                qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False
            )
        )
        cpu.sync()
        assert not out.count_nonzero()
        out.fill_(1)
        cpu.submit(
            moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), True)
        )
        cpu.sync()
        assert bool((out == 1).all())


def test_invalid_quantization_is_rejected(setup):
    ext, _, _, config, *_ = setup
    config.quant_config.zero_point = True
    with pytest.raises(RuntimeError, match="symmetric GPTQ INT4"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_invalid_mode_is_rejected(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_MODE", "typo")
    with pytest.raises(RuntimeError, match="KT_SYCL_XMX_MODE"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_invalid_mapping_is_rejected(setup):
    ext, torch, _, config, *_ = setup
    mapping = torch.tensor([0, 1, 2, 1 << 40], dtype=torch.int64)
    config.physical_to_logical_map = mapping.data_ptr()
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    with pytest.raises(RuntimeError, match="physical_to_logical_map"):
        moe.load_weights()


@pytest.mark.parametrize("numa", ["0", "1"])
def test_workspace_bounds_are_rejected(setup, monkeypatch, numa):
    ext, torch, cpu, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_NUMA_DISPATCH", numa)
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    moe.load_weights()
    x = torch.zeros(18, 128, dtype=torch.bfloat16)
    ids = torch.zeros(18, 2, dtype=torch.int64)
    weights = torch.ones(18, 2)
    qlen = torch.tensor([18], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="workspace"):
        moe.forward(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), x.data_ptr(), False)
    cpu.submit(
        moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), x.data_ptr(), False)
    )
    with pytest.raises(RuntimeError, match="workspace"):
        cpu.sync()
    # The queue remains usable after reporting an invalid request.
    qlen[0] = 1
    cpu.submit(
        moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), x.data_ptr(), False)
    )
    cpu.sync()


@pytest.mark.parametrize(
    "name,value",
    [
        ("KT_SYCL_XMX_ROWS", "16"),
        ("KT_SYCL_XMX_GRAPH", "yes"),
        ("KT_SYCL_XMX_HOST_QUANT", "auto"),
        ("KT_SYCL_XMX_FUSED_QUANT", "yes"),
        ("KT_SYCL_XMX_NUMA_DISPATCH", "auto"),
    ],
)
def test_invalid_tuning_is_rejected(setup, monkeypatch, name, value):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize("fused", ["0", "1", "prefill"])
@pytest.mark.parametrize("decode_gate", ["xmx", "vector", "dp4a", "xmx-split"])
def test_graph_reads_changing_inputs_and_routes(setup, monkeypatch, fused, decode_gate):
    ext, torch, cpu, config, mask, qw, scales = setup
    generator = torch.Generator().manual_seed(20261014)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", fused)
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_GATE_KERNEL", decode_gate)
    if fused == "1" and decode_gate != "xmx":
        pytest.skip("Full quantization fusion requires XMX gate")
    monkeypatch.setenv("KT_SYCL_XMX_GRAPH", "0")
    direct = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv("KT_SYCL_XMX_GRAPH", "1")
    graph = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    for moe in (direct, graph):
        cpu.submit(moe.load_weights_task())
        cpu.sync()
    qlen = torch.tensor([1], dtype=torch.int32)
    for iteration in range(12):
        topk = iteration % 2 + 1
        x = torch.randn(1, 128, generator=generator).bfloat16()
        ids = torch.randperm(4, generator=generator)[:topk].reshape(1, topk)
        if iteration % 3 == 0:
            ids[0, 0] = -1
        weights = torch.softmax(torch.randn(1, topk, generator=generator), 1)
        outputs = []
        for moe in (direct, graph):
            out = torch.empty_like(x)
            cpu.submit(
                moe.forward_task(
                    qlen.data_ptr(), topk, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False
                )
            )
            cpu.sync()
            outputs.append(out)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 17])
@pytest.mark.parametrize("fused_mode", ["1", "prefill"])
def test_fused_quantization_matches_unfused(setup, monkeypatch, rows, fused_mode):
    ext, torch, cpu, config, mask, qw, scales = setup
    generator = torch.Generator().manual_seed(42)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", "0")
    direct = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", fused_mode)
    fused = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    qlen = torch.tensor([rows], dtype=torch.int32)
    x = torch.randn(rows, 128, generator=generator).bfloat16()
    ids = torch.tensor([[0, 1]] * rows, dtype=torch.int64)
    weights = torch.full((rows, 2), 0.5)
    outputs = []
    for moe in (direct, fused):
        cpu.submit(moe.load_weights_task())
        cpu.sync()
        out = torch.empty_like(x)
        cpu.submit(
            moe.forward_task(
                qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False
            )
        )
        cpu.sync()
        outputs.append(out)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "gate,down",
    [
        ("vector", "vector"),
        ("dp4a", "dp4a"),
        ("xmx", "dp4a"),
        ("dp4a", "xmx"),
        ("vector", "xmx"),
        ("xmx", "vector"),
        ("xmx-split", "xmx"),
        ("xmx", "xmx-split"),
    ],
)
@pytest.mark.parametrize("rows", [1, 7])
def test_projection_candidates_against_reference(setup, monkeypatch, gate, down, rows):
    ext, torch, cpu, config, mask, qw, scales = setup
    generator = torch.Generator().manual_seed(73)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    monkeypatch.setenv("KT_SYCL_XMX_MODE", "int8")
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", "0")
    monkeypatch.setenv("KT_SYCL_XMX_GATE_KERNEL", gate)
    monkeypatch.setenv("KT_SYCL_XMX_DOWN_KERNEL", down)
    monkeypatch.delenv("KT_SYCL_XMX_DECODE_GATE_KERNEL", raising=False)
    monkeypatch.delenv("KT_SYCL_XMX_DECODE_DOWN_KERNEL", raising=False)
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    cpu.submit(moe.load_weights_task())
    cpu.sync()
    x = torch.randn(rows, 128, generator=generator).bfloat16()
    ids = torch.tensor([[0, 1]] * rows, dtype=torch.int64)
    weights = torch.full((rows, 2), 0.5)
    out = torch.empty_like(x)
    qlen = torch.tensor([rows], dtype=torch.int32)
    cpu.submit(
        moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False)
    )
    cpu.sync()
    shift = torch.arange(8) * 4
    deq = (((qw.long()[..., None] >> shift) & 15) - 8).permute(0, 1, 3, 2).reshape(4, 128, 128).float() * 0.003
    expected = torch.zeros(rows, 128)
    for e in (0, 1):
        projection = x.float() @ deq[e]
        middle = (torch.nn.functional.silu(projection) * projection).bfloat16().float()
        expected += (middle @ deq[e]).bfloat16().float() * 0.5
    expected = expected.bfloat16().float()
    rmse = (out.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)
    assert bool(torch.isfinite(out).all())
    assert rmse < 0.04


@pytest.mark.parametrize(
    "name,value",
    [
        ("KT_SYCL_XMX_GATE_KERNEL", "typo"),
        ("KT_SYCL_XMX_DOWN_KERNEL", "auto"),
        ("KT_SYCL_XMX_DECODE_GATE_KERNEL", "bad"),
        ("KT_SYCL_XMX_VECTOR_SPLITS", "3"),
        ("KT_SYCL_XMX_GATE_SPLITS", "0"),
        ("KT_SYCL_XMX_DOWN_SPLITS", "16"),
    ],
)
def test_invalid_projection_selection_is_rejected(setup, monkeypatch, name, value):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_prefill_fusion_allows_vector_decode_but_rejects_full_fusion(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_GATE_KERNEL", "vector")
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", "prefill")
    ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", "1")
    with pytest.raises(RuntimeError, match="fused quantization"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize(
    "decode_gate,decode_down",
    [
        ("vector", "xmx"),
        ("xmx", "vector"),
        ("dp4a", "xmx"),
        ("xmx", "dp4a"),
        ("xmx-split", "xmx"),
        ("xmx", "xmx-split"),
    ],
)
@pytest.mark.parametrize("host_quant", ["0", "1"])
def test_decode_overrides_preserve_prefill_and_capture_settings(
    setup, monkeypatch, decode_gate, decode_down, host_quant
):
    """Decode-only choices must not leak into fused prefill or later executors."""
    ext, torch, cpu, config, mask, qw, scales = setup
    generator = torch.Generator().manual_seed(20261014)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    monkeypatch.setenv("KT_SYCL_XMX_GRAPH", "0")
    monkeypatch.setenv("KT_SYCL_XMX_HOST_QUANT", host_quant)
    monkeypatch.setenv("KT_SYCL_XMX_GATE_SPLITS", "2")
    monkeypatch.setenv("KT_SYCL_XMX_DOWN_SPLITS", "2")
    monkeypatch.setenv("KT_SYCL_XMX_GATE_KERNEL", "xmx")
    monkeypatch.setenv("KT_SYCL_XMX_DOWN_KERNEL", "xmx")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_GATE_KERNEL", decode_gate)
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_DOWN_KERNEL", decode_down)
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", "prefill")
    selected = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    cpu.submit(selected.load_weights_task())
    cpu.sync()

    monkeypatch.setenv("KT_SYCL_XMX_DECODE_GATE_KERNEL", "inherit")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_DOWN_KERNEL", "inherit")
    for rows in (1, 17):
        # The explicit reference matches this stage. It also changes the
        # environment after selected captured its own immutable choices.
        monkeypatch.setenv("KT_SYCL_XMX_GATE_KERNEL", decode_gate if rows == 1 else "xmx")
        monkeypatch.setenv("KT_SYCL_XMX_DOWN_KERNEL", decode_down if rows == 1 else "xmx")
        monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", "0" if rows == 1 else "prefill")
        explicit = ext.moe.SYCLXMXGPTQInt4_MOE(config)
        cpu.submit(explicit.load_weights_task())
        cpu.sync()
        x = torch.randn(rows, 128, generator=generator).bfloat16()
        ids = torch.randint(0, 4, (rows, 2), generator=generator)
        weights = torch.softmax(torch.randn(rows, 2, generator=generator), 1)
        qlen = torch.tensor([rows], dtype=torch.int32)
        outputs = []
        for moe in (selected, explicit):
            out = torch.empty_like(x)
            cpu.submit(
                moe.forward_task(
                    qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False
                )
            )
            cpu.sync()
            outputs.append(out)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("mode,quant_fusion", [("fp16", "0"), ("int8", "0"), ("int8", "1"), ("int8", "prefill")])
@pytest.mark.parametrize("graph", ["0", "1"])
@pytest.mark.parametrize("topk", [1, 2, 8])
def test_fused_decode_merge_matches_separate_with_dynamic_routes(setup, monkeypatch, mode, quant_fusion, graph, topk):
    ext, torch, cpu, config, mask, qw, scales = setup
    config.num_experts_per_tok = topk
    generator = torch.Generator().manual_seed(108)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    monkeypatch.setenv("KT_SYCL_XMX_MODE", mode)
    monkeypatch.setenv("KT_SYCL_XMX_GRAPH", graph)
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_DOWN_KERNEL", "xmx")
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", quant_fusion)
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_MERGE", "separate")
    separate = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_MERGE", "fused")
    fused = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    for moe in (separate, fused):
        cpu.submit(moe.load_weights_task())
        cpu.sync()
    for ids_list in ([0, 1], [1, 1], [-1, 1], [3, 0], [-1, -1], [99, 2]):
        for rows in (1, 17):
            x = torch.randn(rows, 128, generator=generator).bfloat16()
            expanded = (ids_list * ((topk + 1) // 2))[:topk]
            ids = torch.tensor([expanded] * rows, dtype=torch.int64)
            weights = torch.softmax(torch.randn(rows, topk, generator=generator), 1)
            qlen = torch.tensor([rows], dtype=torch.int32)
            outputs = []
            for moe in (separate, fused):
                out = torch.randn(rows, 128, generator=generator).bfloat16()
                cpu.submit(
                    moe.forward_task(
                        qlen.data_ptr(), topk, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False
                    )
                )
                cpu.sync()
                outputs.append(out.clone())
                previous = out.clone()
                cpu.submit(
                    moe.forward_task(
                        qlen.data_ptr(), topk, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), True
                    )
                )
                cpu.sync()
                torch.testing.assert_close(out, (previous.float() * 2).bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,value",
    [("KT_SYCL_XMX_TRACE", "yes"), ("KT_SYCL_XMX_DECODE_MERGE", "auto"), ("KT_SYCL_XMX_TRACE_EVERY", "-1")],
)
def test_invalid_pipeline_options_are_rejected(setup, monkeypatch, name, value):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_fused_merge_rejects_unsupported_down_kernel(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_MERGE", "fused")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_DOWN_KERNEL", "vector")
    with pytest.raises(RuntimeError, match="DECODE_MERGE"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_host_profile_snapshot_reset_and_zero_routes(setup, monkeypatch):
    ext, torch, cpu, config, mask, qw, scales = setup
    monkeypatch.setenv("KT_SYCL_XMX_TRACE", "host")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_MERGE", "separate")
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    cpu.submit(moe.load_weights_task())
    cpu.sync()
    for row_count, expert in ((1, 0), (17, -1)):
        rows = torch.tensor([row_count], dtype=torch.int32)
        ids = torch.full((row_count, 2), expert, dtype=torch.int64)
        weights = torch.full((row_count, 2), 0.5)
        x = torch.ones(row_count, 128, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        cpu.submit(
            moe.forward_task(
                rows.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False
            )
        )
        cpu.sync()
    stats = moe.profile_snapshot()
    assert stats["decode.calls"] == stats["prefill.calls"] == 1
    assert stats["decode.routed"] == 2 and stats["prefill.routed"] == 0
    assert stats["decode.host_executor_us"] > 0
    assert stats["decode.host_wrapper_us"] >= stats["decode.host_bf16_us"]
    assert not any("device_" in key for key in stats)
    moe.reset_profile()
    assert moe.profile_snapshot() == {}


def test_fused_merge_rejects_topk_above_eight(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    config.num_experts_per_tok = 9
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_MERGE", "fused")
    with pytest.raises(RuntimeError, match="topk<=8"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_device_trace_requires_profiled_queue(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_TRACE", "device")
    # The queue property is fixed by this process's first creation.
    if os.getenv("KT_SYCL_PROFILE") == "1":
        pytest.skip("Queue profiling was explicitly enabled for this test process")
    with pytest.raises(RuntimeError, match="requires KT_SYCL_PROFILE=1"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize("splits", ["1", "2"])
@pytest.mark.parametrize("graph", ["0", "1"])
@pytest.mark.parametrize("host_quant", ["0", "1"])
def test_fused_expert_decode_matches_staged_and_preserves_prefill(setup, monkeypatch, splits, graph, host_quant):
    from pathlib import Path
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "projection_reference", Path(__file__).parents[1] / "bench/bench_sycl_xmx_moe.py"
    )
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    ext, torch, cpu, _, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_HOST_QUANT", host_quant)
    generator = torch.Generator().manual_seed(47)
    mask = torch.tensor([False, False, False, True])
    config = ext.moe.MOEConfig(4, 4, 2048, 512, mask.data_ptr())
    config.pool = cpu.backend_
    config.max_len = 17
    config.quant_config.bits = 4
    config.quant_config.group_size = 128
    config.quant_config.zero_point = False
    config.swiglu_limit = 1.5
    config.swiglu_alpha = 1.702
    proj = [
        reference_module.packed_weights(4, 512, 2048, 128, generator),
        reference_module.packed_weights(4, 512, 2048, 128, generator),
        reference_module.packed_weights(4, 2048, 512, 128, generator),
    ]
    for stage, (qw, sc) in zip(("gate", "up", "down"), proj):
        setattr(config, stage + "_projs", [[t.data_ptr() for t in qw]])
        setattr(config, stage + "_scales", [[t.data_ptr() for t in sc]])
    mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    monkeypatch.setenv("KT_SYCL_XMX_MODE", "int8")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_GATE_KERNEL", "xmx-split")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_DOWN_KERNEL", "xmx")
    monkeypatch.setenv("KT_SYCL_XMX_GATE_SPLITS", splits)
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", "prefill")
    monkeypatch.setenv("KT_SYCL_XMX_GRAPH", graph)
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_MERGE", "separate")
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_EXPERT", "0")
    staged = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_EXPERT", "1")
    fused = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    for moe in (staged, fused):
        cpu.submit(moe.load_weights_task(mapping.data_ptr()))
        cpu.sync()
    for rows in (1, 17):
        for route in ([0, 1, 2, -1], [1, 1, 0, 3], [-1, 99, 2, -1], [-1, -1, -1, -1]):
            x = torch.randn(rows, 2048, generator=generator).bfloat16()
            ids = torch.tensor([route] * rows, dtype=torch.int64)
            weights = torch.softmax(torch.randn(rows, 4, generator=generator), 1)
            qlen = torch.tensor([rows], dtype=torch.int32)
            outputs = []
            for moe in (staged, fused):
                out = torch.empty_like(x)
                cpu.submit(
                    moe.forward_task(
                        qlen.data_ptr(), 4, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False
                    )
                )
                cpu.sync()
                outputs.append(out.clone())
                previous = out.clone()
                cpu.submit(
                    moe.forward_task(
                        qlen.data_ptr(), 4, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), True
                    )
                )
                cpu.sync()
                torch.testing.assert_close(out, (previous.float() * 2).bfloat16(), rtol=0, atol=0)
            # Same group boundaries/scales and BF16 rounding; allow rare
            # quantization ties from ESIMD vs scalar SIMD division.
            denominator = outputs[0].float().square().mean().sqrt().clamp_min(1e-12)
            rmse = (outputs[0].float() - outputs[1].float()).square().mean().sqrt() / denominator
            assert rmse < 0.005


def test_fused_expert_rejects_unsupported_dimensions(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_EXPERT", "1")
    with pytest.raises(RuntimeError, match="FUSED_EXPERT"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize(
    "setting,value",
    [("KT_SYCL_XMX_MODE", "fp16"), ("KT_SYCL_XMX_GATE_SPLITS", "4"), ("KT_SYCL_XMX_DECODE_MERGE", "fused")],
)
def test_fused_expert_rejects_incompatible_options(setup, monkeypatch, setting, value):
    ext, torch, cpu, _, *_ = setup
    mask = torch.zeros(4, dtype=torch.bool)
    config = ext.moe.MOEConfig(4, 2, 2048, 512, mask.data_ptr())
    config.pool = cpu.backend_
    config.max_len = 1
    config.quant_config.bits = 4
    config.quant_config.group_size = 128
    config.quant_config.zero_point = False
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_EXPERT", "1")
    monkeypatch.setenv("KT_SYCL_XMX_GATE_SPLITS", "2")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_GATE_KERNEL", "xmx-split")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_DOWN_KERNEL", "xmx")
    monkeypatch.setenv("KT_SYCL_XMX_DECODE_MERGE", "separate")
    monkeypatch.setenv(setting, value)
    with pytest.raises(RuntimeError, match="FUSED_EXPERT"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_xmx_implementation_revision_is_reported(setup):
    ext, _, _, _, *_ = setup
    assert ext.moe.SYCLXMXGPTQInt4_MOE.implementation_revision() >= 6


@pytest.mark.parametrize(
    "prefill_rows,columns,down_rows,load",
    [
        ("16", "16", "16", "block"),
        ("32", "16", "32", "block"),
        ("8", "32", "8", "block"),
        ("16", "32", "16", "block"),
        ("8", "32", "16", "block"),
        ("8", "32", "16", "gather"),
        ("16", "32", "16", "gather"),
    ],
)
@pytest.mark.parametrize("fused", ["0", "prefill"])
@pytest.mark.parametrize("merge", ["scalar", "vector"])
@pytest.mark.parametrize("incremental", [False, True])
def test_wide_prefill_matches_original_with_tails(
    setup, monkeypatch, prefill_rows, columns, down_rows, load, fused, merge, incremental
):
    ext, torch, cpu, config, mask, qw, scales = setup
    generator = torch.Generator().manual_seed(150)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    config.max_len = 65
    monkeypatch.setenv("KT_SYCL_XMX_FUSED_QUANT", fused)
    monkeypatch.setenv("KT_SYCL_XMX_GATE_KERNEL", "xmx")
    monkeypatch.setenv("KT_SYCL_XMX_DOWN_KERNEL", "xmx")
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_ROWS", "8")
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_N", "16")
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_DOWN_ROWS", "8")
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_LOAD", "block")
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_MERGE", "scalar")
    original = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_LOAD", load)
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_MERGE", merge)
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_ROWS", prefill_rows)
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_N", columns)
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_DOWN_ROWS", down_rows)
    wide = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    for moe in (original, wide):
        cpu.submit(moe.load_weights_task())
        cpu.sync()
    for rows in (1, 17, 33, 65):
        x = torch.randn(rows, 128, generator=generator).bfloat16()
        ids = torch.tensor([[1, 1] if r % 3 == 0 else [r % 4, -1] for r in range(rows)], dtype=torch.int64)
        weights = torch.softmax(torch.randn(rows, 2, generator=generator), 1)
        qlen = torch.tensor([rows], dtype=torch.int32)
        initial = torch.randn(rows, 128, generator=generator).bfloat16()
        values = []
        for moe in (original, wide):
            out = initial.clone()
            cpu.submit(
                moe.forward_task(
                    qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), incremental
                )
            )
            cpu.sync()
            values.append(out)
        assert torch.isfinite(values[1]).all()
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("KT_SYCL_XMX_PREFILL_ROWS", "12"),
        ("KT_SYCL_XMX_PREFILL_N", "64"),
        ("KT_SYCL_XMX_PREFILL_DOWN_ROWS", "0"),
        ("KT_SYCL_XMX_PREFILL_MERGE", "invalid"),
    ],
)
def test_invalid_prefill_tiling_is_rejected(setup, monkeypatch, name, value):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize("mode,gate,down", [("fp16", "xmx", "xmx"), ("int8", "vector", "xmx"), ("int8", "xmx", "dp4a")])
def test_n32_rejects_incompatible_prefill_projections(setup, monkeypatch, mode, gate, down):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_MODE", mode)
    monkeypatch.setenv("KT_SYCL_XMX_GATE_KERNEL", gate)
    monkeypatch.setenv("KT_SYCL_XMX_DOWN_KERNEL", down)
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_N", "32")
    with pytest.raises(RuntimeError, match="PREFILL_N"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize("load,columns", [("invalid", "32"), ("gather", "16"), ("packed", "32")])
def test_prefill_load_rejects_invalid_or_unusable_modes(setup, monkeypatch, load, columns):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_LOAD", load)
    monkeypatch.setenv("KT_SYCL_XMX_PREFILL_N", columns)
    with pytest.raises(RuntimeError, match="PREFILL_LOAD"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_panther_lake_profile_matches_explicit_v6_configuration(monkeypatch):
    """Real executor comparison: a preset must preserve the existing arithmetic."""
    import importlib.util
    from pathlib import Path
    import torch
    from kt_kernel import kt_kernel_ext as ext

    for name in list(os.environ):
        if name.startswith("KT_SYCL_XMX_"):
            monkeypatch.delenv(name)
    spec = importlib.util.spec_from_file_location(
        "profile_reference", Path(__file__).parents[1] / "bench/bench_sycl_xmx_moe.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    generator = torch.Generator().manual_seed(20261016)
    cpu = ext.CPUInfer(2)
    mask = torch.zeros(4, dtype=torch.bool)
    config = ext.moe.MOEConfig(4, 4, 2048, 512, mask.data_ptr())
    config.pool = cpu.backend_
    config.max_len = 33
    config.quant_config.bits = 4
    config.quant_config.group_size = 128
    config.quant_config.zero_point = False
    projections = [
        reference.packed_weights(4, 512, 2048, 128, generator),
        reference.packed_weights(4, 512, 2048, 128, generator),
        reference.packed_weights(4, 2048, 512, 128, generator),
    ]
    for name, (qw, scales) in zip(("gate", "up", "down"), projections):
        setattr(config, name + "_projs", [[t.data_ptr() for t in qw]])
        setattr(config, name + "_scales", [[t.data_ptr() for t in scales]])
    explicit = {
        "KT_SYCL_XMX_MODE": "int8",
        "KT_SYCL_XMX_ROWS": "8",
        "KT_SYCL_XMX_GATE_KERNEL": "xmx",
        "KT_SYCL_XMX_DOWN_KERNEL": "xmx",
        "KT_SYCL_XMX_DECODE_GATE_KERNEL": "xmx-split",
        "KT_SYCL_XMX_DECODE_DOWN_KERNEL": "xmx",
        "KT_SYCL_XMX_GATE_SPLITS": "2",
        "KT_SYCL_XMX_DOWN_SPLITS": "1",
        "KT_SYCL_XMX_PREFILL_ROWS": "8",
        "KT_SYCL_XMX_PREFILL_DOWN_ROWS": "16",
        "KT_SYCL_XMX_PREFILL_N": "32",
        "KT_SYCL_XMX_PREFILL_LOAD": "gather",
        "KT_SYCL_XMX_PREFILL_MERGE": "vector",
        "KT_SYCL_XMX_FUSED_EXPERT": "1",
        "KT_SYCL_XMX_DECODE_MERGE": "separate",
        "KT_SYCL_XMX_FUSED_QUANT": "prefill",
        "KT_SYCL_XMX_NUMA_DISPATCH": "1",
        "KT_SYCL_XMX_GRAPH": "0",
        "KT_SYCL_XMX_HOST_QUANT": "0",
    }
    with monkeypatch.context() as settings:
        for key, value in explicit.items():
            settings.setenv(key, value)
        old = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv("KT_SYCL_XMX_PROFILE", "panther-lake")
    tuned = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    assert not any(key in os.environ for key in explicit)
    for moe in (old, tuned):
        cpu.submit(moe.load_weights_task())
        cpu.sync()
    for rows in (1, 17, 33):
        x = torch.randn(rows, 2048, generator=generator).bfloat16()
        ids = torch.tensor([[0, 1, 1, -1]] * rows, dtype=torch.int64)
        weights = torch.softmax(torch.randn(rows, 4, generator=generator), dim=1)
        qlen = torch.tensor([rows], dtype=torch.int32)
        outputs = []
        for moe in (old, tuned):
            output = torch.empty_like(x)
            cpu.submit(
                moe.forward_task(
                    qlen.data_ptr(), 4, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), output.data_ptr(), False
                )
            )
            cpu.sync()
            outputs.append(output.clone())
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
    assert ext.moe.SYCLXMXGPTQInt4_MOE.implementation_revision() >= 7


def test_panther_lake_profile_rejects_unsupported_shape(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_PROFILE", "panther-lake")
    with pytest.raises(RuntimeError, match="requires H=2048 I=512 group=128"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize("profile", ["", "auto", "typo"])
def test_invalid_tuning_profile_is_rejected(setup, monkeypatch, profile):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv("KT_SYCL_XMX_PROFILE", profile)
    with pytest.raises(RuntimeError, match="KT_SYCL_XMX_PROFILE"):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)
