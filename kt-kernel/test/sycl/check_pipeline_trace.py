#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process device trace regression; source oneAPI before running.

Opt-in standalone check: ONEAPI_DEVICE_SELECTOR=level_zero:gpu python check_pipeline_trace.py
"""


def main():
    import os

    os.environ.update(
        KT_SYCL_XMX_GATE_KERNEL="xmx",
        KT_SYCL_XMX_DOWN_KERNEL="xmx",
        KT_SYCL_XMX_HOST_QUANT="0",
        KT_SYCL_XMX_NUMA_DISPATCH="1",
        KT_SYCL_PROFILE="1",
        KT_SYCL_XMX_TRACE="device",
        KT_SYCL_XMX_TRACE_EVERY="0",
        KT_SYCL_XMX_MODE="int8",
        KT_SYCL_XMX_FUSED_QUANT="0",
        KT_SYCL_XMX_GRAPH="0",
        KT_SYCL_XMX_DECODE_GATE_KERNEL="inherit",
        KT_SYCL_XMX_DECODE_DOWN_KERNEL="xmx",
    )
    import torch
    from kt_kernel import kt_kernel_ext as ext

    cpu = ext.CPUInfer(2)
    mask = torch.zeros(4, dtype=torch.bool)
    config = ext.moe.MOEConfig(4, 2, 128, 128, mask.data_ptr())
    config.pool = cpu.backend_
    config.max_len = 1
    config.quant_config.bits = 4
    config.quant_config.group_size = 128
    config.quant_config.zero_point = False
    qw = torch.full((4, 16, 128), -2004318072, dtype=torch.int32)
    sc = torch.ones(4, 1, 128)
    for stage in ("gate", "up", "down"):
        setattr(config, stage + "_projs", [[t.data_ptr() for t in qw]])
        setattr(config, stage + "_scales", [[t.data_ptr() for t in sc]])
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    w = torch.full((1, 2), 0.5)
    rows = torch.tensor([1], dtype=torch.int32)
    out = torch.empty_like(x)
    for mode, commands in (("separate", 5), ("fused", 4)):
        os.environ["KT_SYCL_XMX_DECODE_MERGE"] = mode
        moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
        cpu.submit(moe.load_weights_task())
        cpu.sync()
        cpu.submit(
            moe.forward_task(rows.data_ptr(), 2, ids.data_ptr(), w.data_ptr(), x.data_ptr(), out.data_ptr(), False)
        )
        cpu.sync()
        stats = moe.profile_snapshot()
        assert stats["decode.calls"] == 1 and stats["decode.device_commands"] == commands, stats
        assert stats["decode.device_span_us"] >= stats["decode.device_kernels_us"]
        if mode == "separate":
            assert stats["decode.device_merge_us"] > 0
        else:
            assert stats["decode.device_merge_us"] == 0
        assert not out.count_nonzero()
        moe.reset_profile()
        assert moe.profile_snapshot() == {}
    os.environ["KT_SYCL_XMX_GRAPH"] = "1"
    try:
        ext.moe.SYCLXMXGPTQInt4_MOE(config)
    except RuntimeError as error:
        assert "recorded graph nodes" in str(error)
    else:
        raise AssertionError("Device trace must reject graph node profiling")
    print("Device trace commands, totals, reset and graph guard passed")


if __name__ == "__main__":
    main()
