#!/usr/bin/env python
# coding=utf-8
"""
Bench MOE kernel with runtime tiling params (N_BLOCK_UP_GATE, N_BLOCK_DOWN, N_BLOCK, M_BLOCK, K_BLOCK)
- Demonstrates how to get/set tiling params from Python via kt_kernel_ext.moe.tiling
- Runs a small benchmark similar to bench_moe_kernel.py

Usage examples:
  # 1) Just run with defaults (int8)
  python bench_moe_kernel_tiling.py --quant int8

  # 2) Override tiling params for INT8
  python bench_moe_kernel_tiling.py --quant int8 \
    --n_block_up_gate 32 --n_block_down 64 --n_block 64 --m_block 320 --k_block 7168

  # 3) Set both INT8 and INT4 tiling params (if INT4 kernel is available on your platform)
  python bench_moe_kernel_tiling.py --quant int4 --set_all \
    --n_block_up_gate 256 --n_block_down 1024 --n_block 64 --m_block 320 --k_block 7168
"""
import os
import sys
import time
import argparse

os.environ.setdefault("BLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import torch  # noqa: E402
import kt_kernel_ext as ce  # noqa: E402
from tqdm import tqdm  # noqa: E402


def maybe_get_class(module, name):
    return getattr(module, name) if hasattr(module, name) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=["int8", "int4"], default="int8")
    parser.add_argument("--expert_num", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=7168)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--num_experts_per_tok", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=25600)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--qlen", type=int, default=1024)
    parser.add_argument("--warm_up_iter", type=int, default=200)
    parser.add_argument("--test_iter", type=int, default=500)
    parser.add_argument("--threads", type=int, default=160, help="CPUInfer initialization param")

    # Tiling params
    parser.add_argument("--set_all", action="store_true", help="Apply tiling to both INT8 and INT4 kernels")
    parser.add_argument("--n_block_up_gate", type=int, default=None)
    parser.add_argument("--n_block_down", type=int, default=None)
    parser.add_argument("--n_block", type=int, default=None)
    parser.add_argument("--m_block", type=int, default=None)
    parser.add_argument("--k_block", type=int, default=None)
    parser.add_argument("--n_block_up_gate_prefi", type=int, default=None)
    parser.add_argument("--n_block_down_prefi", type=int, default=None)

    args = parser.parse_args()

    # Show current tiling defaults
    if args.quant == "int8":
        print("[tiling] default int8:", ce.moe.tiling.get_int8())
    if hasattr(ce.moe.tiling, "get_int4") and args.quant == "int4":
        print("[tiling] default int4:", ce.moe.tiling.get_int4())

    # Apply overrides if provided
    if any(v is not None for v in [args.n_block_up_gate, args.n_block_down, args.n_block, args.m_block, args.k_block]):
        # Fill missing values with current defaults to avoid overwriting unrelated params
        def fill_defaults(getter):
            cur = getter()
            return (
                args.n_block_up_gate if args.n_block_up_gate is not None else int(cur["n_block_up_gate"]),
                args.n_block_down if args.n_block_down is not None else int(cur["n_block_down"]),
                args.n_block if args.n_block is not None else int(cur["n_block"]),
                args.m_block if args.m_block is not None else int(cur["m_block"]),
                args.k_block if args.k_block is not None else int(cur["k_block"]),
                (
                    args.n_block_up_gate_prefi
                    if args.n_block_up_gate_prefi is not None
                    else int(cur["n_block_up_gate_prefi"])
                ),
                args.n_block_down_prefi if args.n_block_down_prefi is not None else int(cur["n_block_down_prefi"]),
            )

        if args.set_all and hasattr(ce.moe.tiling, "set_all"):
            nbug, nbd, nb, mb, kb, nbug_prefi, nbd_prefi = fill_defaults(ce.moe.tiling.get_int8)
            ce.moe.tiling.set_all(nbug, nbd, nb, mb, kb, nbug_prefi, nbd_prefi)
            print("[tiling] set_all ->", ce.moe.tiling.get_int8())
            if hasattr(ce.moe.tiling, "get_int4"):
                print("[tiling] set_all -> int4:", ce.moe.tiling.get_int4())
        else:
            if args.quant == "int8":
                nbug, nbd, nb, mb, kb, nbug_prefi, nbd_prefi = fill_defaults(ce.moe.tiling.get_int8)
                ce.moe.tiling.set_int8(nbug, nbd, nb, mb, kb, nbug_prefi, nbd_prefi)
                print("[tiling] set_int8 ->", ce.moe.tiling.get_int8())
            elif args.quant == "int4" and hasattr(ce.moe.tiling, "set_int4"):
                nbug, nbd, nb, mb, kb, nbug_prefi, nbd_prefi = fill_defaults(ce.moe.tiling.get_int4)
                ce.moe.tiling.set_int4(nbug, nbd, nb, mb, kb, nbug_prefi, nbd_prefi)
                print("[tiling] set_int4 ->", ce.moe.tiling.get_int4())

    # Warn about divisibility expectations; kernels assume specific blocking
    # - Some helpers assert n % N_BLOCK == 0, etc. Ensure your dims/tiles align.
    print("[note] Ensure your selected tiling parameters are compatible with hidden/intermediate sizes and blocking.")

    # Initialize CPUInfer
    CPUInfer = ce.CPUInfer(args.threads)

    # Select MOE kernel
    moe_cls = None
    if args.quant == "int8":
        moe_cls = maybe_get_class(ce.moe, "Int8_KERNEL_MOE")
        if moe_cls is None:
            raise RuntimeError("Int8 kernel binding 'Int8_KERNEL_MOE' not found.")
        bytes_per_elem = 1.0
    else:
        moe_cls = maybe_get_class(ce.moe, "Int4_KERNEL_MOE")
        if moe_cls is None:
            raise RuntimeError("Int4 kernel binding 'Int4_KERNEL_MOE' not available on this platform.")
        bytes_per_elem = 0.5

    # Prepare config/weights
    expert_num = args.expert_num
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts_per_tok = args.num_experts_per_tok
    layer_num = args.layer_num
    max_len = args.max_len

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()

    moes = []
    gate_projs, up_projs, down_projs = [], [], []

    for layer_idx in range(layer_num):
        gate_proj = torch.randn(
            (expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cpu"
        ).contiguous()
        up_proj = torch.randn(
            (expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cpu"
        ).contiguous()
        down_proj = torch.randn(
            (expert_num, hidden_size, intermediate_size), dtype=torch.float32, device="cpu"
        ).contiguous()

        cfg = ce.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        cfg.max_len = max_len
        cfg.gate_proj = gate_proj.data_ptr()
        cfg.up_proj = up_proj.data_ptr()
        cfg.down_proj = down_proj.data_ptr()
        cfg.pool = CPUInfer.backend_

        moe = moe_cls(cfg)
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        gate_projs.append(gate_proj)
        up_projs.append(up_proj)
        down_projs.append(down_proj)
        moes.append(moe)

    qlen = args.qlen
    warm_up_iter = args.warm_up_iter
    test_iter = args.test_iter

    expert_ids = (
        torch.rand(test_iter * qlen, expert_num)
        .argsort(dim=-1)[:, :num_experts_per_tok]
        .reshape(test_iter, qlen * num_experts_per_tok)
        .to("cpu")
        .contiguous()
    )
    weights = torch.rand((test_iter, qlen, num_experts_per_tok), dtype=torch.float32).to("cpu").contiguous()
    input_tensor = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16).to("cpu").contiguous()
    output_tensor = torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16).to("cpu").contiguous()
    bsz_tensor = torch.tensor([qlen], dtype=torch.int32).to("cpu").contiguous()

    # Warmup
    for i in tqdm(range(warm_up_iter), desc="Warm-up"):
        CPUInfer.submit(
            moes[i % layer_num].forward_task(
                bsz_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids[i].data_ptr(),
                weights[i].data_ptr(),
                input_tensor[i % layer_num].data_ptr(),
                output_tensor[i % layer_num].data_ptr(),
            )
        )
        CPUInfer.sync()

    # Measure
    start = time.perf_counter()
    for i in tqdm(range(test_iter), desc="Testing"):
        CPUInfer.submit(
            moes[i % layer_num].forward_task(
                bsz_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids[i].data_ptr(),
                weights[i].data_ptr(),
                input_tensor[i % layer_num].data_ptr(),
                output_tensor[i % layer_num].data_ptr(),
                False,
            )
        )
        CPUInfer.sync()
    end = time.perf_counter()

    total_time = end - start
    time_per_iter_us = total_time / test_iter * 1e6
    bandwidth_gbs = (
        hidden_size * intermediate_size * 3 * num_experts_per_tok * qlen * bytes_per_elem * test_iter / total_time / 1e9
    )
    flops_tflops = hidden_size * intermediate_size * qlen * 3 * num_experts_per_tok * 2 * test_iter / total_time / 1e12

    print("\n=== Results ===")
    print("quant:", args.quant)
    if hasattr(ce.moe.tiling, "get_int8") and args.quant == "int8":
        print("tiling int8:", ce.moe.tiling.get_int8())
    if hasattr(ce.moe.tiling, "get_int4") and args.quant == "int4":
        print("tiling int4:", ce.moe.tiling.get_int4())
    print("time (s):", total_time)
    print("iter:", test_iter)
    print("time per iter (us):", time_per_iter_us)
    print("bandwidth (GB/s):", bandwidth_gbs)
    print("TFLOPS:", flops_tflops)


if __name__ == "__main__":
    main()
