# SPDX-License-Identifier: Apache-2.0
"""Measure redistributing excess tokens on the KT native MXFP4 SFT fixture."""

import json
import os
from itertools import permutations
from pathlib import Path
from random import Random
from time import perf_counter

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import kt_kernel
from kt_kernel.sft.dist_utils import _dist_gather_varlen_to_rank0, _dist_scatter_varlen_from_rank0
from kt_kernel.sft.token_redistribution import TokenRedistributionPlan
from per_commit import test_mxfp4_sft_numerical as native


TOKENS = [40, 8, 8, 8]
REPETITIONS = 3
MODES = ("receiver", "local", "balanced")


def relative_l2(actual, expected):
    actual, expected = actual.float(), expected.float()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-12))


def run():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    size = dist.get_world_size()
    assert size == len(TOKENS)
    host_group = dist.new_group(backend="gloo")
    torch.set_num_threads(1)

    nodes = sorted(
        int(p.name[4:])
        for p in Path("/sys/devices/system/node").glob("node[0-9]*")
        if (p / "cpulist").read_text().strip()
    )
    assert len(nodes) >= size
    numa_node = nodes[rank]
    workers = 1

    def make_workers(tp_count):
        assert tp_count == 1
        cfg = kt_kernel.kt_kernel_ext.WorkerPoolConfig()
        cfg.subpool_count = 1
        cfg.subpool_numa_map = [numa_node]
        cfg.subpool_thread_count = [workers]
        return kt_kernel.kt_kernel_ext.CPUInfer(cfg)

    native._make_cpu_infer = make_workers
    native.MAX_QLEN = sum(TOKENS)
    native.QLEN = TOKENS[rank]
    base, lora, experts = native._make_weights()
    kernel, keepalive = native._make_backend(base, lora, tp_count=1)
    if rank == 0:
        workers = size
        owner_kernel, owner_keepalive = native._make_backend(base, lora, tp_count=1)
        owner_grads = native._make_grad_buffers()
    grad_store = native._make_grad_buffers()
    batch = native._make_batch((rank, (rank + 1) % native.EXPERTS), 5489 + rank)
    inputs, ids, scores, dy = (
        x.to(device) for x in (batch.inputs, batch.expert_ids, batch.route_weights, batch.grad_output)
    )
    plan = TokenRedistributionPlan.from_counts(TOKENS)
    transport = dict(all_qlens=TOKENS, rank=rank, world_size=size)
    score_template = scores.to(torch.bfloat16)
    phases = {}

    def timed(name, fn, *args, **kwargs):
        begin = perf_counter()
        value = fn(*args, **kwargs)
        phases[name] = perf_counter() - begin
        return value

    def gather(tensor):
        parts = _dist_gather_varlen_to_rank0(tensor, **transport)
        return torch.cat(parts).cpu() if parts is not None else None

    def return_rows(cpu_tensor, template):
        chunks = list(cpu_tensor.to(device, dtype=template.dtype).split(TOKENS)) if rank == 0 else None
        return _dist_scatter_varlen_from_rank0(
            rank0_chunks=chunks,
            feature_shape=tuple(template.shape[1:]),
            device=device,
            dtype=template.dtype,
            **transport,
        )

    def pack(*tensors):
        return torch.cat([t.contiguous().view(torch.uint8) for t in tensors], dim=1)

    def split(packed, *templates):
        widths = [x.shape[1] * x.element_size() for x in templates]
        return [part.contiguous().view(t.dtype) for part, t in zip(packed.split(widths, dim=1), templates)]

    def source_inputs(balance):
        tensors = [v.cpu() for v in (inputs, ids, scores)]
        if balance:
            tensors = split(plan.to_workers(pack(*tensors), host_group), *tensors)
        return tensors

    def source_outputs(values, templates, balance):
        if balance:
            values = split(plan.to_sources(pack(*values), host_group), *values)
        return tuple(x.to(device, dtype=t.dtype) for x, t in zip(values, templates))

    def lora_gradient(store, reduce):
        gradients = torch.cat([x.flatten().float() for x in store.values()])
        if reduce:
            dist.reduce(gradients, dst=0, group=host_group)
        return gradients if rank == 0 else None

    def forward(mode):
        if mode == "receiver":
            native.QLEN = sum(TOKENS)
            tensors = timed("input_transfer", lambda: [gather(x) for x in (inputs, ids, scores)])
            y = timed("cpu_forward", native._forward, owner_kernel, native.Batch(*tensors, None)) if rank == 0 else None
            return timed("output_transfer", return_rows, y, inputs), tensors
        balance = mode == "balanced"
        native.QLEN = plan.worker_counts[rank] if balance else TOKENS[rank]
        tensors = timed("input_transfer", source_inputs, balance)
        y = timed("cpu_forward", native._forward, kernel, native.Batch(*tensors, None)).bfloat16()
        return timed("output_transfer", source_outputs, (y,), (inputs,), balance)[0], tensors

    def backward(mode, tensors):
        if mode == "receiver":
            grad_y = timed("dy_transfer", gather, dy)
            if rank == 0:
                dx, ds = timed(
                    "cpu_backward",
                    native._backward,
                    owner_kernel,
                    owner_grads,
                    native.Batch(*tensors, grad_y),
                    accumulate=False,
                    grad_scale=1 / size,
                )
            else:
                dx = ds = None
            dx, ds = timed("dx_transfer", lambda: (return_rows(dx, inputs), return_rows(ds, score_template)))
            return dx, ds, timed("lora_aggregate", lora_gradient, owner_grads, False) if rank == 0 else None
        balance = mode == "balanced"
        grad_y = timed("dy_transfer", lambda: plan.to_workers(dy.cpu(), host_group) if balance else dy.cpu())
        dx, ds = timed(
            "cpu_backward",
            native._backward,
            kernel,
            grad_store,
            native.Batch(*tensors, grad_y),
            accumulate=False,
            grad_scale=1 / size,
        )
        dx, ds = timed("dx_transfer", source_outputs, (dx.bfloat16(), ds.bfloat16()), (inputs, score_template), balance)
        return dx, ds, timed("lora_aggregate", lora_gradient, grad_store, True)

    schedules = list(permutations(MODES)) * REPETITIONS
    Random(5489).shuffle(schedules)
    records, outputs = [], {}
    with torch.no_grad():
        for repeat, order in enumerate([*schedules[:2], *schedules], start=-2):
            for mode in order:
                torch.npu.synchronize()
                dist.barrier(group=host_group)
                phases.clear()
                begin = perf_counter()
                y, tensors = forward(mode)
                timed("forward_sync", torch.npu.synchronize)
                forward_done = perf_counter()
                dx, ds, grads = backward(mode, tensors)
                timed("backward_sync", torch.npu.synchronize)
                done = perf_counter()
                if repeat >= 0:
                    sample = dict(trial=repeat, mode=mode, forward_s=forward_done - begin, total_s=done - begin)
                    sample["phases"] = dict(phases)
                    records.append(sample)
                if repeat == -1:
                    outputs[mode] = tuple(v.cpu().clone() if v is not None else None for v in (y, dx, ds, grads))

    errors = {
        mode: {
            label: relative_l2(out, ref)
            for label, out, ref in zip(
                ("output", "input_gradient", "router_gradient", "lora_gradient"), values, outputs["local"]
            )
            if ref is not None
        }
        for mode, values in outputs.items()
        if mode != "local"
    }
    rank_records, rank_errors = [None] * size, [None] * size
    dist.all_gather_object(rank_records, records, group=host_group)
    dist.all_gather_object(rank_errors, errors, group=host_group)
    if rank == 0:
        batches = []
        for i in range(size):
            native.QLEN = TOKENS[i]
            batches.append(native._make_batch((i, (i + 1) % native.EXPERTS), 5489 + i))
        columns = zip(*((b.inputs, b.expert_ids, b.route_weights, b.grad_output) for b in batches))
        with torch.enable_grad():
            _, _, _, expected = native._reference(experts, lora, native.Batch(*(torch.cat(parts) for parts in columns)))
        gradient = torch.cat([expected[n].flatten() for n in grad_store]) / size
        oracle_lora = {mode: relative_l2(values[-1], gradient) for mode, values in outputs.items()}
        assert max(oracle_lora.values()) <= 0.05, oracle_lora
        report = dict(
            tokens=TOKENS,
            compute_tokens=plan.worker_counts,
            moved_tokens=plan.moved_rows,
            trials=len(schedules),
            rank_records=rank_records,
            rank_errors=rank_errors,
            cpu_variant=kt_kernel.__cpu_variant__,
            total_cpu_workers=size,
            experts=native.EXPERTS,
            topk=native.TOP_K,
            numa_nodes=nodes[:size],
            oracle_lora=oracle_lora,
        )
        print("KT_TOKEN_BALANCE", json.dumps(report), flush=True)
    assert all(
        value <= (0.05 if key == "lora_gradient" else 0.03)
        for rank_error in rank_errors
        for mode_errors in rank_error.values()
        for key, value in mode_errors.items()
    ), rank_errors
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
