#!/usr/bin/env python
"""Compare shared BF16 recompute storage with the retained/legacy layout.

Run against either build with --dump to compare independent baseline/candidate
extensions as well. Every tensor includes all six expert LoRA gradients.
"""

import argparse
import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="default")

try:
    import kt_kernel

    ext = kt_kernel.kt_kernel_ext
except (ImportError, AttributeError):
    ext = None


NAMES = ("gate_lora_a", "gate_lora_b", "up_lora_a", "up_lora_b", "down_lora_a", "down_lora_b")
E, H, I = 9, 256, 256


def _random(shape, generator, scale=0.04):
    return (torch.randn(shape, generator=generator) * scale).bfloat16().contiguous()


def _layer(cpu, *, shared, rank, top_k, dropout, layer, max_len):
    generator = torch.Generator().manual_seed(1729 + layer)
    weights = {
        "gate_proj": _random((E, I, H), generator),
        "up_proj": _random((E, I, H), generator),
        "down_proj": _random((E, H, I), generator),
    }
    for name, shape in zip(NAMES, ((E, rank, H), (E, I, rank), (E, rank, H), (E, I, rank), (E, rank, I), (E, H, rank))):
        weights[name] = _random(shape, generator)
    mapping = torch.arange(E, dtype=torch.int64)
    config = ext.moe.MOESFTConfig(E, top_k, H, I)
    config.max_len = max_len
    config.max_cache_depth = 1
    # Layer IDs must match because they participate in the dropout seed.
    config.layer_idx = layer
    config.lora_rank = rank
    config.lora_alpha = 2.0 * rank
    config.lora_dropout = dropout
    config.full_weight_grad = False
    config.authoritative_optimizer_grads = True
    config.share_backward_bb = True
    config.share_cache_pool = shared
    config.physical_to_logical_map = mapping.data_ptr()
    for name, tensor in weights.items():
        setattr(config, name, tensor.data_ptr())
    config.pool = cpu.backend_
    moe = ext.moe.AMXBF16_SFT_MOE(config)
    moe.load_weights()
    grads = {name: torch.full_like(weights[name], 7.0) for name in NAMES}
    return moe, weights, grads, mapping


def _batch(length, top_k, seed, skewed):
    generator = torch.Generator().manual_seed(seed)
    x = _random((length, H), generator, 0.2)
    dy = _random((length, H), generator, 0.1)
    ids = torch.arange(top_k).repeat(length, 1)
    if not skewed:
        ids = (ids + torch.arange(length).unsqueeze(1)) % E
    routes = torch.softmax(torch.randn(length, top_k, generator=generator), -1)
    return x, dy, ids.contiguous(), routes.contiguous()


def _forward(layer, batch, top_k, save):
    x, _, ids, routes = batch
    qlen = torch.tensor([x.shape[0]], dtype=torch.int32)
    # Guard the public output tensors too; native scratch is checked separately
    # by the guarded allocator/ASan run used for this regression.
    storage = torch.full((x.numel() + 64,), 31, dtype=torch.bfloat16)
    out = storage[32:-32].view_as(x)
    layer[0].forward_sft(qlen.data_ptr(), top_k, ids.data_ptr(), routes.data_ptr(), x.data_ptr(), out.data_ptr(), save)
    assert torch.all(storage[:32] == 31) and torch.all(storage[-32:] == 31)
    return out.clone()


def _backward(layer, batch, accumulate, ga):
    moe, _, grads, _ = layer
    # Two independent comparison instances use the same layer ID. Explicitly
    # prepare this instance's views instead of relying on the owner-ID shortcut.
    moe.submit_backward_repack()
    moe.wait_backward_repack()
    x, dy, _, routes = batch
    dx_storage = torch.full((x.numel() + 64,), 31, dtype=torch.bfloat16)
    dr_storage = torch.full((routes.numel() + 32,), 31.0)
    dx = dx_storage[32:-32].view_as(x)
    dr = dr_storage[16:-16].view_as(routes)
    moe.backward(
        dy.data_ptr(),
        dx.data_ptr(),
        *(grads[name].data_ptr() for name in NAMES),
        dr.data_ptr(),
        0,
        0,
        0,
        accumulate,
        1.0 / ga,
    )
    assert torch.all(dx_storage[:32] == 31) and torch.all(dx_storage[-32:] == 31)
    assert torch.all(dr_storage[:16] == 31) and torch.all(dr_storage[-16:] == 31)
    return {"dx": dx.clone(), "droute": dr.clone(), **{name: value.clone() for name, value in grads.items()}}


def _check(actual, expected, exact):
    assert actual.keys() == expected.keys()
    for name in actual:
        a, b = actual[name].float(), expected[name].float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        if exact:
            assert torch.equal(a, b), f"{name}: single-thread result changed"
        else:
            torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-5, msg=name)


def run_case(tp, threads, top_k, rank, ga, dropout):
    torch.set_num_threads(1)
    worker = ext.WorkerPoolConfig()
    worker.subpool_count = tp
    worker.subpool_numa_map = [0] * tp
    worker.subpool_thread_count = [threads] * tp
    cpu = ext.CPUInfer(worker)
    options = dict(rank=rank, top_k=top_k, dropout=dropout)
    # The second layer grows the process-shared arenas; returning to layer 0
    # must reacquire every borrowed pointer. Nonaligned lengths stress padding.
    shapes = ((17, 32), (129, 160), (63, 64))
    legacy = [_layer(cpu, shared=False, layer=i, max_len=m, **options) for i, (_, m) in enumerate(shapes)]
    shared = [_layer(cpu, shared=True, layer=i, max_len=m, **options) for i, (_, m) in enumerate(shapes)]
    exact = tp == 1 and threads == 1
    records = {}
    for micro in range(ga):
        batches = [_batch(n, top_k, 1000 + micro * 7 + i, micro % 2 == 0) for i, (n, _) in enumerate(shapes)]
        # Initial checkpoint forward: storage is populated, no cache is pushed.
        for i, batch in enumerate(batches):
            a = _forward(shared[i], batch, top_k, False)
            b = _forward(legacy[i], batch, top_k, False)
            _check({"initial": a}, {"initial": b}, exact)
            records[f"m{micro}.l{i}.initial"] = a
        for i in reversed(range(len(shapes))):
            results = []
            for layers in (legacy, shared):
                out = _forward(layers[i], batches[i], top_k, True)
                values = _backward(layers[i], batches[i], micro > 0, ga)
                values["forward"] = out
                results.append(values)
            _check(results[1], results[0], exact)
            records.update({f"m{micro}.l{i}.{key}": value for key, value in results[1].items()})
    for i, (old, new) in enumerate(zip(legacy, shared)):
        updates = []
        for layer in (old, new):
            parameters = [layer[1][name].float().clone().requires_grad_() for name in NAMES]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4)
            for name, parameter in zip(NAMES, parameters):
                gradient = layer[2][name]
                assert torch.count_nonzero(gradient), f"{name}: missing expert gradient"
                active = set(range(top_k)) if ga == 1 else set(range(E))
                for expert in range(E):
                    nonzero = bool(torch.count_nonzero(gradient[expert]))
                    assert nonzero == (expert in active), f"{name}: wrong gradient activity for expert {expert}"
                parameter.grad = gradient.float()
            optimizer.step()
            updates.append({name: parameter.detach() for name, parameter in zip(NAMES, parameters)})
        _check(updates[1], updates[0], exact)
        records.update({f"l{i}.updated.{key}": value for key, value in updates[1].items()})
    return records


# Pairwise coverage keeps the default CPU suite bounded; --matrix runs the
# complete Cartesian grid for a hardware acceptance run.
CASES = [(1, 1, 1, 8, 1, 0.0), (1, 1, 8, 16, 4, 0.1), (2, 4, 1, 32, 4, 0.1), (2, 4, 8, 8, 4, 0.0)]


@pytest.mark.cpu
@pytest.mark.parametrize("case", CASES)
def test_bf16_recompute_buffers(case):
    if ext is None or not hasattr(ext.moe, "AMXBF16_SFT_MOE"):
        pytest.skip("BF16 SFT kernel unavailable")
    run_case(*case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--dump")
    parser.add_argument("--compare")
    args = parser.parse_args()
    cases = list(itertools.product((1, 2), (1, 4), (1, 8), (8, 16, 32), (1, 4), (0.0, 0.1))) if args.matrix else CASES
    reference = torch.load(args.compare, weights_only=True) if args.compare else None
    result = {}
    for case in cases:
        values = run_case(*case)
        key = str(case)
        if reference is not None:
            _check(values, reference[key], case[0] == case[1] == 1)
        result[key] = values
        print(f"PASS {case}", flush=True)
    if args.dump:
        torch.save(result, args.dump)
