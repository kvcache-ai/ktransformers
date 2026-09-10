"""CPU-only Kimi-shaped routed-expert replay; no model checkpoint is required."""

import argparse
import json
import resource
import statistics
import time

import torch
from kt_kernel import kt_kernel_ext as ext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=43)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=48, help="Total KT worker threads across TP shards")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensors", help="Optional output/gradient snapshot for A/B numerical comparison")
    parser.add_argument("--window-check", action="store_true", help="Change routing across three-microbatch windows")
    parser.add_argument("--skew", action="store_true", help="Place hot experts at the end of the ID order")
    args = parser.parse_args()
    assert 0 < args.top_k <= args.experts
    assert args.hidden % 256 == 0 and args.intermediate % (256 * args.tp) == 0
    assert args.threads >= args.tp and args.repeat > 0
    torch.set_num_threads(2)
    torch.manual_seed(42)

    workers = ext.WorkerPoolConfig()
    workers.subpool_count = args.tp
    workers.subpool_numa_map = list(range(args.tp))
    workers.subpool_thread_count = [args.threads // args.tp + (i < args.threads % args.tp) for i in range(args.tp)]
    cpu = ext.CPUInfer(workers)
    config = ext.moe.MOESFTConfig(args.experts, args.top_k, args.hidden, args.intermediate)
    config.max_len = max(args.tokens, 32)
    config.max_cache_depth = 1
    config.pool = cpu.backend_
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.lora_rank = 8
    config.lora_alpha = 16.0
    config.authoritative_optimizer_grads = True
    mapping = torch.arange(args.experts, dtype=torch.int64)
    config.physical_to_logical_map = mapping.data_ptr()
    weights, lora, grads = {}, {}, {}
    for name, rows, cols in (
        ("gate", args.intermediate, args.hidden),
        ("up", args.intermediate, args.hidden),
        ("down", args.hidden, args.intermediate),
    ):
        weights[name] = torch.randint(0, 256, (args.experts, rows, cols // 2), dtype=torch.uint8)
        weights[name + "_scale"] = torch.full((args.experts, rows, cols // 32), 0.003, dtype=torch.bfloat16)
        setattr(config, name + "_proj", weights[name].data_ptr())
        setattr(config, name + "_scale", weights[name + "_scale"].data_ptr())
        for suffix, shape in (("a", (args.experts, 8, cols)), ("b", (args.experts, rows, 8))):
            key = name + "_lora_" + suffix
            lora[key] = (torch.randn(shape) * 0.01).bfloat16()
            grads[key] = torch.empty_like(lora[key])
            setattr(config, key, lora[key].data_ptr())
    moe = ext.moe.AMXInt4_KGroup_SFT_MOE(config)
    cpu.submit(moe.load_weights_task())
    cpu.sync()
    lengths = torch.tensor([args.tokens], dtype=torch.int32)
    routing_scores = torch.rand(args.tokens, args.experts)
    if args.skew:
        routing_scores[:, -args.top_k :] += 0.5
    ids = routing_scores.topk(args.top_k, dim=1).indices.contiguous()
    routes = torch.rand(args.tokens, args.top_k).softmax(dim=1).contiguous()
    inputs = (torch.randn(args.tokens, args.hidden) * 0.1).bfloat16()
    grad_output = (torch.randn_like(inputs) * 0.1).contiguous()
    output, grad_input = torch.empty_like(inputs), torch.empty_like(inputs)
    grad_routes = torch.empty_like(routes)
    ordered_grads = [grads[name + "_lora_" + suffix] for name in ("gate", "up", "down") for suffix in ("a", "b")]
    result = {
        "args": vars(args),
        "torch": torch.__version__,
        "extension": ext.__file__,
        "routed_rows": torch.bincount(ids.flatten(), minlength=args.experts).tolist(),
        "samples": [],
    }
    snapshots = {}
    active_window = set()
    for iteration in range(args.warmup + args.repeat):
        accumulate = args.window_check and iteration % 3 != 0
        if args.window_check:
            pool_size = min(args.experts, 2 * args.top_k)
            offset = (iteration * args.top_k) % args.experts
            ids = (torch.rand(args.tokens, pool_size).topk(args.top_k, dim=1).indices + offset) % args.experts
            ids = ids.contiguous()
            routes = torch.rand(args.tokens, args.top_k).softmax(dim=1).contiguous()
            inputs = (torch.randn_like(inputs) * 0.1).contiguous()
            grad_output = (torch.randn_like(inputs) * 0.1).contiguous()
            if not accumulate:
                active_window.clear()
            active_window.update(ids.flatten().tolist())
        start = time.perf_counter()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        cpu.submit(
            moe.forward_sft_task(
                lengths.data_ptr(),
                args.top_k,
                ids.data_ptr(),
                routes.data_ptr(),
                inputs.data_ptr(),
                output.data_ptr(),
                True,
            )
        )
        cpu.sync()
        mid = time.perf_counter()
        cpu_mid = resource.getrusage(resource.RUSAGE_SELF)
        cpu.submit(
            moe.backward_task(
                grad_output.data_ptr(),
                grad_input.data_ptr(),
                *(g.data_ptr() for g in ordered_grads),
                grad_routes.data_ptr(),
                0,
                0,
                0,
                accumulate,
                0.25 if args.window_check else 1.0,
            )
        )
        cpu.sync()
        end = time.perf_counter()
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        for value in (output, grad_input, grad_routes, *ordered_grads):
            assert torch.isfinite(value).all()
        if args.window_check:
            inactive = sorted(set(range(args.experts)) - active_window)
            for value in ordered_grads:
                assert not inactive or torch.count_nonzero(value[inactive]) == 0
            snapshots.update(
                {
                    f"micro{iteration}/{name}": value.clone()
                    for name, value in {
                        "output": output,
                        "grad_input": grad_input,
                        "grad_routes": grad_routes,
                        **grads,
                    }.items()
                }
            )
        if iteration >= args.warmup:
            sample = {"forward_s": mid - start, "backward_s": end - mid, "total_s": end - start}
            sample["forward_cpu_s"] = cpu_mid.ru_utime + cpu_mid.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime
            sample["backward_cpu_s"] = cpu_end.ru_utime + cpu_end.ru_stime - cpu_mid.ru_utime - cpu_mid.ru_stime
            result["samples"].append(sample)
            print(json.dumps(sample), flush=True)
    result["median"] = {
        key: statistics.median(row[key] for row in result["samples"]) for key in ("forward_s", "backward_s", "total_s")
    }
    result["gradient_absmax"] = {name: float(value.abs().max()) for name, value in grads.items()}
    assert all(v > 0 for v in result["gradient_absmax"].values())
    with open(args.output, "x") as handle:
        json.dump(result, handle, indent=2)
    if args.tensors:
        torch.save(
            snapshots or {"output": output, "grad_input": grad_input, "grad_routes": grad_routes, **grads}, args.tensors
        )
    print(json.dumps({"median": result["median"], "gradient_absmax": result["gradient_absmax"]}), flush=True)


if __name__ == "__main__":
    main()
