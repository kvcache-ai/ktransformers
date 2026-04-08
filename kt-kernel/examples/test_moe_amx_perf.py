import os, sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
print("sys.path:", sys.path)

import torch
from kt_kernel import kt_kernel_ext

# Model configuration
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
qlen = 1
# qlen = 640
layer_num = 1

# Test configuration
num_threads = 90
CPUInfer = kt_kernel_ext.CPUInfer(num_threads)
# validation_iter = 10000
validation_iter = 2
k_group_size = 64
debug_print_count = 16  # Number of values to print in debug output
physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()

# Performance test configuration
perf_warmup_iter = 5  # Number of warmup iterations for performance test
perf_test_iter = 20  # Number of iterations for performance measurement
perf_qlen = 128  # Sequence length for performance testing


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj, debug_expert_id=None, debug_print=False):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())

    if debug_print and debug_expert_id is not None:
        print(f"[TORCH DEBUG] Expert {debug_expert_id}:")
        print(f"  gate_buf[:{debug_print_count}] = {gate_buf.flatten()[:debug_print_count]}")
        print(f"  up_buf[:{debug_print_count}] = {up_buf.flatten()[:debug_print_count]}")

    intermediate = act_fn(gate_buf) * up_buf

    if debug_print and debug_expert_id is not None:
        print(f"  intermediate[:{debug_print_count}] = {intermediate.flatten()[:debug_print_count]}")

    ret = torch.mm(intermediate, down_proj.t())

    if debug_print and debug_expert_id is not None:
        print(f"  down_output[:{debug_print_count}] = {ret.flatten()[:debug_print_count]}")

    return ret


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, debug_print=False):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    # Get the first expert from expert_ids array to match AWQ-MoE behavior
    target_debug_expert = expert_ids[0, 0].item()  # First expert in expert_ids array

    outputs = []
    start_idx = 0
    activated_experts = []

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        activated_experts.append(i)
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        # Only debug the target expert that matches AWQ-MoE's first expert
        should_debug = debug_print and i == target_debug_expert
        expert_out = mlp_torch(
            tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i], debug_expert_id=i, debug_print=should_debug
        )
        outputs.append(expert_out)
        start_idx = end_idx

    if debug_print:
        print(f"[TORCH DEBUG] Processing activated experts: {activated_experts}")
        print(f"[TORCH DEBUG] Target debug expert (matches AWQ): {target_debug_expert}")

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )

    if debug_print:
        print(f"[TORCH DEBUG] Final MoE output[:{debug_print_count}] = {t_output.flatten()[:debug_print_count]}")

    return t_output


def test_moe(quant_mode: str):
    assert (
        quant_mode == "bf16"
        or quant_mode == "int8"
        or quant_mode == "int4"
        or quant_mode == "int4_1"
        or quant_mode == "int4_1k"
    )
    with torch.inference_mode(mode=True):
        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = (
                torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
                .to("cpu")
                .contiguous()
            )
            up_proj = (
                torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
                .to("cpu")
                .contiguous()
            )
            down_proj = (
                torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
                .to("cpu")
                .contiguous()
            )
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()
            config.gate_scale = 0
            config.pool = CPUInfer.backend_
            if quant_mode == "bf16":
                moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up_task())
                CPUInfer.sync()
            elif quant_mode == "int8":
                moe = kt_kernel_ext.moe.AMXInt8_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                # CPUInfer.submit(moe.warm_up_task())
                # CPUInfer.sync()
            elif quant_mode == "int4":
                moe = kt_kernel_ext.moe.AMXInt4_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up_task())
                CPUInfer.sync()
            elif quant_mode == "int4_1":
                moe = kt_kernel_ext.moe.AMXInt4_1_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up_task())
                CPUInfer.sync()
            elif quant_mode == "int4_1k":
                config.quant_config.bits = 4
                config.quant_config.group_size = k_group_size
                config.quant_config.zero_point = True
                moe = kt_kernel_ext.moe.AMXInt4_1KGroup_MOE(config)
                # import debugpy
                # debugpy.listen(("127.0.0.1", 5678))
                # debugpy.wait_for_client()
                # debugpy.breakpoint()
                print(f"the physical_logical map:{physical_to_logical_map.data_ptr()}")
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                # CPUInfer.submit(moe.warm_up_task())
                # CPUInfer.sync()
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)

        # validation
        for i in range(validation_iter):
            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            input = input / 100
            moe = moes[i % layer_num]

            # Enable debug for first few iterations
            enable_debug = i < 2
            enable_debug = False
            if enable_debug:
                print(f"\n=== Iteration {i} Debug Info ===")
                print(f"input[:{debug_print_count}] = {input.flatten()[:debug_print_count]}")
                print(f"expert_ids = {expert_ids}")
                print(f"weights = {weights}")
                # Print which experts will be activated for comparison
                activated_experts = []
                for token in range(expert_ids.shape[0]):
                    for expert_idx in range(expert_ids.shape[1]):
                        expert_id = expert_ids[token][expert_idx].item()
                        if expert_id not in activated_experts:
                            activated_experts.append(expert_id)
                print(f"[TORCH DEBUG] Activated experts: {sorted(activated_experts)}")
                print(f"[TORCH DEBUG] First expert from expert_ids array: {expert_ids[0, 0].item()}")
            print(f"expert_ids = {expert_ids}")
            # print('expert ids:',expert_ids)
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            if enable_debug:
                print(f"[AWQ-MOE DEBUG] AMX output[:{debug_print_count}] = {output.flatten()[:debug_print_count]}")

            gate_proj = gate_projs[i % layer_num]
            up_proj = up_projs[i % layer_num]
            down_proj = down_projs[i % layer_num]
            t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, debug_print=enable_debug)
            print("torch output", t_output)
            print("amx output", output)

            # print(output - t_output)
            # print(torch.abs(output - t_output))
            diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
            # print(f'output_shape:{output.shape}, t_output_shape:{t_output.shape}\n')
            print(f"Iteration {i}, diff = {diff:.6f}")

            if enable_debug:
                abs_diff = torch.abs(output - t_output)
                print(f"[COMPARE] Max abs diff = {torch.max(abs_diff):.6f}")
                print(f"[COMPARE] Mean abs diff = {torch.mean(abs_diff):.6f}")
                print(f"[COMPARE] Relative diff = {diff:.6f}")
                print("=" * 50)

            if quant_mode == "int4" or quant_mode == "int4_1" or quant_mode == "int4_1k":
                assert diff < 0.35
            else:
                assert diff < 0.05


def test_moe_performance(quant_mode: str):
    """
    Test MOE inference performance (forward latency and throughput).

    Measures:
    - Forward pass latency (ms)
    - Throughput (tokens/second)

    Args:
        quant_mode: Quantization mode, "bf16" or "int8"
    """
    import time

    assert quant_mode in ("bf16", "int8"), f"Performance test only supports bf16 and int8, got {quant_mode}"

    print(f"\n{'='*60}")
    print(f"Performance Test - {quant_mode.upper()} mode (Inference)")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  qlen (batch size): {perf_qlen}")
    print(f"  warmup iterations: {perf_warmup_iter}")
    print(f"  test iterations: {perf_test_iter}")
    print(f"  num_threads: {num_threads}")
    print(f"{'='*60}")

    with torch.inference_mode(mode=True):
        # Initialize weights
        gate_proj = (
            torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
            .to("cpu")
            .contiguous()
        )
        up_proj = (
            torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
            .to("cpu")
            .contiguous()
        )
        down_proj = (
            torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
            .to("cpu")
            .contiguous()
        )

        # Create MOE config
        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()
        config.gate_scale = 0
        config.pool = CPUInfer.backend_

        # Create MOE instance based on quant_mode
        if quant_mode == "bf16":
            moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
        elif quant_mode == "int8":
            moe = kt_kernel_ext.moe.AMXInt8_MOE(config)
        else:
            raise ValueError(f"Unsupported quant_mode for performance test: {quant_mode}")

        print(f"[INFO] Using {quant_mode.upper()} MOE class")

        # Load weights
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        # Warm up task
        if quant_mode == "bf16":
            CPUInfer.submit(moe.warm_up_task())
            CPUInfer.sync()

        # Prepare test data
        bsz_tensor = torch.tensor([perf_qlen], device="cpu")
        expert_ids = torch.stack(
            [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(perf_qlen)]
        ).contiguous()
        weights = torch.rand((perf_qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        input_data = torch.randn((perf_qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100
        output = torch.empty((perf_qlen, hidden_size), dtype=torch.bfloat16).contiguous()

        # =========================================================================
        # Warmup Phase
        # =========================================================================
        print(f"\n[INFO] Warmup phase ({perf_warmup_iter} iterations)...")
        for _ in range(perf_warmup_iter):
            CPUInfer.submit(
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
            CPUInfer.sync()

        # =========================================================================
        # Forward Performance Test
        # =========================================================================
        print(f"[INFO] Testing forward pass performance ({perf_test_iter} iterations)...")
        forward_times = []
        for _ in range(perf_test_iter):
            start_time = time.perf_counter()
            CPUInfer.submit(
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
            CPUInfer.sync()
            end_time = time.perf_counter()
            forward_times.append((end_time - start_time) * 1000)  # Convert to ms

        # =========================================================================
        # Results Summary
        # =========================================================================
        import statistics

        avg_forward = statistics.mean(forward_times)
        std_forward = statistics.stdev(forward_times) if len(forward_times) > 1 else 0
        min_forward = min(forward_times)
        max_forward = max(forward_times)

        # Calculate throughput (tokens per second)
        forward_throughput = perf_qlen / (avg_forward / 1000)  # tokens/second

        print(f"\n{'='*60}")
        print(f"Performance Results - {quant_mode.upper()} mode (Inference)")
        print(f"{'='*60}")
        print(f"\nForward Pass:")
        print(f"  Average latency: {avg_forward:.3f} ms (±{std_forward:.3f})")
        print(f"  Min latency:     {min_forward:.3f} ms")
        print(f"  Max latency:     {max_forward:.3f} ms")
        print(f"  Throughput:      {forward_throughput:.1f} tokens/s")

        print(f"\n[OK] Performance Test - {quant_mode.upper()} mode completed")

        return {
            "quant_mode": quant_mode,
            "forward_avg_ms": avg_forward,
            "forward_std_ms": std_forward,
            "forward_throughput": forward_throughput,
        }


def run_performance_tests():
    """Run performance tests for AMXBF16 and AMXINT8 modes (Inference)."""
    print("\n" + "=" * 70)
    print(" MOE AMX Inference Performance Test Suite")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  expert_num: {expert_num}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  perf_qlen: {perf_qlen}")
    print(f"  num_threads: {num_threads}")
    print("=" * 70)

    # Only test BF16 and INT8 as requested
    quant_modes = ["bf16", "int8"]

    results = []
    try:
        for quant_mode in quant_modes:
            result = test_moe_performance(quant_mode)
            results.append(result)

        # Print comparison table
        print("\n" + "=" * 70)
        print(" Performance Comparison Summary (Inference)")
        print("=" * 70)
        print(f"\n{'Mode':<10} {'Forward(ms)':<15} {'Throughput(tok/s)':<20}")
        print("-" * 45)
        for r in results:
            print(
                f"{r['quant_mode'].upper():<10} " f"{r['forward_avg_ms']:<15.3f} " f"{r['forward_throughput']:<20.1f}"
            )
        print("-" * 45)

        # Calculate speedup if we have both results
        if len(results) == 2:
            bf16_forward = results[0]["forward_avg_ms"]
            int8_forward = results[1]["forward_avg_ms"]
            speedup = bf16_forward / int8_forward
            print(f"\nINT8 vs BF16 speedup: {speedup:.2f}x")

        print("\n" + "=" * 70)
        print(" PERFORMANCE TESTS COMPLETED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Performance test failed with error: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)

    return results


def run_all_tests():
    """Run all MOE accuracy tests for bf16 and int8 modes."""
    print("\n" + "=" * 70)
    print(" MOE AMX Inference Accuracy Test Suite")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  expert_num: {expert_num}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  qlen: {qlen}")
    print(f"  num_threads: {num_threads}")
    print("=" * 70)

    # Only test BF16 and INT8 as requested
    quant_modes = ["bf16", "int8"]

    try:
        for quant_mode in quant_modes:
            print(f"\n{'='*70}")
            print(f" Testing MOE AMX - {quant_mode.upper()} Mode")
            print(f"{'='*70}")
            test_moe(quant_mode)

        print("\n" + "=" * 70)
        print(" ALL ACCURACY TESTS PASSED!")
        print(f" Tested quantization modes: {', '.join(m.upper() for m in quant_modes)}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MOE AMX Inference Test Suite")
    parser.add_argument(
        "--mode",
        choices=["all", "accuracy", "perf"],
        default="perf",
        help="Test mode: 'all' runs both, 'accuracy' runs correctness tests, 'perf' runs performance tests",
    )
    parser.add_argument(
        "--qlen",
        type=int,
        default=None,
        help=f"Override perf_qlen for performance tests (default: {perf_qlen})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=f"Override warmup iterations for performance tests (default: {perf_warmup_iter})",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help=f"Override test iterations for performance tests (default: {perf_test_iter})",
    )
    args = parser.parse_args()

    # Override performance test parameters if specified
    if args.qlen is not None or args.warmup is not None or args.iter is not None:
        # Need to use global to modify module-level variables
        if args.qlen is not None:
            globals()["perf_qlen"] = args.qlen
        if args.warmup is not None:
            globals()["perf_warmup_iter"] = args.warmup
        if args.iter is not None:
            globals()["perf_test_iter"] = args.iter

    if args.mode == "all":
        run_all_tests()
        run_performance_tests()
    elif args.mode == "accuracy":
        run_all_tests()
    elif args.mode == "perf":
        run_performance_tests()
