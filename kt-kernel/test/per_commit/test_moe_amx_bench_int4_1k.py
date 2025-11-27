#!/usr/bin/env python
# coding=utf-8
"""AMX MOE INT4_1K benchmark tests for KT-Kernel.

Benchmarks performance (bandwidth and FLOPS) of AMX-accelerated INT4_1K group quantization MOE operations.
"""

import os
import sys
import time
import json
import subprocess
import platform
import pytest

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

# Register this test for CPU CI with estimated runtime of 300 seconds
register_cpu_ci(est_time=300, suite="default")

# Check if dependencies are available
try:
    import torch
    import kt_kernel_ext
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    import_error = str(e)

# Test parameters (from original bench_moe_amx.py)
expert_num = 16
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
layer_num = 2
qlen = 2048
warm_up_iter = 1000
test_iter = 2000
k_group_size = 64

# Worker configuration
worker_config_dict = {
    "subpool_count": 2,
    "subpool_numa_map": [0, 1],
    "subpool_thread_count": [45, 45],
}
CPUINFER_PARAM = 90


def get_git_commit():
    """Get current git commit information."""
    result = {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip()
        result["commit"] = commit
        result["commit_message"] = commit_msg

        dirty_output = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        if dirty_output:
            result["dirty"] = True
            result["dirty_files"] = dirty_output.splitlines()
        else:
            result["dirty"] = False
    except Exception as e:
        result["commit"] = None
        result["commit_message"] = None
        result["dirty"] = None
        result["error"] = str(e)
    return result


def get_system_info():
    """Get system information including CPU model, memory, cores, and sockets."""
    info = {}
    uname = platform.uname()
    info["system_name"] = uname.system
    info["node_name"] = uname.node

    # Get CPU model (Linux only)
    cpu_model = None
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            cpu_model = f"Error: {e}"
    info["cpu_model"] = cpu_model

    # Get memory size in GB (Linux only)
    mem_total_gb = None
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = float(line.split(":", 1)[1].split()[0])
                        mem_total_gb = round(mem_kb / (1024 * 1024), 2)
                        break
        except Exception as e:
            mem_total_gb = f"Error: {e}"
    info["memory_size_GB"] = mem_total_gb

    # Get CPU core count
    info["cpu_core_count"] = os.cpu_count()

    # Get socket count
    sockets = set()
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "physical id" in line:
                        sockets.add(line.split(":", 1)[1].strip())
        except Exception as e:
            sockets = set()
    info["cpu_socket_count"] = len(sockets) if len(sockets) > 0 else 1

    return info


def record_results(result, filename):
    """Append results to JSONL file."""
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


@pytest.mark.cpu
def test_moe_amx_int4_1k_benchmark():
    """Benchmark AMX INT4_1K MOE performance."""
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")

    quant_mode = "int4_1k"
    bytes_per_elem = 0.5

    # Setup output file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "bench_moe_amx_int4_1k.jsonl")

    with torch.inference_mode():
        # Initialize CPUInfer with worker config
        worker_config = kt_kernel_ext.WorkerPoolConfig()
        worker_config.subpool_count = worker_config_dict["subpool_count"]
        worker_config.subpool_numa_map = worker_config_dict["subpool_numa_map"]
        worker_config.subpool_thread_count = worker_config_dict["subpool_thread_count"]
        CPUInfer = kt_kernel_ext.CPUInfer(worker_config)

        # Initialize MOE layers
        moes = []
        for layer_index in range(layer_num):
            gate_proj = (
                torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cuda")
                .to("cpu")
                .contiguous()
            )
            up_proj = (
                torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cuda")
                .to("cpu")
                .contiguous()
            )
            down_proj = (
                torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32, device="cuda")
                .to("cpu")
                .contiguous()
            )
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()
            config.pool = CPUInfer.backend_

            # Configure INT4_1K quantization settings
            config.quant_config.bits = 4
            config.quant_config.group_size = k_group_size
            config.quant_config.zero_point = True

            moe = kt_kernel_ext.moe.AMXInt4_1KGroup_MOE(config)
            CPUInfer.submit(moe.load_weights_task())
            CPUInfer.sync()
            moes.append(moe)

        # Generate test data
        gen_iter = 3000
        expert_ids = (
            torch.rand(gen_iter * qlen, expert_num, device="cpu")
            .argsort(dim=-1)[:, :num_experts_per_tok]
            .reshape(gen_iter, qlen * num_experts_per_tok)
            .to("cpu")
            .contiguous()
        )
        weights = (
            torch.rand((gen_iter, qlen, num_experts_per_tok), dtype=torch.float32, device="cpu").to("cpu").contiguous()
        )
        input_tensor = (
            torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cuda").to("cpu").contiguous()
        )
        output_tensor = (
            torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cuda").to("cpu").contiguous()
        )
        bsz_tensor = torch.tensor([qlen], device="cpu")

        # Warm-up iterations
        print(f"Running warm-up for {warm_up_iter} iterations...")
        for i in tqdm(range(warm_up_iter), desc="Warm-up"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i % gen_iter].data_ptr(),
                    weights[i % gen_iter].data_ptr(),
                    input_tensor[i % layer_num].data_ptr(),
                    output_tensor[i % layer_num].data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

        # Test iterations
        print(f"Running test for {test_iter} iterations...")
        start = time.perf_counter()
        for i in tqdm(range(test_iter), desc="Testing"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i % gen_iter].data_ptr(),
                    weights[i % gen_iter].data_ptr(),
                    input_tensor[i % layer_num].data_ptr(),
                    output_tensor[i % layer_num].data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start

        # Calculate performance metrics
        time_per_iter_us = total_time / test_iter * 1e6
        bandwidth = (
            hidden_size
            * intermediate_size
            * 3
            * num_experts_per_tok
            * (1 / 8 * 256 * (1 - (31 / 32) ** qlen))
            * bytes_per_elem
            * test_iter
            / total_time
            / 1e9
        )  # GB/s
        flops = (
            hidden_size * intermediate_size * qlen * 3 * num_experts_per_tok * 2 * test_iter / total_time / 1e12
        )  # TFLOPS

        print("Quant mode: ", quant_mode)
        print("Time(s): ", total_time)
        print("Iteration: ", test_iter)
        print("Time(us) per iteration: ", time_per_iter_us)
        print("Bandwidth: ", bandwidth, "GB/s")
        print("Flops: ", flops, "TFLOPS")

        # Record results
        result = {
            "quant_mode": quant_mode,
            "total_time_seconds": total_time,
            "iterations": test_iter,
            "time_per_iteration_us": time_per_iter_us,
            "bandwidth_GBs": bandwidth,
            "flops_TFLOPS": flops,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "test_parameters": {
                "expert_num": expert_num,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "max_len": max_len,
                "num_experts_per_tok": num_experts_per_tok,
                "layer_num": layer_num,
                "qlen": qlen,
                "warm_up_iter": warm_up_iter,
                "test_iter": test_iter,
                "k_group_size": k_group_size,
                "CPUInfer_parameter": CPUINFER_PARAM,
            },
        }
        result.update(get_git_commit())
        result.update(get_system_info())
        record_results(result, json_path)

        print(f"Results saved to {json_path}")


def run_all_tests():
    """Run all tests in this file (for standalone execution)."""
    if not HAS_DEPS:
        print(f"⚠ Dependencies not available: {import_error}")
        print("Skipping AMX MOE INT4_1K benchmark tests")
        return

    try:
        print("Running AMX MOE INT4_1K benchmark test...")
        test_moe_amx_int4_1k_benchmark()
        print("✓ AMX MOE INT4_1K benchmark test passed")
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
