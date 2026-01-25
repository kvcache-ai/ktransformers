#!/usr/bin/env python
# coding=utf-8
"""
Description  :
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022
LastEditTime : 2024-08-06 10:41:28
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
"""
import os, sys, time, json, subprocess, platform

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import torch
from kt_kernel import kt_kernel_ext
import numpy as np

# 测试参数设置
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
layer_num = 5

qlen = 1
warm_up_iter = 1000
test_iter = 10000
physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()

# 将 CPUInfer 参数设为变量
# CPUINFER_PARAM = 257
# CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

worker_config = kt_kernel_ext.WorkerPoolConfig()
worker_config.subpool_count = 2
worker_config.subpool_numa_map = [0, 1]
worker_config.subpool_thread_count = [32, 32]
CPUINFER_PARAM = 64
CPUInfer = kt_kernel_ext.CPUInfer(worker_config)


def get_git_commit():
    """
    获取当前 git 提交记录（commit hash 和提交信息），
    并检查是否存在未提交的更改（dirty）
    """
    result = {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip()
        result["commit"] = commit
        result["commit_message"] = commit_msg

        # 检查是否存在未提交的更改
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
    """
    获取系统信息，包括系统名称、CPU 型号、内存大小（GB）、CPU 核数及 socket 数量
    """
    info = {}
    # 系统名称及主机名
    uname = platform.uname()
    info["system_name"] = uname.system  # 如 Linux, Windows 等
    info["node_name"] = uname.node  # 主机名称

    # 获取 CPU 型号（仅 Linux 支持）
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

    # 获取内存大小（单位：GB），仅 Linux 支持
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

    # 获取 CPU 核数（逻辑核数）
    info["cpu_core_count"] = os.cpu_count()

    # 解析 /proc/cpuinfo 获取 socket 数量
    sockets = set()
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "physical id" in line:
                        sockets.add(line.split(":", 1)[1].strip())
        except Exception as e:
            sockets = set()
    # 如果没有解析到 socket 信息，则默认至少有 1 个 socket
    info["cpu_socket_count"] = len(sockets) if len(sockets) > 0 else 1

    return info


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_name = os.path.splitext(os.path.basename(script_path))[0]
json_path = os.path.join(script_dir, script_name + ".jsonl")


def record_results(result, filename=json_path):
    """
    将结果以 JSON 格式追加到文件中
    """
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


def bench_moe(quant_mode: str):
    with torch.inference_mode():
        if quant_mode == "bf16":
            bytes_per_elem = 2.0
        elif quant_mode == "int8":
            bytes_per_elem = 1.0
        elif quant_mode == "int4":
            bytes_per_elem = 0.5
        else:
            raise ValueError("不支持的量化模式")

        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
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
            if quant_mode == "bf16":
                moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
            elif quant_mode == "int8":
                moe = kt_kernel_ext.moe.AMXInt8_MOE(config)
            elif quant_mode == "int4":
                moe = kt_kernel_ext.moe.AMXInt4_MOE(config)
            CPUInfer.submit(moe.load_weights_task())
            CPUInfer.sync()
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)
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

        # 预热迭代
        for i in tqdm(range(warm_up_iter), desc="Warm-up"):
            # start_it = time.time_ns()
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
            # end_it = time.time_ns()
            # print('python Time(ns): ', end_it - start_it)

        # 测试迭代
        start = time.perf_counter()
        for i in tqdm(range(test_iter), desc="Testing"):
            # print(f'test iteration {i}')
            # start_it = time.time_ns()
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
            # end_it = time.time_ns()
            # print('python Time(ns): ', end_it - start_it)
        end = time.perf_counter()
        total_time = end - start

        # 计算性能指标
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
        )  # 单位：GB/s
        flops = (
            hidden_size * intermediate_size * qlen * 3 * num_experts_per_tok * 2 * test_iter / total_time / 1e12
        )  # 单位：TFLOPS

        print("Quant mode: ", quant_mode)
        print("Time(s): ", total_time)
        print("Iteration: ", test_iter)
        print("Time(us) per iteration: ", time_per_iter_us)
        print("Bandwidth: ", bandwidth, "GB/s")
        print("Flops: ", flops, "TFLOPS")
        print("")

        # 整理结果记录，包括测试参数
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
                "CPUInfer_parameter": CPUINFER_PARAM,
            },
        }
        # 添加 git 提交记录信息
        result.update(get_git_commit())
        # 添加系统信息（包括 CPU 核数和 socket 数量）
        result.update(get_system_info())
        # 将结果以 JSON 形式追加到文件中
        record_results(result)


if __name__ == "__main__":
    # 选择需要测试的量化模式
    # bench_moe("bf16")
    bench_moe("int8")
    # bench_moe("int4")
