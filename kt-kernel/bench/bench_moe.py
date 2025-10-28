import os
import sys
import time
import json
import subprocess
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import cpuinfer_ext
import torch
from tqdm import tqdm

# 测试参数设置
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
m_block = 1
group_min_len = 10
group_max_len = 1024
num_experts_per_tok = 8
# layer_num = 5  # 测试时不同的层数
# qlen = 1
# warm_up_iter = 100
# test_iter = 10000

layer_num = 1  # 测试时不同的层数
qlen = 1024
warm_up_iter = 100
test_iter = 10000
CPUINFER_PARAM = 304
# 初始化 CPUInfer（此处使用原始构造函数，可根据需要调整配置参数）
CPUInfer = cpuinfer_ext.CPUInfer(CPUINFER_PARAM)

# 获取脚本相关信息，用于生成结果保存文件名
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_name = os.path.splitext(os.path.basename(script_path))[0]
json_path = os.path.join(script_dir, "bench_results "+ ".jsonl")


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
    uname = platform.uname()
    info["system_name"] = uname.system
    info["node_name"] = uname.node

    # 获取 CPU 型号（仅 Linux 支持）
    cpu_model = None
    if os.path.exists('/proc/cpuinfo'):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            cpu_model = f"Error: {e}"
    info["cpu_model"] = cpu_model

    # 获取内存大小（单位：GB），仅 Linux 支持
    mem_total_gb = None
    if os.path.exists('/proc/meminfo'):
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = float(line.split(":", 1)[1].split()[0])
                        mem_total_gb = round(mem_kb / (1024 * 1024), 2)
                        break
        except Exception as e:
            mem_total_gb = f"Error: {e}"
    info["memory_size_GB"] = mem_total_gb

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
    info["cpu_socket_count"] = len(sockets) if len(sockets) > 0 else 1

    return info


def record_results(result, filename=json_path):
    """
    将结果以 JSON 格式追加到文件中
    """
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


def bench_moe(quant_mode: str):
    """
    依据不同量化模式进行 MoE 性能测试，包含预热与测试阶段
    """
    with torch.inference_mode():
        # 根据量化模式设置数据类型与 bytes_per_elem
        hidden_type = 30  # ggml_type::GGML_TYPE_BF16（固定）
        if quant_mode == "fp32":
            gate_type = 0    # ggml_type::GGML_TYPE_F32
            up_type = 0
            down_type = 0
            bytes_per_elem = 4.0
        elif quant_mode == "fp16":
            gate_type = 1    # ggml_type::GGML_TYPE_F16
            up_type = 1
            down_type = 1
            bytes_per_elem = 2.0
        elif quant_mode == "bf16":
            gate_type = 30   # ggml_type::GGML_TYPE_BF16
            up_type = 30
            down_type = 30
            bytes_per_elem = 2.0
        elif quant_mode == "q8_0":
            gate_type = 8    # ggml_type::GGML_TYPE_Q8_0
            up_type = 8
            down_type = 8
            bytes_per_elem = 1.062500
        elif quant_mode == "q6_k":
            gate_type = 14   # ggml_type::GGML_TYPE_Q6_K
            up_type = 14
            down_type = 14
            bytes_per_elem = 0.820312
        elif quant_mode == "q5_k_m":
            gate_type = 13   # ggml_type::GGML_TYPE_Q5_K
            up_type = 13
            down_type = 14   # ggml_type::GGML_TYPE_Q6_K
            bytes_per_elem = 0.731771
        elif quant_mode == "q4_k_m":
            gate_type = 12   # ggml_type::GGML_TYPE_Q4_K
            up_type = 12
            down_type = 14   # ggml_type::GGML_TYPE_Q6_K
            bytes_per_elem = 0.648437
        elif quant_mode == "q3_k_m":
            gate_type = 11   # ggml_type::GGML_TYPE_Q3_K
            up_type = 11
            down_type = 13   # ggml_type::GGML_TYPE_Q5_K
            bytes_per_elem = 0.515625
        elif quant_mode == "q2_k":
            gate_type = 10   # ggml_type::GGML_TYPE_Q2_K
            up_type = 10
            down_type = 11   # ggml_type::GGML_TYPE_Q3_K
            bytes_per_elem = 0.328125
        elif quant_mode == "iq3_xs":
            gate_type = 21   # ggml_type::GGML_TYPE_IQ3_S
            up_type = 21
            down_type = 21
            bytes_per_elem = 0.429688
        elif quant_mode == "iq2_xxs":
            gate_type = 16   # ggml_type::GGML_TYPE_IQ2_XXS
            up_type = 16
            down_type = 16
            bytes_per_elem = 0.257812
        else:
            raise ValueError("不支持的量化模式")

        # 构建各层 MoE 模型
        moes = []
        for _ in tqdm(range(layer_num), desc="Initializing MOEs"):
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float16, device="cpu").to("cpu").contiguous()
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float16, device="cpu").to("cpu").contiguous()
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float16, device="cpu").to("cpu").contiguous()
            
            config = cpuinfer_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
            config.pool = CPUInfer.backend_
            config.m_block = m_block 
            config.group_min_len = group_min_len
            config.group_max_len = group_max_len
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()
            config.gate_type = gate_type
            config.up_type = up_type
            config.down_type = down_type
            config.hidden_type = hidden_type

            moe = cpuinfer_ext.moe.MOE(config)
            CPUInfer.submit(moe.load_weights_task())
            CPUInfer.sync()
            moes.append(moe)
        
        # 生成输入数据
        print('Generating data...')
        # 专家路由索引与权重，每层一个
        gen_iter = 1000
        expert_ids = torch.rand(gen_iter * qlen , expert_num, device="cpu").argsort(dim=-1)[:, :num_experts_per_tok].reshape(gen_iter, qlen * num_experts_per_tok).contiguous()
        weights = torch.rand((gen_iter, qlen, num_experts_per_tok), dtype=torch.float32, device="cpu").contiguous() 
        input_tensor = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
        output_tensor = torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
        # 将 qlen 封装成 tensor，用于 forward 调用
        qlen_tensor = torch.tensor([qlen], dtype=torch.int32)

        # 预热阶段
        print('Warming up...')
        for i in tqdm(range(warm_up_iter), desc="Warm-up"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    qlen_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i%gen_iter].data_ptr(),
                    weights[i%gen_iter].data_ptr(),
                    input_tensor[i % layer_num].data_ptr(),
                    output_tensor[i % layer_num].data_ptr(),
                    False
                )
            )
            CPUInfer.sync()

        # 测试阶段
        print('Start testing...')
        start = time.perf_counter()
        for i in tqdm(range(test_iter), desc="Testing"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    qlen_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i%gen_iter].data_ptr(),
                    weights[i%gen_iter].data_ptr(),
                    input_tensor[i % layer_num].data_ptr(),
                    output_tensor[i % layer_num].data_ptr(),
                    False
                )
            )
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start

        # 计算性能指标
        time_per_iter_us = total_time / test_iter * 1e6
        bandwidth = hidden_size * intermediate_size * 3 * num_experts_per_tok * (1/8 * 256 * (1-(31/32)**qlen)) * bytes_per_elem * test_iter / total_time / 1e9  # 单位：GB/s
        flops = hidden_size * intermediate_size * qlen * 3 * num_experts_per_tok * 2 * test_iter / total_time / 1e12         # 单位：TFLOPS

        # 打印结果
        print('Quant mode:', quant_mode)
        print('Time(s):', total_time)
        print('Iteration:', test_iter)
        print('Time(us) per iteration:', time_per_iter_us)
        print('Bandwidth:', bandwidth, 'GB/s')
        print('TFLOPS:', flops)
        print('')

        # 整理测试结果
        result = {
            "test_name": os.path.basename(__file__),
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
                "m_block": m_block,
                "group_min_len": group_min_len,
                "group_max_len": group_max_len,
                "num_experts_per_tok": num_experts_per_tok,
                "layer_num": layer_num,
                "qlen": qlen,
                "warm_up_iter": warm_up_iter,
                "test_iter": test_iter,
                "CPUInfer_parameter": CPUINFER_PARAM 
            }
        }
        # 添加 git 与系统信息
        result.update(get_git_commit())
        result.update(get_system_info())
        # 将结果记录到 JSON 文件中
        record_results(result)


if __name__ == "__main__":
    # 根据需要选择量化模式，目前调用 q4_k_m 模式，对 layer_nums 列表中各层数进行测试
    bench_moe("q4_k_m")
    # 其他量化模式调用可以按需取消注释
    # bench_moe("fp32", layer_num)
    # bench_moe("fp16", layer_num)
    # bench_moe("bf16", layer_num)
    # bench_moe("q8_0")
    # bench_moe("q6_k", layer_num)
    # bench_moe("q5_k_m", layer_num)
    # bench_moe("q3_k_m", layer_num)
    # bench_moe("q2_k", layer_num)
    # bench_moe("iq3_xs", layer_num)
    # bench_moe("iq2_xxs", layer_num)