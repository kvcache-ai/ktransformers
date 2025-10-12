import os, sys
import time
import subprocess
import platform
import json
os.environ["BLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import cpuinfer_ext
from cpuinfer_ext.kvcache import ggml_type
import torch
from torch import inf, nn
from torch.nn import init

from tqdm import tqdm

qlen = 4096
kvlen = 0
page_table = list(range(20))
page_size = 256
pages_count = 200


hidden_size = 7168
num_heads = 128
kv_lora_rank = 512
q_lora_rank = 512
nope_size = 128
rope_size = 64
page_size = 512
layer_num = 10


rope_theta = 10000
max_qlen = qlen+kvlen
max_kvlen = 4096
max_position_embeddings =  163840

rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
}

CPUINFER_PARAM = 304
# 初始化 CPUInfer（此处使用原始构造函数，可根据需要调整配置参数）
CPUInfer = cpuinfer_ext.CPUInfer(CPUINFER_PARAM)


warm_up_iter = 20
test_iter = 100




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

def bench_mla(quant_mode: str):
    """
    测试 MLA 模型的性能
    """
    with torch.inference_mode():
        # 这里可以添加 MLA 模型的具体实现和测试代码
        hidden_type = 1  # ggml_type::GGML_TYPE_FP16（固定）
        if quant_mode == "fp32":
            q_a_proj_type = 0  # ggml_type::GGML_TYPE_F32
            q_b_proj_type = 0
            kv_a_proj_with_mqa_type = 0
            kv_b_proj_type = 0
            w_o_type = 0
            bytes_per_elem = 4.000000
        elif quant_mode == "fp16":
            q_a_proj_type = 1  # ggml_type::GGML_TYPE_F32
            q_b_proj_type = 1
            kv_a_proj_with_mqa_type = 1
            kv_b_proj_type = 1
            w_o_type = 1
            bytes_per_elem = 2.000000
        elif quant_mode == "q4_k_m":
            q_a_proj_type = 12   # ggml_type::GGML_TYPE_Q4_K
            q_b_proj_type = 12
            kv_a_proj_with_mqa_type = 12   # ggml_type::GGML_TYPE_Q6_K
            kv_b_proj_type = 12
            w_o_type = 12
            bytes_per_elem = 0.5625
        else:
            raise ValueError("不支持的量化模式")
    
    # 构建各层 MLA 模型的输入数据
        mlas = []
        for i in tqdm(range(layer_num)):
            q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False, dtype=torch.float16)
            q_b_proj = nn.Linear(q_lora_rank, num_heads * (nope_size+rope_size) , bias=False, dtype=torch.float16)
            kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + rope_size, bias=False, dtype=torch.float16)
            kv_b_proj = nn.Linear( num_heads * (nope_size + nope_size),kv_lora_rank, bias=False, dtype=torch.float16)
            o_proj = nn.Linear(num_heads * nope_size, hidden_size, bias=False, dtype=torch.float16)

            init.normal_(q_a_proj.weight, mean=0.0, std=0.02)
            init.normal_(q_b_proj.weight, mean=0.0, std=0.02)
            init.normal_(kv_a_proj_with_mqa.weight, mean=0.0, std=0.02)
            init.normal_(kv_b_proj.weight, mean=0.0, std=0.02)
            init.normal_(o_proj.weight, mean=0.0, std=0.02)
            q_a_proj_weight = q_a_proj.weight.to(torch.float16).to('cpu').contiguous()
            q_b_proj_weight = q_b_proj.weight.to(torch.float16).to('cpu').contiguous()
            kv_a_proj_with_mqa_weight = kv_a_proj_with_mqa.weight.to('cpu').to(torch.float16).contiguous()
            kv_b_proj_weight = kv_b_proj.weight.to(torch.float16).to('cpu').contiguous()
            o_proj_weight = o_proj.weight.to(torch.float16).to('cpu').contiguous()

            config = cpuinfer_ext.mla.MLAConfig(
                hidden_size,
                q_lora_rank,
                kv_lora_rank,
                num_heads,
                nope_size,
                rope_size,
            )
            config.max_qlen = max_qlen
            config.max_kvlen = max_kvlen
            config.max_position_embeddings = max_position_embeddings 
            config.rope_scaling_factor = rope_scaling["factor"]
            config.rope_theta = rope_theta
            config.rope_scaling_beta_fast = rope_scaling["beta_fast"]
            config.rope_scaling_beta_slow = rope_scaling["beta_slow"]
            config.rope_scaling_mscale = rope_scaling["mscale"]
            config.rope_scaling_mscale_all_dim = rope_scaling["mscale_all_dim"]
            config.rope_scaling_original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]

            config.q_a_proj = q_a_proj_weight.data_ptr()
            config.q_b_proj = q_b_proj_weight.data_ptr()
            config.kv_a_proj_with_mqa = kv_a_proj_with_mqa_weight.data_ptr()
            config.kv_b_proj = kv_b_proj_weight.data_ptr()
            config.o_proj = o_proj_weight.data_ptr()

            config.q_a_proj_type = ggml_type.FP16
            config.q_b_proj_type = ggml_type.FP16
            config.kv_a_proj_with_mqa_type = ggml_type.FP16
            config.kv_b_proj_type = ggml_type.FP16
            config.w_o_type = ggml_type.FP16


            config.pool = CPUInfer.backend_



            mla = cpuinfer_ext.mla.MLA(config)
            mla.load_weights()
            mla.set_local_pages(pages_count)
            mlas.append(mla)

        print('Generating data...')
        input_tensor = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cpu").to("cpu").contiguous()
        output_tensor = torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cpu").to("cpu").contiguous()
        
        print('Warming up...')

        for i in tqdm(range(warm_up_iter)):
            mlas[i%layer_num].forward([qlen],[page_table],[kvlen],
                        input_tensor[i%layer_num].data_ptr(),output_tensor[i%layer_num].data_ptr())


        print('Start testing...')

        start = time.perf_counter()
        for i in tqdm(range(test_iter)):
            mlas[i%layer_num].forward([qlen],[page_table],[kvlen],
                        input_tensor[i%layer_num].data_ptr(),output_tensor[i%layer_num].data_ptr())

        end = time.perf_counter()
        total_time = end - start

        time_per_iter_us = (total_time * 1e6) / test_iter
        bandwidth = bytes_per_elem * (q_lora_rank * hidden_size 
                     + (kv_lora_rank+rope_size) * hidden_size 
                     + (nope_size+rope_size) * q_lora_rank * num_heads
                     + (nope_size+nope_size)*kv_lora_rank * num_heads
                     + hidden_size * nope_size * num_heads
                     + hidden_size * qlen) * test_iter / (total_time * 1e9)
        flops =  2*(
                    q_lora_rank*hidden_size*qlen 
                    + kv_lora_rank * hidden_size * qlen
                    +num_heads* (nope_size+rope_size)*q_lora_rank*qlen 
                    + num_heads * qlen * nope_size * kv_lora_rank
                    + num_heads * (kvlen+qlen) * kv_lora_rank * qlen
                    + num_heads * rope_size * qlen * (qlen+kvlen)
                    + num_heads * kv_lora_rank * (qlen + kvlen) * qlen
                    + num_heads * nope_size * kv_lora_rank * qlen
                    + hidden_size * num_heads* nope_size * qlen
                    ) * test_iter / (total_time * 1e12)


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
                 "qlen": qlen,
                "kvlen": kvlen,
                "page_table": page_table,
                "page_size": page_size,
                "pages_count": pages_count,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "kv_lora_rank": kv_lora_rank,
                "q_lora_rank": q_lora_rank,
                "nope_size": nope_size,
                "rope_size": rope_size,


                "layer_num": layer_num,
               
                "rope_theta": rope_theta,
                "max_qlen": max_qlen,
                "max_kvlen": max_kvlen,
                "max_position_embeddings": max_position_embeddings,

                "rope_scaling": rope_scaling,

                "warm_up_iter": warm_up_iter,
                "test_iter": test_iter,
                "CPUInfer_parameter": CPUINFER_PARAM 
            }
        }
        # 添加 git 与系统信息
        result.update(get_git_commit())
        result.update(get_system_info())
        # 将结果记录到 JSON 文件中
        print(result)
        record_results(result)
        

bench_mla("fp16")