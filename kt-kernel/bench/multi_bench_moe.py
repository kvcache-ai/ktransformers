#!/usr/bin/env python
# coding=utf-8
"""
自动展开 list 参数的 benchmark 脚本。
只要将所有测试参数放在 all_params 字典中，凡是值为 list 的键都会被自动展开，
生成参数组合后依次调用 bench_moe/bench_moe_amx 运行测试。
"""

import os
import sys
import itertools
from collections.abc import Sequence

# 将当前目录加入搜索路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

#####################################################################
# 1. 在此处一次性写好所有测试参数
#####################################################################
all_params = {
    # 固定参数
    "test_operator_type": "llamafile",   # "llamafile" 或 "amx" "kml"
    "expert_num": 256,
    "num_experts_per_tok": 8,
    "hidden_size": 7168,
    "intermediate_size": 2048,
    "max_len": 25600,         # amx 专用，llamafile 可保留不使用
    "group_max_len": 1024,    # llamafile 专用
    "group_min_len": 10,      # llamafile 专用
    "m_block": [256],             # llamafile 专用
     "qlen": range(1,11,1),
    "layer_num": 3,
    "warm_up_iter": 100,
    "test_iter": 10000,

    # ↓↓↓ 下面这些值是 list，会被自动展开 ↓↓↓
    "CPUINFER_PARAM": [304],
    # "CPUINFER_PARAM": [144], # Kunpeng 920 7280Z
    "quant_mode": "q4_k_m", # llamafile
    # "quant_mode": ["int4", "int8"], # amx
    # "quant_mode": "int8", # amx
}
#####################################################################


def expand_param_dict(param_dict):
    """对值为 list 的键做笛卡儿积展开"""
    vary_keys, vary_values, fixed_items = [], [], {}
    for k, v in param_dict.items():
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            vary_keys.append(k)
            vary_values.append(v)
        else:
            fixed_items[k] = v

    if not vary_keys:
        yield param_dict
        return

    for combo in itertools.product(*vary_values):
        params = fixed_items.copy()
        params.update(dict(zip(vary_keys, combo)))
        yield params


# 根据 operator 类型动态导入 bench 模块
if all_params["test_operator_type"] == "llamafile":
    import bench_moe as bench
elif all_params["test_operator_type"] == "amx":
    import bench_moe_amx as bench
elif all_params["test_operator_type"] == "kml":
    import bench_moe_kml as bench
else:
    raise ValueError(f"Unknown test_operator_type: {all_params['test_operator_type']}")


def update_bench_parameters(params):
    """同步参数到 bench 模块并重新初始化 CPUInfer"""
    bench.expert_num = params["expert_num"]
    bench.hidden_size = params["hidden_size"]
    bench.intermediate_size = params["intermediate_size"]
    bench.max_len = params["max_len"]
    bench.group_max_len = params["group_max_len"]
    bench.group_min_len = params["group_min_len"]
    bench.m_block = params["m_block"]
    bench.num_experts_per_tok = params["num_experts_per_tok"]
    bench.layer_num = params["layer_num"]
    bench.qlen = params["qlen"]
    bench.warm_up_iter = params["warm_up_iter"]
    bench.test_iter = params["test_iter"]
    bench.CPUINFER_PARAM = params["CPUINFER_PARAM"]
    # 重新初始化 CPUInfer 对象
    bench.CPUInfer = bench.cpuinfer_ext.CPUInfer(bench.CPUINFER_PARAM)


def main():
    for params in expand_param_dict(all_params):
        print("=" * 60)
        print("开始测试参数集:", params)
        update_bench_parameters(params)
        bench.bench_moe(params["quant_mode"])
        print("完成测试，量化模式:", params["quant_mode"])
        print("=" * 60, "\n")


if __name__ == "__main__":
    main()
