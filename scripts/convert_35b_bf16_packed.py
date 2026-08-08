#!/usr/bin/env python3
"""将 35B bf16 deepseek 格式转为 MESH 需要的 packed 格式。

输入: model.layers.{L}.mlp.experts.experts.{id}.{gate,up,down}_proj.weight
输出: model.layers.{L}.mlp.experts.gate_up_proj [E, 2*I, H]
      model.layers.{L}.mlp.experts.down_proj    [E, H, I]

用法: python3 convert_35b_bf16_packed.py
"""
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import glob

SRC = "/mnt/data2/models/Qwen3.5-35B-A3B-Unfused"
DST = "/mnt/data2/models/Qwen3.5-35B-A3B-BF16-PACKED"
NUM_LAYERS = 40
NUM_EXPERTS = 256
HIDDEN = 2048
INTER = 512

os.makedirs(DST, exist_ok=True)

# 收集所有 shard 文件
shards = sorted(glob.glob(os.path.join(SRC, "model.safetensors-*")))
print(f"Found {len(shards)} shards")

# 构建 tensor 索引：tensor_name -> shard_path
tensor_index = {}
for s in shards:
    f = safe_open(s, "pt")
    for k in f.keys():
        if "experts.experts" in k:
            tensor_index[k] = s
print(f"Indexed {len(tensor_index)} expert tensors")

for layer in range(NUM_LAYERS):
    # 收集该层所有专家的 gate/up/down
    gate_list = []
    up_list = []
    down_list = []
    for eid in range(NUM_EXPERTS):
        g_key = f"model.layers.{layer}.mlp.experts.experts.{eid}.gate_proj.weight"
        u_key = f"model.layers.{layer}.mlp.experts.experts.{eid}.up_proj.weight"
        d_key = f"model.layers.{layer}.mlp.experts.experts.{eid}.down_proj.weight"
        if g_key not in tensor_index:
            print(f"WARNING: missing {g_key}")
            continue
        f = safe_open(tensor_index[g_key], "pt")
        gate_list.append(f.get_tensor(g_key))  # [I, H]
        f = safe_open(tensor_index[u_key], "pt")
        up_list.append(f.get_tensor(u_key))    # [I, H]
        f = safe_open(tensor_index[d_key], "pt")
        down_list.append(f.get_tensor(d_key))  # [H, I]

    # gate_up: [E, 2*I, H] — gate 在前，up 在后
    gate_stack = torch.stack(gate_list)  # [E, I, H]
    up_stack = torch.stack(up_list)      # [E, I, H]
    gate_up = torch.cat([gate_stack, up_stack], dim=1)  # [E, 2*I, H]

    # down: [E, H, I]
    down = torch.stack(down_list)  # [E, H, I]

    # 保存
    out_file = os.path.join(DST, f"layer_{layer:02d}.safetensors")
    save_file({
        f"model.layers.{layer}.mlp.experts.gate_up_proj": gate_up.contiguous(),
        f"model.layers.{layer}.mlp.experts.down_proj": down.contiguous(),
    }, out_file)
    print(f"Layer {layer}: saved {gate_up.shape} + {down.shape} -> {out_file}")

    # 释放内存
    del gate_list, up_list, down_list, gate_stack, up_stack, gate_up, down

# 复制 config.json
import shutil
for fname in ["config.json", "generation_config.json"]:
    src_f = os.path.join(SRC, fname)
    if os.path.exists(src_f):
        shutil.copy2(src_f, os.path.join(DST, fname))
print(f"\nDone! Output: {DST}")
