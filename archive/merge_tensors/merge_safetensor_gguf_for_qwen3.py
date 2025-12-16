# coding=utf-8
# Copyright (c) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import torch
from ktransformers.util.custom_loader import GGUFLoader, translate_name_to_gguf
from safetensors import safe_open
from safetensors.torch import save_file
import re
from collections import defaultdict

def read_safetensor_keys_from_folder(folder_path) -> dict:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Safetensors dir not found: {folder_path}")
    if os.path.isfile(folder_path):
        folder_path = os.path.dirname(folder_path)

    key_to_file_map = {}
    found_safetensor = False

    for root, dirs, files in os.walk(folder_path):
        files = sorted(files)
        for file in files:
            if not file.endswith(".safetensors"):
                continue
            found_safetensor = True
            file_path = os.path.join(root, file)
            try:
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        key_to_file_map[key] = file_path
            except Exception as e:
                print(f"Error reading Safetensor file {file_path}: {e}")

    if not found_safetensor:
        raise FileNotFoundError(f"No Safetensor files found in {folder_path}")

    return key_to_file_map


# 可选：如果你希望对某些非 MoE tensor 也用 GGUF，可以把关键子串填到下面这个列表里
tensor_from_gguf = []  # e.g. ["self_attn.q_proj.weight"]


def translate_name(name: str) -> str:
    name = translate_name_to_gguf(name)
    name = name.replace(".up_proj.", ".ffn_up_exps.")
    name = name.replace(".down_proj.", ".ffn_down_exps.")
    name = name.replace(".gate_proj.", ".ffn_gate_exps.")
    name = name.replace(".ffn_gate_inp.e_score_correction_bias", ".exp_probs_b.bias")
    return name


def combine_tensor_sources(safetensor_path: str, gguf_path: str):
    gguf_loader = GGUFLoader(gguf_path)
    gguf_tensor_file_map = gguf_loader.tensor_file_map
    safetensor_tensor_file_map = read_safetensor_keys_from_folder(safetensor_path)

    target_tensor_map = {}

    for key, st_file in safetensor_tensor_file_map.items():
        if ".mlp.experts." in key and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) < 8:
                raise ValueError(f"Unexpected MoE expert key format: {key}")
            norm_key = ".".join(parts[:5] + parts[-2:])

            gguf_name = translate_name(norm_key)
            if gguf_name not in gguf_tensor_file_map:
                raise KeyError(
                    f"[MoE] GGUF tensor not found for safetensors key {key} -> {gguf_name}"
                )
            target_tensor_map[norm_key] = gguf_tensor_file_map[gguf_name]
            continue
        if any(tag in key for tag in tensor_from_gguf):
            gguf_name = translate_name(key)
            if gguf_name not in gguf_tensor_file_map:
                raise KeyError(
                    f"[Non-MoE] GGUF tensor not found for safetensors key {key} -> {gguf_name}"
                )
            target_tensor_map[key] = gguf_tensor_file_map[gguf_name]
        else:
            target_tensor_map[key] = st_file

    return target_tensor_map, gguf_loader


def write_combined_tensor(target_tensor_map: dict, output_path: str, gguf_loader: GGUFLoader):
    os.makedirs(output_path, exist_ok=True)

    safetensors_cache = {}
    layer_groups = defaultdict(list)
    non_layer_keys = []
    layer_pattern = re.compile(r"\.layers\.(\d+)\.")

    for key in target_tensor_map:
        m = layer_pattern.search(key)
        if m:
            layer_num = int(m.group(1))
            layer_groups[layer_num].append(key)
        else:
            non_layer_keys.append(key)

    total_shards = len(layer_groups) + (1 if non_layer_keys else 0) - 1
    if total_shards <= 0:
        raise ValueError("No tensors to save")

    shard_idx = 0

    if non_layer_keys:
        tensors = {}
        for key in non_layer_keys:
            file_path = target_tensor_map[key]
            tensor = None
            ggml_type = None

            if file_path.endswith(".safetensors"):
                if file_path not in safetensors_cache:
                    safetensors_cache[file_path] = safe_open(file_path, framework="pt")
                f = safetensors_cache[file_path]
                tensor = f.get_tensor(key)
            elif file_path.endswith(".gguf"):
                gguf_name = translate_name(key)
                tensor, ggml_type = gguf_loader.get_undequanted_tensor_and_ggml_type(gguf_name)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            out_key = translate_name(key)
            tensors[out_key] = tensor
            if ggml_type is not None:
                ggml_type = torch.tensor(ggml_type)
                if out_key.endswith(".weight"):
                    ggml_key = out_key[:-7] + ".ggml_type"
                else:
                    ggml_key = out_key + ".ggml_type"
                tensors[ggml_key] = ggml_type

        output_file = os.path.join(
            output_path, f"model-{shard_idx:05}-of-{total_shards:05}.safetensors"
        )
        print(f"[WRITE] Saving non-layer tensors to {output_file}")
        save_file(tensors, output_file)
        shard_idx += 1

    for layer_num in sorted(layer_groups.keys()):
        layer_keys = layer_groups[layer_num]
        tensors = {}

        for key in layer_keys:
            file_path = target_tensor_map[key]
            tensor = None
            ggml_type = None

            if file_path.endswith(".safetensors"):
                if file_path not in safetensors_cache:
                    safetensors_cache[file_path] = safe_open(file_path, framework="pt")
                f = safetensors_cache[file_path]
                tensor = f.get_tensor(key)
            elif file_path.endswith(".gguf"):
                gguf_name = translate_name(key)
                tensor, ggml_type = gguf_loader.get_undequanted_tensor_and_ggml_type(gguf_name)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            out_key = translate_name(key)
            tensors[out_key] = tensor
            if ggml_type is not None:
                ggml_type = torch.tensor(ggml_type)
                if out_key.endswith(".weight"):
                    ggml_key = out_key[:-7] + ".ggml_type"
                else:
                    ggml_key = out_key + ".ggml_type"
                tensors[ggml_key] = ggml_type

        output_file = os.path.join(
            output_path, f"model-{shard_idx:05}-of-{total_shards:05}.safetensors"
        )
        print(f"[WRITE] Saving layer {layer_num} to {output_file}")
        save_file(tensors, output_file)
        shard_idx += 1


def main():
    parser = argparse.ArgumentParser(
        description="Merge FP8 safetensors and GGUF tensors for Qwen3-30B-A3B"
    )
    parser.add_argument(
        "--safetensor_path",
        type=str,
        help="Path to the FP8 Safetensor folder",
        default="/mnt/data/model/Qwen3-30B-A3B-FP8",
    )
    parser.add_argument(
        "--gguf_path",
        type=str,
        help="Path to the GGUF file or folder",
        default="/mnt/data/model/Qwen3-30B-A3B-GGUF",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output safetensors folder",
        default="/mnt/data/model/ktrans-safetensors/Qwen3-30B-A3B-q4km-fp8",
    )

    args = parser.parse_args()

    print("[ARGS]", args)

    safetensor_path = args.safetensor_path
    gguf_path = args.gguf_path
    output_path = args.output_path

    target_tensor_map, gguf_loader = combine_tensor_sources(safetensor_path, gguf_path)
    write_combined_tensor(target_tensor_map, output_path, gguf_loader)


if __name__ == "__main__":
    main()

