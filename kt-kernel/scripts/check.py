import os

# insert the path of the project
import sys

# sys.path.insert(0, "/home/azure/ktransformers")
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import re
from collections import defaultdict
import itertools
import os
import torch
import numpy as np

tensor_from_amx = [".mlp.experts."]  # todo: add keys in gguf that should be used in the final tensor


def safe_open_binary_to_tensor(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"没有权限读取文件: {file_path}")

    try:
        with open(file_path, "rb") as f:
            binary_data = f.read()

        np_array = np.frombuffer(binary_data, dtype=np.int8)

        tensor = torch.from_numpy(np_array)

        return tensor

    except Exception as e:
        raise IOError(f"file process error: {str(e)}")


def read_safetensor_keys_from_folder(folder_path) -> dict:
    """
    :param folder_path: folder path
    :return: key_to_file_map
    """
    # check if the folder path is exist
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"GGUF dir not found: {folder_path}")
    if os.path.isfile(folder_path):
        folder_path = os.path.dirname(folder_path)

    key_to_file_map = {}

    found_safetensor = False
    for root, dirs, files in os.walk(folder_path):
        # sort files
        files = sorted(files)
        for file in files:
            if file.endswith(".safetensors"):
                found_safetensor = True
                file_path = os.path.join(root, file)
                try:
                    with safe_open(file_path, framework="pt") as f:
                        for key in f.keys():
                            if "model.layers.61" in key:
                                # skip MTP layer
                                continue
                            # try:
                            #     if int(key.split('.')[2]) > 4:
                            #         continue
                            # except:
                            #     pass
                            key_to_file_map[key] = file_path
                except Exception as e:
                    print(f"Error reading Safetensor file {file_path}: {e}")

    if not found_safetensor:
        raise FileNotFoundError(f"No Safetensor files found in {folder_path}")

    return key_to_file_map


def read_amx_tensor_from_folder(folder_path, keys) -> dict:
    layer_list = [f"_layer_{i}" for i in range(3, 61)]
    numa_list = ["_numa_0", "_numa_1"]

    down_list = [f"INT4_down_{i}_quant_.kt" for i in range(256)]
    gate_list = [f"INT4_gate_{i}_quant_.kt" for i in range(256)]
    up_list = [f"INT4_up_{i}_quant_.kt" for i in range(256)]
    down_scale_list = [f"INT4_down_{i}_scale_.kt" for i in range(256)]
    gate_scale_list = [f"INT4_gate_{i}_scale_.kt" for i in range(256)]
    up_scale_list = [f"INT4_up_{i}_scale_.kt" for i in range(256)]
    target = ["ffn_up_exps", "ffn_down_exps", "ffn_gate_exps"]
    tensor_file_map = {}
    for key in keys:
        layer = int(key.split(".")[1])
        if layer < 3:
            continue
        layer_path = f"_layer_{layer}"
        # concatenate the path layer/numa/(down|gate|up)_(0-255)_3670016Byte_quant_.kt
        # store the path in the tensor_file_map
        # key = key+'.idx.weight'
        # scale_key = key+'.idx.scale'
        for numa_idx, numa in enumerate(numa_list):
            # TODO: 256 should be a variable
            for i in range(256):
                prefix_key = ".".join(key.split(".")[:-1])

                experts_key = prefix_key + f".{i}.numa.{numa_idx}.weight"
                scale_key = prefix_key + f".{i}.numa.{numa_idx}.scale"
                if "down" in experts_key:
                    tensor_file_map[experts_key] = os.path.join(folder_path, layer_path, numa, down_list[i])
                    tensor_file_map[scale_key] = os.path.join(folder_path, layer_path, numa, down_scale_list[i])
                elif "gate" in experts_key:
                    tensor_file_map[experts_key] = os.path.join(folder_path, layer_path, numa, gate_list[i])
                    tensor_file_map[scale_key] = os.path.join(folder_path, layer_path, numa, gate_scale_list[i])
                elif "up" in experts_key:
                    tensor_file_map[experts_key] = os.path.join(folder_path, layer_path, numa, up_list[i])
                    tensor_file_map[scale_key] = os.path.join(folder_path, layer_path, numa, up_scale_list[i])
    return tensor_file_map


# def translate_name(name:str)->str:
#     """
#     :param name: name of the tensor
#     :return: translated name
#     """
#     name = translate_name_to_gguf(name)
#     name = name.replace(".up_proj.", ".ffn_up_exps.")
#     name = name.replace(".down_proj.", ".ffn_down_exps.")
#     name = name.replace(".gate_proj.", ".ffn_gate_exps.")
#     name = name.replace(".ffn_gate_inp.e_score_correction_bias", ".exp_probs_b.bias")
#     return name


def _clean_keys(keys):
    keys = list(keys)
    target = ["ffn_up_exps", "ffn_down_exps", "ffn_gate_exps"]
    # only keep the keys that contain the target
    keys = [key for key in keys if any(target_key in key for target_key in target) and "ggml_type" not in key]
    return keys


def combine_tensor_sources(safetensor_path, amx_path):
    safetensor_tensor_file_map = read_safetensor_keys_from_folder(safetensor_path)

    keys = _clean_keys(safetensor_tensor_file_map.keys())

    amx_tensor_file_map = read_amx_tensor_from_folder(amx_path, keys)
    target_tensor_map = {}
    for key in safetensor_tensor_file_map.keys():
        if "_exps." in key:
            continue

        target_tensor_map[key] = safetensor_tensor_file_map[key]

    for key in amx_tensor_file_map.keys():
        target_tensor_map[key] = amx_tensor_file_map[key]

    return target_tensor_map


def write_combined_tensor(target_tensor_map: dict, output_path: str):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Cache for safetensor file handles and GGUF loaders
    safetensors_cache = {}
    amx_cache = {}

    # Group tensors by layer
    layer_groups = defaultdict(list)
    non_layer_keys = []
    layer_pattern = re.compile(r"blk\.(\d+)\.")

    for key in target_tensor_map:
        match = layer_pattern.search(key)
        if match:
            layer_groups[int(match.group(1))].append(key)
        else:
            non_layer_keys.append(key)

    # Calculate the number of shards
    total_shards = len(layer_groups) + (1 if non_layer_keys else 0) - 1

    shard_idx = 0
    # Save non-layer tensors to the first shard if they exist
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
            elif file_path.endswith(".kt"):
                tensor = safe_open_binary_to_tensor(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            tensors[key] = tensor

        output_file = os.path.join(output_path, f"model-{shard_idx:05}-of-{total_shards:05}.safetensors")
        print(f"Saving non-layer tensors to {output_file}")
        save_file(tensors, output_file)
        shard_idx += 1

    # Save each layer's tensors to subsequent shards
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
                tensor_info = tensor.shape
            elif file_path.endswith(".kt"):
                tensor = safe_open_binary_to_tensor(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            tensors[key] = tensor

        output_file = os.path.join(output_path, f"model-{shard_idx:05}-of-{total_shards:05}.safetensors")
        print(f"Saving layer {layer_num} to {output_file}")
        save_file(tensors, output_file)
        shard_idx += 1
    return


def main():
    # 输入已经处理过的混合模型路径，提前处理好的amx路径，输出路径
    parser = argparse.ArgumentParser(description="Read parameters from Safetensor and GGUF files")
    parser.add_argument(
        "--safetensor_path",
        type=str,
        help="Path to the Safetensor file",
        default="/mnt/data/models/DeepSeek-R1-GGML-FP8-Hybrid/DeepSeek-R1-IQ1S-FP8",
    )
    parser.add_argument(
        "--amx_path", type=str, help="Path to the GGUF file", default="/mnt/data/models/DeepSeek-R1-INT4"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output file",
        default="/mnt/data/models/DeepSeek-R1-GGML-FP8-Hybrid/DeepSeek-R1-AMXQ4-FP8",
    )

    # print all the arguments
    print("All the arguments:")
    print(parser.parse_args())

    # 解析命令行参数
    args = parser.parse_args()

    safetensor_path = args.safetensor_path
    amx_path = args.amx_path
    output_path = args.output_path

    target_tensor_map = combine_tensor_sources(safetensor_path, amx_path)
    for key, value in target_tensor_map.items():
        print(f"{key}: {value}")
    write_combined_tensor(target_tensor_map, output_path)

    return


if __name__ == "__main__":
    main()
