# this script targets to merge the fp8 safe tensor and the gguf quantized tensors.

import os
# insert the path of the project
import sys
# sys.path.insert(0, "/home/azure/ktransformers")
import argparse
import torch
from ktransformers.util.custom_loader import GGUFLoader, translate_name_to_gguf
from safetensors import safe_open
from safetensors.torch import save_file
import re
from collections import defaultdict

def read_safetensor_keys_from_folder(folder_path)->dict:
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

tensor_from_gguf = [] # todo: add keys in gguf that should be used in the final tensor

def translate_name(name:str)->str:
    """
    :param name: name of the tensor
    :return: translated name
    """
    name = translate_name_to_gguf(name)
    name = name.replace(".up_proj.", ".ffn_up_exps.")
    name = name.replace(".down_proj.", ".ffn_down_exps.")
    name = name.replace(".gate_proj.", ".ffn_gate_exps.")
    name = name.replace(".ffn_gate_inp.e_score_correction_bias", ".exp_probs_b.bias") 
    return name
    

def combine_tensor_sources(safetensor_path:str, gguf_path:str):
    gguf_loader = GGUFLoader(gguf_path)
    gguf_tensor_file_map = gguf_loader.tensor_file_map
    safetensor_tensor_file_map = read_safetensor_keys_from_folder(safetensor_path)
    
    # build a map for the key to the tensor
    # according to the key, we can get the tensor from the file
    
    target_tensor_map = {}
    for key in safetensor_tensor_file_map.keys():
        # for all experts, we use the gguf tensor
        if ".mlp.experts." in key:
            if '.weight_scale_inv' in key:
                continue
            key = '.'.join(key.split('.')[:5]+key.split('.')[-2:])
            translated_key = translate_name(key)
            target_tensor_map[key] = gguf_tensor_file_map[translated_key]
            continue
        
        if any(target_key in key for target_key in tensor_from_gguf):
            target_tensor_map[key] = gguf_tensor_file_map[translate_name(key)]
        else:
            target_tensor_map[key] = safetensor_tensor_file_map[key]
    
    return target_tensor_map, gguf_loader

def write_combined_tensor(target_tensor_map: dict, output_path: str, gguf_loader: GGUFLoader):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Cache for safetensor file handles and GGUF loaders
    safetensors_cache = {}
    gguf_cache = {}
    
    # Group tensors by layer
    layer_groups = defaultdict(list)
    non_layer_keys = []
    layer_pattern = re.compile(r'\.layers\.(\d+)\.')
    
    for key in target_tensor_map:
        match = layer_pattern.search(key)
        if match:
            layer_num = int(match.group(1))
            layer_groups[layer_num].append(key)
        else:
            non_layer_keys.append(key)
    
    # Calculate total shards
    total_shards = len(layer_groups) + (1 if non_layer_keys else 0) - 1
    if total_shards == 0:
        raise ValueError("No tensors to save")
    
    shard_idx = 0
    
    # Save non-layer tensors to the first shard if they exist
    if non_layer_keys:
        tensors = {}
        for key in non_layer_keys:
            file_path = target_tensor_map[key]
            tensor = None
            ggml_type = None
            if file_path.endswith('.safetensors'):
                if file_path not in safetensors_cache:
                    safetensors_cache[file_path] = safe_open(file_path, framework='pt')
                f = safetensors_cache[file_path]
                tensor = f.get_tensor(key)
            elif file_path.endswith('.gguf'):
                gguf_name = translate_name(key)
                tensor, ggml_type = gguf_loader.get_undequanted_tensor_and_ggml_type(gguf_name)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            tensors[translate_name(key)] = tensor
            if ggml_type:
                ggml_type = torch.tensor(ggml_type)
                ggml_key = translate_name(key)[:-7] + ".ggml_type" if translate_name(key).endswith(".weight") else translate_name(key) + ".ggml_type"
                tensors[ggml_key] = ggml_type
        
        output_file = os.path.join(output_path, f"model-{shard_idx:05}-of-{total_shards:05}.safetensors")
        print(f"Saving non-layer tensors to {output_file}")
        save_file(tensors, output_file)
        print(tensors.keys())

        shard_idx += 1
    
    # Save each layer's tensors to subsequent shards
    for layer_num in sorted(layer_groups.keys()):
        layer_keys = layer_groups[layer_num]
        tensors = {}
        for key in layer_keys:
            file_path = target_tensor_map[key]
            tensor = None
            ggml_type = None
            if file_path.endswith('.safetensors'):
                if file_path not in safetensors_cache:
                    safetensors_cache[file_path] = safe_open(file_path, framework='pt')
                f = safetensors_cache[file_path]
                tensor = f.get_tensor(key)
                tensor_info = tensor.shape
            elif file_path.endswith('.gguf'):
                gguf_name = translate_name(key)
                tensor, ggml_type = gguf_loader.get_undequanted_tensor_and_ggml_type(gguf_name)
                # tensor_info = gguf_loader.tensor_info[gguf_name]
                # ggml_type = gguf_loader.tensor_info[gguf_name]['ggml_type']
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            tensors[translate_name(key)] = tensor
            if ggml_type:
                ggml_type = torch.tensor(ggml_type)
                ggml_key = translate_name(key)[:-7] + ".ggml_type" if translate_name(key).endswith(".weight") else translate_name(key) + ".ggml_type"
                tensors[ggml_key] = ggml_type
        
        output_file = os.path.join(output_path, f"model-{shard_idx:05}-of-{total_shards:05}.safetensors")
        print(f"Saving layer {layer_num} to {output_file}")
        # print(tensors.keys())
        save_file(tensors, output_file)
        shard_idx += 1
    
    return
    
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Read parameters from Safetensor and GGUF files")
    parser.add_argument("--safetensor_path", type=str, help="Path to the Safetensor file", default="/mnt/data/model/DeepSeek-V3")
    parser.add_argument("--gguf_path", type=str, help="Path to the GGUF file", default="/mnt/data/model/DeepseekV3-q4km-gguf")
    parser.add_argument("--output_path", type=str, help="Path to the output file", default="/mnt/data/model/ktrans-safetensors/DeepSeek-V3-q4km-fp8")
    
    # print all the arguments
    print("All the arguments:")
    print(parser.parse_args())
    
    # 解析命令行参数
    args = parser.parse_args()

    safetensor_path = args.safetensor_path
    gguf_path = args.gguf_path
    output_path = args.output_path
    
    target_tensor_map, gguf_loader = combine_tensor_sources(safetensor_path, gguf_path)
    write_combined_tensor(target_tensor_map, output_path, gguf_loader)
    
    return

if __name__ == "__main__":
    main()