import struct
import warnings
import numpy as np
import re
import numpy.typing as npt
from typing import Sequence
import os
from enum import IntEnum
import torch
import KTransformersOps
from safetensors import safe_open
from ktransformers.ktransformers_ext.triton.fp8gemm import fp8_gemm, act_quant, weight_dequant
from safetensors.torch import save_file

class SafeTensorLoader:
    tensor_file_map = {}
    tensor_type_map = {}
    file_handle_map = {}
    
    def __init__(self, file_path: str):
        self.__load_tensor_file_map(file_path)

    def __load_tensor_file_map(self, file_path: str):
        # 处理传入路径，确保是文件夹路径
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")
        if os.path.isfile(file_path):
            folder_path = os.path.dirname(file_path)
        else:
            folder_path = file_path

        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework="pt")
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f"Error opening Safetensor file {file_path}: {e}")
                            continue

                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f"Error reading Safetensor file {file_path}: {e}")

        # if not found_safetensor:
        #     raise FileNotFoundError(f"No Safetensor files found in {folder_path}")

    def load_tensor(self, key: str, device: str="cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)

    def close_all_handles(self):
        for handle in self.file_handle_map.values():
            handle.close()
        self.file_handle_map.clear()

    def load_dequantized_tensor(self, key:str, device: str="cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key).to(device)
        if key.endswith(".weight"):
            if key[:-7] + ".weight_scale_inv" in self.tensor_file_map:
                weight_scale_inv = f.get_tensor(key[:-7] + ".weight_scale_inv").to(device)
                tensor = weight_dequant(tensor, weight_scale_inv)
        return tensor.to(device)