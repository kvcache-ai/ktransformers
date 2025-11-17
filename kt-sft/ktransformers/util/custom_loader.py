import struct
import warnings
import numpy as np
import re
import numpy.typing as npt
from typing import Sequence
import os
from enum import IntEnum
import torch
if not torch.xpu.is_available():
    import KTransformersOps
from safetensors import safe_open
from ktransformers.ktransformers_ext.triton.fp8gemm import fp8_gemm, act_quant, weight_dequant
from ktransformers.util.custom_gguf import *
from safetensors.torch import save_file
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    Defines the interface that all model loaders must implement.
    """
    tensor_file_map = {}
    @abstractmethod
    def has_tensor(cls, name: str):
        """
        Check if the tensor exists in the loader.
        
        Args:
            name: Name of the tensor to check
            
        Returns:
            bool: True if the tensor exists, False otherwise
        """
        pass

class SafeTensorLoader(ModelLoader):
    tensor_file_map: dict
    tensor_type_map: dict
    file_handle_map: dict
    tensor_device_map: dict
    
    def __init__(self, file_path: str):
        self.__load_tensor_file_map(file_path)

    def __load_tensor_file_map(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")
        if os.path.isfile(file_path):
            folder_path = os.path.dirname(file_path)
        else:
            folder_path = file_path
        self.file_handle_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

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
        if translate_name_to_gguf(key) in self.tensor_file_map:
            key = translate_name_to_gguf(key)
        elif key in self.tensor_file_map:
            pass
        else:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)

    def load_experts(self, key: str, device: str="cpu"):
        '''
        Load experts from safetensor
        key: the name of the experts
        device: the device to load the experts to
        return: dict, 
        {up: tensor, down: tensor, gate: tensor, up_type: int, down_type: int, gate_type: int}
        {xxx}_type: the type of the up tensor, corresponding to the ggml type
        '''
        if self.has_tensor(translate_name_to_gguf(key)+".ffn_gate_exps.weight"):
            # legacy branch for loading hybrid model
            base_key = translate_name_to_gguf(key)
            # Load experts from safetensor
            gate_key = f"{base_key}.ffn_gate_exps.weight"
            gate_type_key = f"{base_key}.ffn_gate_exps.ggml_type"
            up_key = f"{base_key}.ffn_up_exps.weight"
            up_type_key = f"{base_key}.ffn_up_exps.ggml_type"
            down_key = f"{base_key}.ffn_down_exps.weight"
            down_type_key = f"{base_key}.ffn_down_exps.ggml_type"
            gate_tensor = self.load_tensor(gate_key, device).numpy()
            up_tensor = self.load_tensor(up_key, device).numpy()
            down_tensor = self.load_tensor(down_key, device).numpy()
            gate_type = self.load_tensor(gate_type_key, device).item()
            up_type = self.load_tensor(up_type_key, device).item()
            down_type = self.load_tensor(down_type_key, device).item()

            return {
                "up": up_tensor,
                "gate": gate_tensor,
                "down": down_tensor,
                "up_type": up_type,
                "gate_type": gate_type,
                "down_type": down_type
            }

        else:
            # Load experts from safetensor
            base_key = key  # e.g. "model.layers.3.mlp.experts"
            experts_count = 0
            
            # First, count how many experts we have by checking for expert 0's up_proj
            while self.has_tensor(f"{base_key}.{experts_count}.up_proj.weight"):
                experts_count += 1
            
            if experts_count == 0:
                raise ValueError(f"No experts found for key {base_key}")
            
            # Initialize empty lists to store tensors for each projection type
            up_projs = []
            gate_projs = []
            down_projs = []
            
            # Load all expert weights
            for expert_id in range(experts_count):
                up_key = f"{base_key}.{expert_id}.up_proj.weight"
                gate_key = f"{base_key}.{expert_id}.gate_proj.weight"
                down_key = f"{base_key}.{expert_id}.down_proj.weight"
                
                up_tensor = self.load_tensor(up_key, device)
                gate_tensor = self.load_tensor(gate_key, device)
                down_tensor = self.load_tensor(down_key, device)
                
                up_projs.append(up_tensor)
                gate_projs.append(gate_tensor)
                down_projs.append(down_tensor)
            
            # Stack the tensors along a new dimension
            up_tensor = torch.stack(up_projs, dim=0)
            gate_tensor = torch.stack(gate_projs, dim=0)
            down_tensor = torch.stack(down_projs, dim=0)
            
            # Get original dtype for GGML type determination
            orig_up_dtype = up_tensor.dtype
            orig_gate_dtype = gate_tensor.dtype
            orig_down_dtype = down_tensor.dtype
            
            # Convert to numpy with proper bfloat16 support
            up_numpy = up_tensor.view(torch.uint16).numpy()
            gate_numpy = gate_tensor.view(torch.uint16).numpy()
            down_numpy = down_tensor.view(torch.uint16).numpy()
            
            # Determine tensor data types for GGML conversion
            def get_ggml_type(dtype):
                if dtype == torch.float32:
                    return GGMLQuantizationType.F32
                elif dtype == torch.float16:
                    return GGMLQuantizationType.F16
                elif dtype == torch.bfloat16:
                    return GGMLQuantizationType.BF16
                else:
                    raise ValueError(f"Unsupported tensor dtype: {dtype}")
            
            return {
                "up": up_numpy,
                "gate": gate_numpy,
                "down": down_numpy,
                "up_type": get_ggml_type(orig_up_dtype),
                "gate_type": get_ggml_type(orig_gate_dtype),
                "down_type": get_ggml_type(orig_down_dtype)
            }
                
    def load_gate(self, key: str, device: str="cpu"):
        '''
        Load gate from safetensor
        key: the name of the gate
        device: the device to load the gate to
        return: dict, 
        {'weight': tensor, 'e_score_correction_bias': tensor}
        '''
        target = ["weight", "e_score_correction_bias"]
        res = {'weight': None, 'e_score_correction_bias': None}
        if self.has_tensor(translate_name_to_gguf(key)+".ffn_gate_exps.weight"):
            # legacy branch for loading hybrid model
            base_key = key
            for k in target:
                translated_key = translate_name_to_gguf(f"{base_key}.{k}")
                if self.has_tensor(translated_key):
                    tensor = self.load_tensor(translated_key, device)
                    res[k] = tensor
        else:
            # Load gate from safetensor
            base_key = key
            for k in target:
                if self.has_tensor(f"{base_key}.{k}"):
                    tensor = self.load_tensor(f"{base_key}.{k}", device)
                    res[k] = tensor
        return res
    
    def close_all_handles(self):
        for handle in self.file_handle_map.values():
            handle.close()
        self.file_handle_map.clear()

    def load_dequantized_tensor(self, key:str, device: str="cpu"):
        if key in self.tensor_file_map and translate_name_to_gguf(key):
            pass
        elif translate_name_to_gguf(key) in self.tensor_file_map:
            key = translate_name_to_gguf(key)
        else:
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
    
    def has_tensor(self, name: str):
        return name in self.tensor_file_map or translate_name_to_gguf(name) in self.tensor_file_map

class GGUFLoader(ModelLoader):
    tensor_info: dict
    gguf_path: str
    tensor_file_map: dict # {tensor_name: tensor_file_path}
    gguf_file_meta: dict
    safetensor_loader: SafeTensorLoader
    def __init__(self, gguf_path: str):
        # Check dir exist
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF dir not found: {gguf_path}")
        if os.path.isfile(gguf_path):
            gguf_path = os.path.dirname(gguf_path)

        self.safetensor_loader = None
        
        self.tensor_info = {}
        self.gguf_path = gguf_path
        self.tensor_file_map = {}
        self.file_data_map = {}
        self.gguf_file_meta = {}
        self.tensor_device_map = {}

		# I know this is ugly, but I don't want to change the original code too much
        # TODO: merge gguf load and other loads.
        safetensor_loader = SafeTensorLoader(gguf_path)
        if safetensor_loader.tensor_file_map:
            self.safetensor_loader = safetensor_loader
            return
        # Walk through all the .gguf files in the directory
        found_gguf = False
        for root, dirs, files in os.walk(gguf_path):
            for file in files:
                if file.endswith(".gguf"):
                    found_gguf = True
                    file_name = os.path.join(root, file)
                    with open(file_name, "rb") as f:
                        self.load_gguf(f)
                        if file_name not in self.file_data_map:
                            self.file_data_map[file_name] = np.memmap(file_name, mode = 'r')
        if not found_gguf:
            raise FileNotFoundError(f"Cannot find any .gguf files in: {gguf_path}")
                            
    def load_gguf(self, f):
        f.seek(0)
        assert f.read(4) == b'GGUF'
        values = struct.unpack("<IQQ", f.read(4+8+8))
        version, n_tensors, n_kv = values
        if version != 3:
            warnings.warn(f"Version {version} has never been tested, might not work")

        info = {}
        for _ in range(n_kv):
            name = read_value(f, DATA_TYPES["string"])

            data_type = struct.unpack("<I", f.read(4))[0]

            info[name] = read_value(f, data_type)

        tensor_info = {}
        for _ in range(n_tensors):
            name = read_value(f, DATA_TYPES["string"])
            shape_len = read_value(f, DATA_TYPES["uint32"])
            shape = [read_value(f, DATA_TYPES["uint64"]) for _ in range(shape_len)]
            ggml_type = read_value(f, DATA_TYPES["uint32"])
            bad_offset = read_value(f, DATA_TYPES["uint64"])
            n_elems = int(math.prod(shape))
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            np_dims = tuple(reversed(shape))
        
            item_type: npt.DTypeLike
            if ggml_type == GGMLQuantizationType.F16:
                item_count = n_elems
                item_type = np.float16
            elif ggml_type == GGMLQuantizationType.F32:
                item_count = n_elems
                item_type = np.float32
            elif ggml_type == GGMLQuantizationType.F64:
                item_count = n_elems
                item_type = np.float64
            elif ggml_type == GGMLQuantizationType.I8:
                item_count = n_elems
                item_type = np.int8
            elif ggml_type == GGMLQuantizationType.I16:
                item_count = n_elems
                item_type = np.int16
            elif ggml_type == GGMLQuantizationType.I32:
                item_count = n_elems
                item_type = np.int32
            elif ggml_type == GGMLQuantizationType.I64:
                item_count = n_elems
                item_type = np.int64
            else:
                item_count = n_bytes
                item_type = np.uint8
                np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)

            tensor_info[name] = {
                "ggml_type": ggml_type,
                "shape": shape,
                "bad_offset": bad_offset,
                "item_type": item_type,
                "item_count": item_count,
                "np_dims": np_dims
            }

        start = f.tell()
        # Alignment is 32 by default.
        # https://github.com/ggerganov/ggml/blob/e1daebbf9d38d510ba456c4d50b4500a73ac2b14/docs/gguf.md?plain=1#L253
        alignment = info.get("general.alignment", 32)

        # Inconveniently, the offset defined in gguf files is relative to the
        # end of the header and is unaligned.
        # We need to compute the absolute file offset ourselves instead.
        for t in tensor_info.values():
            offset = start + t["bad_offset"]
            offset += (alignment - offset % alignment) % alignment
            t["offset"] = offset
            
        for name in tensor_info:
            self.tensor_file_map[name] = f.name
        self.tensor_info.update(tensor_info)
        self.gguf_file_meta.update(info)
    
    def get_mmap_tensor(self, name):
        name = translate_name_to_gguf(name)
        t = self.tensor_info[name]
        mmap_data = self.file_data_map[ self.tensor_file_map[name] ]

        offset = t["offset"]
        item_type = t["item_type"]
        item_count = t["item_count"]
        itemsize = int(np.empty([], dtype = item_type).itemsize)
        return mmap_data[offset : offset + itemsize * item_count]

    def get_undequanted_tensor_and_ggml_type(self, name):
        name = translate_name_to_gguf(name)
        t = self.tensor_info[name]
        data = self.get_mmap_tensor(name)
        ggml_type = t["ggml_type"]
        data = torch.from_numpy(data)
        return data, ggml_type

    def load_expert_tensor(self, name, data, expert_id, elements_per_expert, device = "cuda", target_dtype = torch.get_default_dtype())->torch.Tensor:
        name = translate_name_to_gguf(name)
        t = self.tensor_info[name]
        shape = t["shape"]
        ggml_type = t["ggml_type"]
        if ggml_type not in GGML_NAMES:
            raise NotImplementedError(f"ggml_type {ggml_type} not implemented")
        ggml_name = GGML_NAMES[ggml_type]

        # TODO: experts may fused in quant block, split it
        assert elements_per_expert % GGML_ELEMENTS_PER_BLOCK[ggml_name] == 0, "experts may fused in quant block, please use CPU dequant"

        blocks_per_experts = elements_per_expert // GGML_ELEMENTS_PER_BLOCK[ggml_name]
        block_size = GGML_BLOCK_SIZES[ggml_name]
        offset = expert_id * block_size * blocks_per_experts
        data = data[offset: offset + block_size * blocks_per_experts]

        if "cuda" in device.lower():
            values = GGML_DEQUANTIZE_GPU[ggml_name](data, device, target_dtype)
        else:
            values = GGML_DEQUANTIZE[ggml_name](data)
            values = torch.from_numpy(values.copy())

        if ggml_name == "BF16":
            values = values.view(torch.bfloat16)
        values = values.view(shape[-2::-1])

        return values

    def load_gguf_tensor(self, name: str, device:str = "cpu", target_dtype = None)->torch.Tensor:
        name = translate_name_to_gguf(name)
        t = self.tensor_info[name]
        if target_dtype == None:
            target_dtype = torch.get_default_dtype()
        
        shape = t["shape"]
        ggml_type = t["ggml_type"]

        if ggml_type not in GGML_NAMES:
            raise NotImplementedError(f"ggml_type {ggml_type} not implemented")

        ggml_name = GGML_NAMES[ggml_type]

        data = self.get_mmap_tensor(name)

        block_size = GGML_BLOCK_SIZES[ggml_name]
        elements_per_block = GGML_ELEMENTS_PER_BLOCK[ggml_name]
        num_elements = int(np.prod(shape))
        num_blocks = num_elements // elements_per_block
        
        blocks_per_iter = 16384
        if num_blocks > blocks_per_iter: # dequant large tensor
            values = torch.empty((num_blocks, elements_per_block), dtype=target_dtype, device=device)
            for i in range( (num_blocks + blocks_per_iter - 1) // blocks_per_iter):
                blocks_begin = i * blocks_per_iter
                blocks_end = min(blocks_begin + blocks_per_iter, num_blocks)
                if "cuda" in device.lower():
                    try:
                        cur_values = GGML_DEQUANTIZE_GPU[ggml_name](data[blocks_begin*block_size : blocks_end*block_size], device, target_dtype)
                    except:
                        cur_values = GGML_DEQUANTIZE[ggml_name](data[blocks_begin*block_size : blocks_end*block_size])
                        cur_values = torch.from_numpy(cur_values.copy()).to(device)
                else:
                    cur_values = GGML_DEQUANTIZE[ggml_name](data[blocks_begin*block_size : blocks_end*block_size])
                    cur_values = torch.from_numpy(cur_values.copy())
                
                cur_values = cur_values.view(-1, elements_per_block)
                if ggml_name == "BF16":
                    cur_values = cur_values.view(torch.bfloat16)
                values[blocks_begin : blocks_end] = cur_values
        else:
            if "cuda" in device.lower():
                values = GGML_DEQUANTIZE_GPU[ggml_name](data, device)
            else:
                np_values = np.copy(GGML_DEQUANTIZE[ggml_name](data))
                values = torch.from_numpy(np_values).to(device)
                del np_values

        if ggml_name == "BF16":
            values = values.view(torch.bfloat16)
            

        values = values.view(shape[::-1])
        if "attn_q" in name and self.gguf_file_meta['general.architecture'] in ["llama"]:
            n_head = self.gguf_file_meta['llama.attention.head_count']
            values = (values.reshape(n_head, values.shape[0] // n_head // 2, 2, *values.shape[1:])
            .swapaxes(1, 2)
            .reshape(values.shape))
        elif "attn_k" in name and self.gguf_file_meta['general.architecture'] in ["llama"]:
            n_head = self.gguf_file_meta['llama.attention.head_count_kv'] 
            values = (values.reshape(n_head, values.shape[0] // n_head // 2, 2, *values.shape[1:])
            .swapaxes(1, 2)
            .reshape(values.shape))
        return values
    def has_tensor(self, name: str):
        name = translate_name_to_gguf(name)
        return name in self.tensor_info

    def get_ggml_type(self, name: str):
        name = translate_name_to_gguf(name)
        if name not in self.tensor_info:
            raise KeyError(f"Key {name} not found in GGUF files")
        return self.tensor_info[name]["ggml_type"]
    
class ModelLoaderFactory:
    """
    Factory class for creating model loaders.
    Automatically detects the model format based on file extensions in the directory.
    """
    
    @staticmethod
    def create_loader(path: str):
        """
        Create a model loader for the given path by detecting the model format.
        The function checks for the presence of .safetensors or .gguf files
        in the specified path and creates the appropriate loader.
        
        Args:
            path: Path to the model directory or file
            
        Returns:
            An appropriate ModelLoader instance (SafeTensorLoader or GGUFLoader)
        
        Raises:
            FileNotFoundError: If no supported model files are found in the path
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
            
        # Normalize to directory path if a file was provided
        if os.path.isfile(path):
            if path.endswith(".safetensors"):
                return SafeTensorLoader(path)
            elif path.endswith(".gguf"):
                return GGUFLoader(path)
            else:
                folder_path = os.path.dirname(path)
        else:
            folder_path = path
            
        # Check for safetensors files
        has_safetensors = False
        has_gguf = False
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".safetensors"):
                    has_safetensors = True
                    break
                elif file.endswith(".gguf"):
                    has_gguf = True
                    break
            if has_safetensors or has_gguf:
                break
                
        # Create the appropriate loader based on detected file types
        # Prioritize SafeTensor over GGUF if both are present
        if has_safetensors:
            try:
                return SafeTensorLoader(folder_path)
            except Exception as e:
                print(f"Failed to create SafeTensorLoader: {e}")
                # Fall through to try GGUF if SafeTensor fails
                if not has_gguf:
                    raise
        
        if has_gguf:
            try:
                return GGUFLoader(folder_path)
            except Exception as e:
                print(f"Failed to create GGUFLoader: {e}")
                raise
        
        # No supported model files found
        raise FileNotFoundError(f"No .safetensors or .gguf files found in: {folder_path}")