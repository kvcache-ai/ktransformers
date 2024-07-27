#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang, chenht2022
Date         : 2024-07-26 08:48:54
Version      : 1.0.0
LastEditors  : Azure 
LastEditTime : 2024-07-26 09:28:25
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
# copied from llama.cpp/gguf-py/gguf/constants.py to satisfy dependence of gguf
# GGUF specification
# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
import struct
import warnings
import numpy as np
import numpy.typing as npt
from typing import Sequence
import os
from enum import IntEnum
import torch
import KTransformersOps

class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30

QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
}

# copied from llama.cpp/gguf-py/gguf/quants.py to avoid dependence of gguf
def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)

GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
}

GGML_NAMES = {ggml_type: name for name, ggml_type in GGML_TYPES.items()}

GGML_BLOCK_SIZES = {
    "F32": 4,
    "F16": 2,
    "Q8_0": 2 + 32,
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q4_K": 2 + 2 + 12 + 256 // 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
    "Q6_K": 256 // 2 + 256 // 4 + 256 // 16 + 2,
}

GGML_ELEMENTS_PER_BLOCK = {
    "F32": 1,
    "F16": 1,
    "Q8_0": 32,
    "Q2_K": 256,
    "Q3_K": 256,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
}

# DATA_TYPES = {
#     "uint32": 4,
#     "int32": 5,
#     "float32": 6,
#     "string": 8,
#     "array": 9,
#     "uint64": 10,
# }
DATA_TYPES = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
}

class GGUFLoader:
    tensor_info: dict
    gguf_path: str
    tensor_file_map: dict # {tensor_name: tensor_file_path}
    gguf_file_meta: dict
    def __init__(self, gguf_path: str):
        # Check dir exist
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF dir not found: {gguf_path}")
        
        self.tensor_info = {}
        self.gguf_path = gguf_path
        self.tensor_file_map = {}
        self.file_data_map = {}
        self.gguf_file_meta = {}
        
        # Walk through all the .gguf files in the directory
        for root, dirs, files in os.walk(gguf_path):
            for file in files:
                if file.endswith(".gguf"):
                    file_name = os.path.join(root, file)
                    with open(file_name, "rb") as f:
                        self.load_gguf(f)
                        if file_name not in self.file_data_map:
                            self.file_data_map[file_name] = np.memmap(file_name, mode = 'r')
                            
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
            n_elems = int(np.prod(shape))
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
        t = self.tensor_info[name]
        mmap_data = self.file_data_map[ self.tensor_file_map[name] ]

        offset = t["offset"]
        item_type = t["item_type"]
        item_count = t["item_count"]
        itemsize = int(np.empty([], dtype = item_type).itemsize)
        return mmap_data[offset : offset + itemsize * item_count]
    
    def load_gguf_tensor(self, name: str, device:str = "cpu")->torch.Tensor:
        t = self.tensor_info[name]

        shape = t["shape"]
        ggml_type = t["ggml_type"]

        if ggml_type not in GGML_NAMES:
            raise NotImplementedError(f"ggml_type {ggml_type} not implemented")

        ggml_name = GGML_NAMES[ggml_type]

        data = self.get_mmap_tensor(name)
        

        if "cuda" in device.lower():
            values = GGML_DEQUANTIZE_GPU[ggml_name](data, device)
        else:
            values = GGML_DEQUANTIZE[ggml_name](data)
            values = torch.from_numpy(values)
        
        return values.view(shape[::-1])

def read_value(f, data_type):
    if data_type == DATA_TYPES["string"]:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    elif data_type == DATA_TYPES["bool"]:
        return bool(struct.unpack("<?", f.read(1))[0])

    elif data_type == DATA_TYPES["uint8"]:
        return struct.unpack("<B", f.read(1))[0]

    elif data_type == DATA_TYPES["int8"]:
        return struct.unpack("<b", f.read(1))[0]

    elif data_type == DATA_TYPES["uint16"]:
        return struct.unpack("<H", f.read(2))[0]

    elif data_type == DATA_TYPES["int16"]:
        return struct.unpack("<h", f.read(2))[0]

    elif data_type == DATA_TYPES["uint32"]:
        return struct.unpack("<I", f.read(4))[0]

    elif data_type == DATA_TYPES["int32"]:
        return struct.unpack("<i", f.read(4))[0]

    elif data_type == DATA_TYPES["float32"]:
        return struct.unpack("<f", f.read(4))[0]

    elif data_type == DATA_TYPES["uint64"]:
        return struct.unpack("<Q", f.read(8))[0]

    elif data_type == DATA_TYPES["int64"]:
        return struct.unpack("<q", f.read(8))[0]

    elif data_type == DATA_TYPES["float64"]:
        return struct.unpack("<d", f.read(8))[0]

    elif data_type == DATA_TYPES["array"]:
        elem_type, count = struct.unpack("<IQ", f.read(4 + 8))
        return [read_value(f, elem_type) for _ in range(count)]

    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def dequantize_q2_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    block_size = GGML_BLOCK_SIZES["Q2_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack([
        qs[:, 00:16] >> 0,
        qs[:, 16:32] >> 0,
        qs[:, 00:16] >> 2,
        qs[:, 16:32] >> 2,
        qs[:, 00:16] >> 4,
        qs[:, 16:32] >> 4,
        qs[:, 00:16] >> 6,
        qs[:, 16:32] >> 6,
        qs[:, 32:48] >> 0,
        qs[:, 48:64] >> 0,
        qs[:, 32:48] >> 2,
        qs[:, 48:64] >> 2,
        qs[:, 32:48] >> 4,
        qs[:, 48:64] >> 4,
        qs[:, 32:48] >> 6,
        qs[:, 48:64] >> 6,
    ], axis=1)

    return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)

def dequantize_q2_k_gpu(data):
    pass

def dequantize_q3_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    block_size = GGML_BLOCK_SIZES["Q3_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32:32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96: 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    return d * (scales - 32) * np.stack([
        (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
        (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
        (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
        (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
        (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
        (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
        (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
        (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
        (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
        (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
        (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
        (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
        (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
        (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
        (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
        (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7])
    ], axis=1)

def dequantize_q3_k_gpu(data):
    pass

def dequantize_q4_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)

    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
    offsets = scale_offsets * np.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)

    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    return factors * qs2 - offsets

def dequantize_q4_k_gpu(data, device:str ="cuda"):
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    data = torch.from_numpy(data)
    return KTransformersOps.dequantize_q4_k(data, 144, device)

def dequantize_q5_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    block_size = GGML_BLOCK_SIZES["Q5_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16: 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48: 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    m1 = dmin * scales_lo_6[:, 4]
    m2 = dmin * scales_lo_6[:, 5]
    m3 = dmin * scales_lo_6[:, 6]
    m4 = dmin * scales_lo_6[:, 7]
    m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    d1 = d * scales_lo_6[:, 0]
    d2 = d * scales_lo_6[:, 1]
    d3 = d * scales_lo_6[:, 2]
    d4 = d * scales_lo_6[:, 3]
    d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    return np.concatenate([
        d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
        d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
        d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
        d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
        d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
        d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
        d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
        d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
    ], axis=1)

def dequantize_q5_k_gpu(data):
    pass


def dequantize_q6_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)
    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:,   :32 ] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64 ] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:,   :32 ] >>  4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64 ] >>  4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96 ] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96 ] >>  4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >>  4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    return scales * np.concatenate([
        sc[:,  0] * q1[:, :16],
        sc[:,  1] * q1[:, 16:],
        sc[:,  2] * q2[:, :16],
        sc[:,  3] * q2[:, 16:],
        sc[:,  4] * q3[:, :16],
        sc[:,  5] * q3[:, 16:],
        sc[:,  6] * q4[:, :16],
        sc[:,  7] * q4[:, 16:],
        sc[:,  8] * q5[:, :16],
        sc[:,  9] * q5[:, 16:],
        sc[:, 10] * q6[:, :16],
        sc[:, 11] * q6[:, 16:],
        sc[:, 12] * q7[:, :16],
        sc[:, 13] * q7[:, 16:],
        sc[:, 14] * q8[:, :16],
        sc[:, 15] * q8[:, 16:],
    ], axis=1) 

# @torch.jit.script
def dequantize_q6_k_gpu(data: np.ndarray, device:str = "cuda"):
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    device = torch.device(device)
    num_blocks = len(data) // block_size
    data = np.frombuffer(data, dtype=data.dtype)
    data = torch.from_numpy(data)
    return KTransformersOps.dequantize_q6_k(data, 210, device)

def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q8_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]
    return scales * qs

def dequantize_q8_0_gpu(data, device:str = "cuda"):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q8_0"]
    device = torch.device(device)
    data = np.frombuffer(data, dtype=data.dtype)
    data = torch.from_numpy(data)
    return KTransformersOps.dequantize_q8_0(data, 34, device)


def dequantize_f32(data):
    return np.frombuffer(data, dtype=np.float32)

def dequantize_f32_gpu(data, device):
    data = np.frombuffer(data, dtype=np.float32)
    res = torch.from_numpy(data)
    res_gpu = torch.empty_like(res, device=device)
    res_gpu.copy_(res)
    return res_gpu

def dequantize_f16(data):
    return np.frombuffer(data, dtype=np.float16)

def dequantize_f16_gpu(data, device):
    data = np.frombuffer(data, dtype=np.float16)
    res = torch.from_numpy(data)
    res_gpu = torch.empty_like(res, device=device)
    res_gpu.copy_(res)
    return res

GGML_DEQUANTIZE = {
    "F32": dequantize_f32,
    "F16": dequantize_f16,
    "Q8_0": dequantize_q8_0,
    "Q2_K": dequantize_q2_k,
    "Q3_K": dequantize_q3_k,
    "Q4_K": dequantize_q4_k,
    "Q5_K": dequantize_q5_k,
    "Q6_K": dequantize_q6_k,
}

GGML_DEQUANTIZE_GPU = {
    "F32": dequantize_f32_gpu,
    "F16": dequantize_f16_gpu,
    "Q8_0": dequantize_q8_0_gpu,
    "Q2_K": dequantize_q2_k_gpu,
    "Q3_K": dequantize_q3_k_gpu,
    "Q4_K": dequantize_q4_k_gpu,
    "Q5_K": dequantize_q5_k_gpu,
    "Q6_K": dequantize_q6_k_gpu,
}

def translate_name_to_gguf(name):
    name = name.replace("lm_head.", "output.")
    name = name.replace("model.embed_tokens.", "token_embd.")
    name = name.replace("model.norm.", "output_norm.")
    
    name = name.replace("model.layers.", "blk.")
    name = name.replace(".input_layernorm", ".attn_norm")
    name = name.replace(".mlp.down_proj", ".ffn_down")
    name = name.replace(".mlp.gate_proj", ".ffn_gate")
    name = name.replace(".mlp.up_proj", ".ffn_up")
    name = name.replace(".post_attention_layernorm", ".ffn_norm")
    name = name.replace(".self_attn.q_proj", ".attn_q")
    name = name.replace(".self_attn.k_proj", ".attn_k")
    name = name.replace(".self_attn.v_proj", ".attn_v")
    name = name.replace(".self_attn.o_proj", ".attn_output")
    name = name.replace(".self_attn.qkv_proj", ".attn_qkv")
    name = name.replace(".self_attn.kv_a_proj_with_mqa", ".attn_kv_a_mqa")
    name = name.replace(".self_attn.kv_a_layernorm", ".attn_kv_a_norm")
    name = name.replace(".self_attn.kv_b_proj", ".attn_kv_b")
    name = name.replace(".self_attn.q_a_proj", ".attn_q_a")
    name = name.replace(".self_attn.q_a_layernorm", ".attn_q_a_norm")
    name = name.replace(".self_attn.q_b_proj", ".attn_q_b")
    
    name = name.replace(".shared_expert.", ".shared_experts.")
    name = name.replace(".shared_expert_", ".shared_experts_")
    name = name.replace(".gate_up_proj.", ".up_proj")
    
    name = name.replace(".mlp.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".mlp.gate", ".ffn_gate_inp")
    name = name.replace(".mlp.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".mlp.shared_experts.up_proj", ".ffn_up_shexp")
    name = name.replace(".mlp.shared_experts_gate", ".ffn_gate_inp_shexp")
    name = name.replace(".mlp.experts", "")
    name = name.replace(".mlp.experts.ffn_down_exps", ".ffn_down_exps")
    name = name.replace(".mlp.experts.ffn_gate_exps", ".ffn_gate_exps")
    name = name.replace(".mlp.experts.ffn_up_exps", ".ffn_up_exps")

    return name

if __name__ == '__main__':
    gguf_path = '/mnt/data/model/DeepSeek-Coder-V2-GGUF-WJH'
    loader = GGUFLoader(gguf_path)
    loader.load_gguf_tensor('token_embd.weight')
