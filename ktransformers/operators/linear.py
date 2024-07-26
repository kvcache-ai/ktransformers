#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure 
LastEditTime : 2024-07-26 09:27:53
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''


import torch
from torch import nn
import KTransformersOps 
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import InferenceState
from ktransformers.ktransformers_ext.operators.custom_marlin.quantize.utils.marlin_utils import (
    MarlinWorkspace,
    marlin_quantize,
    GPTQ_MARLIN_MIN_THREAD_N,
    GPTQ_MARLIN_MAX_PARALLEL,
)
from ktransformers.operators.base_operator import BaseInjectedModule
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod


#class QuantizedLinearBase(BaseInjectedModule, ABC):
class QuantizedLinearBase(ABC):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs,
    ):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        super().__init__()
        self.key = key
        self.gguf_loader = gguf_loader
        self.device = device
        self.config = config

        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        if orig_module is not None:
            self.in_features = orig_module.in_features
            self.out_features = orig_module.out_features
        else:
            shape = self.gguf_loader.tensor_info[key + ".weight"]["shape"]
            if len(shape) == 1:
                print("Warning: orig_module is not set, but has in_features or out_features equals to 1, can't get in_features and out_features from GGUF")
            self.in_features  = self.gguf_loader.tensor_info[key + ".weight"]["shape"][0]
            self.out_features = self.gguf_loader.tensor_info[key + ".weight"]["shape"][1]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def load_weight(self, override_key: str | None = None, device: str | None = None):
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        for key in keys:
            if key + ".weight" in self.gguf_loader.tensor_file_map:
                if key + ".bias" in self.gguf_loader.tensor_file_map:
                    tensors = self.load_multi(key, ["weight", "bias"], device=device)
                    tensor = tensors["weight"]
                    bias = tensors["bias"]
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[key + ".weight"]["ggml_type"]]
                    # print(torch.isinf(tensor).any(), torch.isinf(bias).any())
                    return nn.Parameter(tensor), nn.Parameter(bias)
                else:
                    tensors = self.load_multi(key, ["weight"], device=device)
                    tensor = tensors["weight"]
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[key + ".weight"]["ggml_type"]]
                    return nn.Parameter(tensor)
            else:
                raise FileNotFoundError(f"Weight file not found for key {key}")

    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + "." + k, device=device)
        return tensors

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = "cuda"):
        pass

    @abstractmethod
    def unload(self):
        pass


class QuantizedLinearTorch(QuantizedLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.w = None
        self.has_bias = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out_device = x.device
        x = x.to(device=self.device, dtype=self.dtype)
        x = x @ self.w
        if self.has_bias:
            x = x + self.bias
        x = x.to(dtype=dtype, device=out_device)
        return x

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: w = self.load_weight(device=device)

        if isinstance(w, nn.Parameter):
            self.w = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T
            self.has_bias = False
        elif isinstance(w, tuple):
            self.w = w[0].to(dtype=self.dtype).view(self.out_features, self.in_features).T
            self.bias = w[1].to(dtype=self.dtype)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        # self.linear = self.linear.to(device)
        self.w = self.w.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)

    def unload(self):
        if self.w is not None:
            self.w = None
        if self.has_bias:
            self.bias = None


class QuantizedLinearMarlin(QuantizedLinearBase):
    marlin_q_w: torch.Tensor
    marlin_s: torch.Tensor
    g_idx: torch.Tensor
    sort_indices: torch.Tensor
    has_bias: bool
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        num_bits: int = 4,  # 4-bit/8-bit is supported
        group_size: int = 64,  # -1, 32, 64, 128
        act_order: bool = False,
        is_k_full=True,
        **kwargs,
    ):
        assert device.lower() != "cpu", "Marlin quantized linear only supports GPU device"
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.num_bits = num_bits
        self.group_size = group_size
        self.act_order = act_order
        self.is_k_full = is_k_full

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = "cuda"):
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin quantized linear only supports GPU device"
        if w is None: w = self.load_weight(device=device)

        if isinstance(w, nn.Parameter):
            # pad weight
            weight = w.view(self.out_features, self.in_features).T
            self.has_bias = False
        elif isinstance(w, tuple):
            w = list(w)
            weight = w[0].view(self.out_features, self.in_features).T
            self.bias = w[1]
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        weight = weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        # Pack Marlin linear
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            weight, self.num_bits, self.group_size, self.act_order
        )
        self.workspace = MarlinWorkspace(
            self.out_features, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
        )
        self.marlin_q_w = marlin_q_w
        self.marlin_s = marlin_s
        self.g_idx = g_idx
        self.sort_indices = sort_indices
        self.k = weight.shape[0]
        self.n = weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only support input x as BF16 and FP16
        x = x.to(self.device)
        orig_shape = list(x.shape)
        orig_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        marlin_s = self.marlin_s.to(x.dtype)
        x = KTransformersOps.gptq_marlin_gemm(
            x,
            self.marlin_q_w,
            marlin_s,
            self.g_idx,
            self.sort_indices,
            self.workspace.scratch,
            self.num_bits,
            x.shape[0],
            self.n,
            x.shape[-1],
            self.is_k_full,
        )
        if self.has_bias:
            x = x + self.bias
        orig_shape[-1] = self.n
        return x.reshape(orig_shape).to(orig_dtype)

    def unload(self):

        if self.has_bias:
            self.bias = None
        self.marlin_q_w = None
        self.marlin_s = None
        self.g_idx = None
        self.sort_indices = None
        self.workspace = None
    
LINEAR_MAP = {
    "QuantizedLinearMarlin": QuantizedLinearMarlin,
    "QuantizedLinearTorch": QuantizedLinearTorch,
    "QuantizedLinearTorch": QuantizedLinearTorch,
}

class KTransformerLinear(BaseInjectedModule, QuantizedLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        generate_device: str = "cuda",
        generate_op: str| None = "QuantizedLinearMarlin",
        prefill_device: str = "cuda",
        prefill_op: str| None = "QuantizedLinearTorch",
        **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        QuantizedLinearBase.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        # build all the linear operators
        if prefill_op is not None:
            assert prefill_op in LINEAR_MAP, f"linear_type {prefill_op} not supported"
            if prefill_op == "QuantizedLinearMarlin" and (orig_module.in_features%GPTQ_MARLIN_MIN_THREAD_N!=0 or orig_module.out_features%GPTQ_MARLIN_MIN_THREAD_N!=0):
                print(f"This linear module's in_features or out_features is not divisible by GPTQ_MARLIN_MIN_THREAD_N({GPTQ_MARLIN_MIN_THREAD_N}), using QuantizedLinearTorch instead.")
                print(f"module info: key:{key} orig_module:{orig_module}")
                self.prefill_linear = QuantizedLinearTorch(key, gguf_loader, config, orig_module, prefill_device, **kwargs)
            else:
                self.prefill_linear = LINEAR_MAP[prefill_op](key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        else:
            self.prefill_linear = None

        if generate_op is not None:
            assert generate_op in LINEAR_MAP, f"linear_type {generate_op} not supported"
            if generate_op == "QuantizedLinearMarlin" and (orig_module.in_features%GPTQ_MARLIN_MIN_THREAD_N!=0 or orig_module.out_features%GPTQ_MARLIN_MIN_THREAD_N!=0):
                print(f"This linear module's in_features or out_features is not divisible by GPTQ_MARLIN_MIN_THREAD_N({GPTQ_MARLIN_MIN_THREAD_N}), using QuantizedLinearTorch instead.")
                print(f"module info: key:{key} orig_module:{orig_module}")
                self.generate_op = "QuantizedLinearTorch"
                self.generate_linear = QuantizedLinearTorch(key, gguf_loader, config, orig_module, generate_device, **kwargs)
            else:
                self.generate_linear = LINEAR_MAP[generate_op](key, gguf_loader, config, orig_module, generate_device, **kwargs)
        else:
            self.generate_linear = None
        self.device = device
        self.mode = InferenceState.UNLOAD

    def forward(self, x):
        if self.mode == InferenceState.PREFILL:
            assert self.prefill_linear is not None, "cpu linear is not initialized"
            return self.prefill_linear.forward(x)
        else:
            assert self.generate_linear is not None, "gpu linear is not initialized"
            return self.generate_linear.forward(x)

    def load(self, w: dict | nn.Parameter | tuple | None = None, mode: InferenceState = InferenceState.GENERATE):
        if not mode:
            mode = InferenceState.GENERATE
        # load to device
        if mode == InferenceState.PREFILL:
            self.generate_linear.unload()
            self.prefill_linear.load(w=w)
            self.device = self.prefill_linear.device 
        elif mode == InferenceState.GENERATE:
            self.prefill_linear.unload()
            self.generate_linear.load(w=w)
            self.device = self.generate_linear.device
        elif mode == InferenceState.UNLOAD:
            self.prefill_linear.unload()
            self.generate_linear.unload()
            self.device = "cpu"
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")
        self.mode = mode

    def unload(self):
        if self.prefill_linear is not None:
            self.prefill_linear.unload()
        if self.generate_linear is not None:
            self.generate_linear.unload()
        self.device = self.generate_linear.device

    def set_inference_mode(self, mode: InferenceState):
        if not mode: 
            mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")
