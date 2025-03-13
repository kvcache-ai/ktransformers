#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure 
LastEditTime : 2024-08-29 09:11:16
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''


import ctypes
import torch
from torch import Tensor, nn
import KTransformersOps 
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import InferenceState
from ktransformers.ktransformers_ext.operators.custom_marlin.quantize.utils.marlin_utils import (
    MarlinWorkspace,
    marlin_quantize,
    GPTQ_MARLIN_MIN_THREAD_N,
    GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MAX_PARALLEL,
)
from ktransformers.operators.base_operator import BaseInjectedModule
from transformers.configuration_utils import PretrainedConfig
from ktransformers.ktransformers_ext.triton.fp8gemm import fp8_gemm, act_quant, weight_dequant
from abc import ABC, abstractmethod
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext
from ktransformers.operators.cpuinfer import CPUInfer
from ktransformers.server.config.config import Config

#class KLinearBase(BaseInjectedModule, ABC):
class KLinearBase(ABC):
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

        self.loaded = False # for lm_head pre-load, TODO: use new way to do lm_head pre-load when layer wise prefill.

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def load_weight(self, override_key: str | None = None, device: str | None = None):
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        for key in keys:
            if self.gguf_loader.safetensor_loader is not None:
                # using safetensor_loader
                tensor = self.gguf_loader.safetensor_loader.load_tensor(key+'.weight')
                weight_scale_inv = self.gguf_loader.safetensor_loader.load_tensor(key+'.weight_scale_inv')
                return nn.Parameter(tensor), nn.Parameter(weight_scale_inv)
                
            elif key + ".weight" in self.gguf_loader.tensor_file_map:
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


class KLinearTorch(KLinearBase):
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
        self.weight = None
        self.has_bias = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out_device = x.device
        # TODO: support CUDA Graph when using cpu, but CPUInfer is recommended.
        x = x.to(device=self.device, dtype=self.dtype)
        x = x @ self.weight
        if self.has_bias:
            x = x + self.bias
        x = x.to(dtype=dtype, device=out_device)
        return x

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if self.loaded: return
        if device is None: device = self.device
        if w is None: w = self.load_weight(device=device)
        # else: self.out_features = w.shape[0], self.in_features = w.shape[1]
        
        if isinstance(w, nn.Parameter):
            try:
                self.weight = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T
            except: 
                self.weight = w.to(dtype=self.dtype).T
            self.has_bias = False
        elif isinstance(w, tuple):
            try:
                self.weight = w[0].to(dtype=self.dtype).view(self.out_features, self.in_features).T
            except:
                self.weight = w[0].to(dtype=self.dtype).T
            self.bias = w[1].to(dtype=self.dtype)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        # self.linear = self.linear.to(device)
        self.weight = self.weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        self.loaded = True

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None

class KLinearFP8(KLinearBase):
    # this kernel requires special handling for weight
    # Please load the weight file downloaded from KVCache.AI
    marlin_q_w: torch.Tensor
    marlin_s: torch.Tensor
    g_idx: torch.Tensor
    sort_indices: torch.Tensor
    has_bias: bool
    weight: torch.Tensor
    scale_w: torch.Tensor
    bias: torch.Tensor
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        block_size: int = 128,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        orig_dtype = x.dtype        
        x_quantized, scale_x = act_quant(x, self.block_size)
        y = fp8_gemm(x_quantized, scale_x, self.weight, self.weight_scale_inv)
        return y.to(dtype=orig_dtype)
    
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: 
            w = self.load_weight(device=device) 
        ### TODO fit weight_inv format
        if isinstance(w, tuple):
            self.weight = w[0].to(device)
            self.weight_scale_inv = w[1].to(device)
            self.has_bias = False
        else:
            raise ValueError("Invalid weight type")
        self.weight = self.weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        
    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None
        
        
class KLinearMarlin(KLinearBase):
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
        self.padding = False
        self.orin_in_features = self.in_features
        self.orin_out_features = self.out_features
        if self.in_features%GPTQ_MARLIN_MIN_THREAD_K!=0 or self.out_features%GPTQ_MARLIN_MIN_THREAD_K!=0:
            #print(f"warning!, in_features={in_features} or out_features={out_features} is undivisible by GPTQ_MARLIN_MIN_THREAD_K={GPTQ_MARLIN_MIN_THREAD_K} and GPTQ_MARLIN_MIN_THREAD_N={GPTQ_MARLIN_MIN_THREAD_N}, padding")
            self.padding = True
            self.in_features = (self.in_features+GPTQ_MARLIN_MIN_THREAD_K-1)//GPTQ_MARLIN_MIN_THREAD_K*GPTQ_MARLIN_MIN_THREAD_K
            self.out_features = (self.out_features+GPTQ_MARLIN_MIN_THREAD_N-1)//GPTQ_MARLIN_MIN_THREAD_N*GPTQ_MARLIN_MIN_THREAD_N
            #print(f"After padding: in_features={in_features}, out_features={out_features}")
        
        self.k = self.in_features
        self.n = self.out_features

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if self.loaded: return
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin quantized linear only supports GPU device"
        
        #if self.in_features * self.out_features:
        if w is None: 
            w = self.load_weight(device=device) 

        if isinstance(w, nn.Parameter):
            # pad weight
            weight = w.view(self.orin_out_features, self.orin_in_features).T
            self.has_bias = False
        elif isinstance(w, tuple):
            w = list(w)
            weight = w[0].view(self.orin_out_features, self.orin_in_features).T
            self.bias = w[1].view(self.orin_out_features)
            self.bias = w[1]
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        weight = weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
            
        if self.padding:
            padded_weight = torch.zeros(self.in_features, self.out_features, device=self.device)
            padded_weight[:self.orin_in_features, :self.orin_out_features] = weight
            weight = padded_weight

        # Pack Marlin linear
        marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            weight, self.num_bits, self.group_size, self.act_order
        )
        self.workspace = MarlinWorkspace(
            self.out_features, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL,self.device
        )
        self.weight = marlin_q_w # modeling_xxx.py may use linear.weight
        self.marlin_q_w = marlin_q_w
        self.marlin_s = marlin_s
        self.g_idx = g_idx
        self.sort_indices = sort_indices
        self.k = weight.shape[0]
        self.n = weight.shape[1]
        self.loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only support input x as BF16 and FP16
        x = x.to(self.device)
        orig_shape = list(x.shape)
        orig_dtype = x.dtype
        x = x.reshape(-1, orig_shape[-1])
        x = x.reshape(-1, x.shape[-1])
        if self.padding:
            padding_input=torch.empty(x.shape[0], self.in_features, device=x.device, dtype=x.dtype)
            padding_input[:,:self.orin_in_features] = x
            x = padding_input
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
        if self.padding:
            x = x[:,:self.orin_out_features]
            orig_shape[-1] = self.orin_out_features
        else:
            orig_shape[-1] = self.out_features
        if self.has_bias:
            x = x + self.bias
        return x.reshape(orig_shape).to(orig_dtype)

    def unload(self):

        if self.has_bias:
            self.bias = None
        self.marlin_q_w = None
        self.marlin_s = None
        self.g_idx = None
        self.sort_indices = None
        self.workspace = None

class KLinearCPUInfer(KLinearBase):
    CPU_INFER = CPUInfer(Config().cpu_infer)
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda", # this device mean which device the output should on. TODO: support cpu.
        stride = 16,
        group_max_len = 1024,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.w = None
        self.has_bias = False
        self.stride = stride
        self.group_max_len = group_max_len
        self.out_device = out_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape # [batch_size, q_len, hidden_size]
        if origin_shape[1] == 1 and torch.cuda.is_current_stream_capturing():
            out_device = x.device
            self.input_tensor_cpu.copy_(x, non_blocking=True)
            qlen = origin_shape[1]
            KLinearCPUInfer.CPU_INFER.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                self.linear.forward(
                    qlen, 
                    self.input_tensor_cpu.data_ptr(), 
                    self.output_cpu.data_ptr()
                )
            )
            KLinearCPUInfer.CPU_INFER.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
            self.output_gpu.copy_(self.output_cpu, non_blocking=True)
            if self.has_bias:
                self.output_gpu += self.bias
            return self.output_gpu
        else:
            dtype = x.dtype
            out_device = x.device
            x = x.to(device=self.device)
            qlen = origin_shape[1]
            output_shape = (*origin_shape[:-1], self.out_features)
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
            KLinearCPUInfer.CPU_INFER.submit(
                self.linear.forward(
                    qlen, 
                    x.data_ptr(), 
                    output.data_ptr()
                )
            )
            KLinearCPUInfer.CPU_INFER.sync()
            if self.has_bias:
                output = output + self.bias
            output = output.to(dtype=dtype, device=out_device)
            return output

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None, warmup:bool = True):
        print(f"loading {self.key} to {self.device} using CPUInfer")
        if device is None: device = self.device
        self.load_weights(w=w, device=device)
        if self.bias is not None:
            self.has_bias = True
            self.bias = self.bias.to(device)
            
        weight_ptr = ctypes.addressof(
            ctypes.cast(self.weight.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        config = cpuinfer_ext.linear.LinearConfig(self.in_features, self.out_features, self.stride, self.group_max_len, weight_ptr, self.weight_type, 30)
        self.linear = cpuinfer_ext.linear.Linear(config)
        
        if warmup:
            KLinearCPUInfer.CPU_INFER.submit(self.linear.warm_up())
            KLinearCPUInfer.CPU_INFER.sync()
        self.input_tensor_cpu = torch.zeros((1, 1, self.in_features), device="cpu", pin_memory=True)
        self.output_cpu = torch.zeros((1, 1, self.out_features), device="cpu", pin_memory=True, dtype=torch.bfloat16)
        self.output_gpu = torch.zeros((1, 1, self.out_features), device=self.out_device)

    def load_weights(self, w: dict | nn.Parameter | tuple | None = None, device: str = "cpu"):
        if self.key + ".weight" in self.gguf_loader.tensor_info:
            if self.key + ".bias" in self.gguf_loader.tensor_file_map:
                self.weight = self.gguf_loader.get_mmap_tensor(self.key + ".weight")
                self.weight_type = self.gguf_loader.tensor_info[self.key + ".weight"]["ggml_type"]
                self.bias = self.gguf_loader.load_gguf_tensor(self.key + ".bias", device=device)
            else:
                self.weight = self.gguf_loader.get_mmap_tensor(self.key + ".weight")
                self.weight_type = self.gguf_loader.tensor_info[self.key + ".weight"]["ggml_type"]
                self.bias = None
        else:
            raise ValueError(f"Linear {self.key} not found in gguf_loader")

    def unload(self):
        if self.w is not None:
            self.w = None
        if self.has_bias:
            self.bias = None        

LINEAR_MAP = {
    "KLinearMarlin": KLinearMarlin,
    "KLinearTorch": KLinearTorch,
    "KLinearCPUInfer": KLinearCPUInfer,
    "KLinearFP8": KLinearFP8,
}

class KTransformersLinear(BaseInjectedModule, KLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        # device: str = "cuda",
        generate_device: str = "cuda",
        generate_op: str| None = "KLinearMarlin",
        prefill_device: str = "cuda",
        prefill_op: str| None = "KLinearTorch",
        **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KLinearBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        # build all the linear operators
        if prefill_op is not None:
            assert prefill_op in LINEAR_MAP, f"linear_type {prefill_op} not supported"
            self.prefill_linear = LINEAR_MAP[prefill_op](key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        else:
            self.prefill_linear = None

        if generate_op is not None:
            assert generate_op in LINEAR_MAP, f"linear_type {generate_op} not supported"
            self.generate_linear = LINEAR_MAP[generate_op](key, gguf_loader, config, orig_module, generate_device, **kwargs)
        else:
            self.generate_linear = None
        self.mode = InferenceState.UNLOAD

    def forward(self, x):
        if self.mode == InferenceState.PREFILL:
            assert self.prefill_linear is not None, "cpu linear is not initialized"
            y = self.prefill_linear.forward(x)
        else:
            assert self.generate_linear is not None, "gpu linear is not initialized"
            y = self.generate_linear.forward(x)
        return y

    def load(self, w: dict | nn.Parameter | tuple | None = None, mode: InferenceState = InferenceState.GENERATE):
        if not mode:
            mode = InferenceState.GENERATE
        # load to device
        if mode == InferenceState.PREFILL:
            self.generate_linear.unload()
            self.prefill_linear.load(w=w)
            self.device = self.prefill_linear.device
            self.weight = self.prefill_linear.weight # modeling_xxx.py may use linear.weight
        elif mode == InferenceState.GENERATE:
            self.prefill_linear.unload()
            self.generate_linear.load(w=w)
            self.device = self.generate_linear.device
            self.weight = self.generate_linear.weight # modeling_xxx.py may use linear.weight
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
