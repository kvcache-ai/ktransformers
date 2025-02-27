
from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from ktransformers.operators.base_operator import BaseInjectedModule

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, MOE
import ctypes
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
import time


# class Base(BaseInjectedModule, ABC):
class KMoEGateBase(ABC):
    def __init__(self, 
                 key: str, 
                 gguf_loader: GGUFLoader, 
                 config: PretrainedConfig, 
                 orig_module: nn.Module, 
                 device: str = "cuda", 
                 **kwargs):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        super().__init__()
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device
        self.orig_module = orig_module
    
    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str = "cpu", warmup: bool = False):
        pass
    
    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            key = ".".join(key.split(".")[:-1])
            if self.gguf_loader.safetensor_loader is not None:
                targets = [".ffn_gate_inp.weight", ".exp_probs_b.bias"]
                weight = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_gate_inp.weight") 
                e_score_correction_bias = self.gguf_loader.safetensor_loader.load_tensor(key + ".exp_probs_b.bias")
                weight_type = weight.dtype
                e_score_correction_bias_type = e_score_correction_bias.dtype
                res = {"weight": weight, "e_score_correction_bias": e_score_correction_bias,  "weight_type": weight_type, "e_score_correction_bias_type": e_score_correction_bias_type}
            elif key + ".ffn_gate_inp.weight" in self.gguf_loader.tensor_info:
                targets = [".ffn_gate_inp.weight", ".exp_probs_b.bias"]
                tensors = self.load_multi(key, targets, device=device)
                weight = tensors[".ffn_gate_inp.weight"]
                e_score_correction_bias = tensors[".exp_probs_b.bias"]
                weight_type = self.gguf_loader.tensor_info[key + ".ffn_gate_inp.weight"]["ggml_type"]
                e_score_correction_bias_type = self.gguf_loader.tensor_info[key + ".exp_probs_b.bias"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {"weight": weight, "e_score_correction_bias": e_score_correction_bias,  "weight_type": weight_type, "e_score_correction_bias_type": e_score_correction_bias_type}
        return res
    
    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors


class KMoEGate(BaseInjectedModule, KMoEGateBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        prefill_device: str = "cuda",
        generate_device: str = "cuda",
        **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KMoEGateBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def forward(self, hidden_states) -> torch.Tensor:
        return self.orig_module.forward(hidden_states)

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: w = self.load_weights(device=device)
        
        if isinstance(w, dict):
            self.weight_type = w["weight_type"]
            self.e_score_correction_bias_type = w["e_score_correction_bias_type"]
            self.orig_module.weight = nn.Parameter(w["weight"])
            self.orig_module.e_score_correction_bias = nn.Parameter(w["e_score_correction_bias"])
        else:
            raise ValueError("Invalid weight type")
        self.orig_module.weight = nn.Parameter(self.orig_module.weight.to(device))
        self.orig_module.e_score_correction_bias = nn.Parameter(self.orig_module.e_score_correction_bias.to(device))

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias = None
