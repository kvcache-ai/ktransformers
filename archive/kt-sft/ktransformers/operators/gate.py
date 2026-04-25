from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import os
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.operators.linear import KTransformersLinear
from ktransformers.util.custom_loader import GGUFLoader, ModelLoader, SafeTensorLoader
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod


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
            # key = ".".join(key.split(".")[:-1])
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_gate(key, device=device)
            elif self.gguf_loader.has_tensor(key+".weight"):
                # targets = [".ffn_gate_inp.weight", ".exp_probs_b.bias"]
                targets = [".weight", ".e_score_correction_bias"]
                tensors = self.load_multi(key, targets, device=device)
                weight = tensors[".weight"]
                e_score_correction_bias = tensors[".e_score_correction_bias"]
                # weight_type = self.gguf_loader.tensor_info[key + ".weight"]["ggml_type"]
                res = {"weight": weight, "e_score_correction_bias": e_score_correction_bias}
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")

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
        generate_device: str = "cuda",
        prefill_device: str = "cuda",
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


class KMoEGateQwen2Moe(BaseInjectedModule, KMoEGateBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "cuda",
        generate_op: str| None = "KLinearMarlin",
        prefill_device: str = "cuda",
        prefill_op: str| None = "KLinearMarlin",
        use_quant: bool = False,
        **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KMoEGateBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device
        self.generate_op = generate_op
        self.prefill_op = prefill_op
        self.is_windows = os.name == 'nt'
        self.use_quant = use_quant
        if not self.is_windows and use_quant:
            self.gate_linear = nn.Linear(self.gating_dim, self.n_routed_experts, device=generate_device)
            self.gate_linear = KTransformersLinear(key + ".ffn_gate_inp", 
                                               gguf_loader, config, self.gate_linear, #orig_module
                                               generate_device, generate_op, prefill_device, prefill_op)
        else:
            self.gate_linear = None

    def forward(self, hidden_states) -> torch.Tensor:
        if self.is_windows:
            return self.orig_module.forward(hidden_states)
        
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        if self.use_quant:
            logits = self.gate_linear.forward(logits)
        else:
            logits = F.linear(
                hidden_states.type(torch.float32), self.weight.type(torch.float32), None
            )
            
        return grouped_topk(hidden_states, logits,
                            self.top_k, self.norm_topk_prob,
                            self.n_group, self.topk_group)

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: w = self.load_weights(device=device)
        
        if isinstance(w, dict):
            self.orig_module.weight = nn.Parameter(w["weight"])
            self.orig_module.e_score_correction_bias = nn.Parameter(w["e_score_correction_bias"])
        else:
            raise ValueError("Invalid weight type")
        self.orig_module.weight = nn.Parameter(self.orig_module.weight.to(device))
        self.orig_module.e_score_correction_bias = nn.Parameter(self.orig_module.e_score_correction_bias.to(device))
        if not self.is_windows and self.use_quant:
            self.gate_linear.load(self.orig_module.weight)

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias = None


class KMoEGateIPEXLLM(KMoEGate):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "xpu",
        prefill_device: str = "xpu",
        **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KMoEGate.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def forward(self, hidden_states) -> torch.Tensor:
        x = hidden_states.view(-1, hidden_states.size(-1))
        logits = torch.nn.functional.linear(
            x.type(torch.float32), self.orig_module.weight.type(torch.float32), None
        )
        scores = logits.sigmoid()

        from ipex_llm.transformers.models.common import moe_group_topk
        topk_idx, topk_weight = moe_group_topk(scores, self.orig_module.e_score_correction_bias,
                                               self.n_group, self.topk_group, self.top_k,
                                               self.norm_topk_prob, self.routed_scaling_factor)
        return topk_idx, topk_weight.to(x.dtype)