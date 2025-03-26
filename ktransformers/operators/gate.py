from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import os
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.operators.linear import KTransformersLinear
from ktransformers.util.custom_gguf import GGUFLoader
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



# adapted from https://github.com/vllm-project/vllm/blob/c77620d22d43daa7e0440e6267cbdd83f849ac64/vllm/model_executor/layers/fused_moe/fused_moe.py#L1071
# This is used by the Deepseek-V2 and Deepseek-V3 model
#@torch.compile(dynamic=True)
def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 routed_scaling_factor: float = 1.0,
                 scoring_func: str = "sigmoid",
                 e_score_correction_bias: Optional[torch.Tensor] = None):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
                                    #float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if topk > 1 and renormalize:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor # must multiply the scaling factor
    return topk_ids.to(torch.long), topk_weights.to(torch.float32)

class KMoEGateDeepSeekV3(BaseInjectedModule, KMoEGateBase):
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
            print("injecting gate_linear")
            self.gate_linear = nn.Linear(self.gating_dim, self.n_routed_experts, device=generate_device)
            self.gate_linear = KTransformersLinear(key + ".ffn_gate_inp", 
                                               gguf_loader, config, self.gate_linear, #orig_module
                                               generate_device, generate_op, prefill_device, prefill_op)
        else:
            self.gate_linear = None

    def forward(self, hidden_states) -> torch.Tensor:
        if True or self.is_windows:
            return self.orig_module.forward(hidden_states)
        
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        if self.use_quant:
            logits = self.gate_linear.forward(hidden_states)
        else:
            logits = F.linear(
                hidden_states.type(torch.float32), self.weight.type(torch.float32), None
            )
        return grouped_topk(hidden_states, logits, self.top_k, self.norm_topk_prob, self.n_group,
                            self.topk_group, self.routed_scaling_factor, "sigmoid", self.e_score_correction_bias)

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
        if not self.is_windows and self.use_quant:
            self.gate_linear.load(self.orig_module.weight)

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias = None
