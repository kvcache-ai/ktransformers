
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_loader import GGUFLoader
from transformers import PretrainedConfig
import torch.nn as nn
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3MLP
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeMLP
class kDeepseekV3MLP(DeepseekV3MLP, BaseInjectedModule):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.hidden_size, orig_module.intermediate_size)
    def forward(self, x, bsz_tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, bsz_tensor)) * self.up_proj(x, bsz_tensor), bsz_tensor)
        return down_proj
class KQwen2MoeMLP(Qwen2MoeMLP, BaseInjectedModule):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.intermediate_size)
    def forward(self, x, bsz_tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, bsz_tensor)) * self.up_proj(x, bsz_tensor), bsz_tensor)
        return down_proj