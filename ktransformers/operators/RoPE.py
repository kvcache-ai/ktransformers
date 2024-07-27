'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
from torch import nn
from ktransformers.models.modeling_deepseek import DeepseekV2YarnRotaryEmbedding, DeepseekV2RotaryEmbedding
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import InferenceState
from transformers.configuration_utils import PretrainedConfig
# Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding with Mixtral->Qwen2Moe
class RotaryEmbedding(BaseInjectedModule, DeepseekV2RotaryEmbedding):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.dim,
            orig_module.max_position_embeddings,
            orig_module.base)
    
    def load(self):
        self.orig_module.__init__(self.orig_module.dim,
            self.orig_module.max_position_embeddings,
            self.orig_module.base,
            self.device)
    
class YarnRotaryEmbedding(BaseInjectedModule, DeepseekV2YarnRotaryEmbedding):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.dim,
            orig_module.max_position_embeddings,
            orig_module.base,
            None, #device
            orig_module.scaling_factor,
            orig_module.original_max_position_embeddings,
            orig_module.beta_fast,
            orig_module.beta_slow,
            orig_module.mscale,
            orig_module.mscale_all_dim)
        
    
    def load(self):
        self.orig_module.__init__(self.orig_module.dim,
            self.orig_module.max_position_embeddings,
            self.orig_module.base,
            self.device,
            self.orig_module.scaling_factor,
            self.orig_module.original_max_position_embeddings,
            self.orig_module.beta_fast,
            self.orig_module.beta_slow,
            self.orig_module.mscale,
            self.orig_module.mscale_all_dim)
    
