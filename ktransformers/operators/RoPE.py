"""
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

from torch import nn
from transformers import ROPE_INIT_FUNCTIONS
from ktransformers.models.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
)
from ktransformers.models.modeling_deepseek_v3 import (
    DeepseekV3RotaryEmbedding
)
from ktransformers.models.modeling_deepseek import (
    DeepseekV2YarnRotaryEmbedding,
    DeepseekV2RotaryEmbedding,
)
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import InferenceState
from transformers.configuration_utils import PretrainedConfig
import torch

# Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding with Mixtral->Qwen2Moe
class RotaryEmbedding(BaseInjectedModule, DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        #  device: str = "cuda",
        generate_device: str = "cuda",
        prefill_device: str = "cuda",
        **kwargs,
    ):
        BaseInjectedModule.__init__(
            self, key, gguf_loader, config, orig_module, generate_device, **kwargs
        )
        self.orig_module.__init__(
            orig_module.dim, orig_module.max_position_embeddings, orig_module.base
        )
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def load(self):
        self.orig_module.__init__(
            self.orig_module.dim,
            self.orig_module.max_position_embeddings,
            self.orig_module.base,
            self.device,
        )


class RotaryEmbeddingV3(BaseInjectedModule):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        #  device: str = "cuda",
        generate_device: str = "cuda",
        prefill_device: str = "cuda",
        **kwargs,
    ):
        BaseInjectedModule.__init__(
            self, key, gguf_loader, config, orig_module, generate_device, **kwargs
        )
        self.generate_device = generate_device
        self.prefill_device = prefill_device
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)   

    def load(self):
        self._init(
            dim=self.config.qk_rope_head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            device=self.device,
        )
    def _init(self, dim, max_position_embeddings, base, device, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

class RotaryEmbeddingV2(BaseInjectedModule, LlamaRotaryEmbedding):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        generate_device: str = "cuda",
        prefill_device: str = "cuda",
        **kwargs,
    ):
        BaseInjectedModule.__init__(
            self, key, gguf_loader, config, orig_module, generate_device, **kwargs
        )
        self.orig_module.__init__(
            orig_module.dim,
            orig_module.max_position_embeddings,
            orig_module.base,
            None,
            orig_module.scaling_factor,
            orig_module.rope_type,
            orig_module.config,
        )
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def load(self):
        self.orig_module.__init__(
            self.orig_module.dim,
            self.orig_module.max_position_embeddings,
            self.orig_module.base,
            self.device,
            self.orig_module.scaling_factor,
            self.orig_module.rope_type,
            self.orig_module.config,
        )

class YarnRotaryEmbedding(BaseInjectedModule, DeepseekV2YarnRotaryEmbedding):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        #  device: str = "cuda",
        generate_device: str = "cuda",
        prefill_device: str = "cuda",
        **kwargs,
    ):
        BaseInjectedModule.__init__(
            self, key, gguf_loader, config, orig_module, generate_device, **kwargs
        )
        self.orig_module.__init__(
            orig_module.dim,
            orig_module.max_position_embeddings,
            orig_module.base,
            None,  # device
            orig_module.scaling_factor,
            orig_module.original_max_position_embeddings,
            orig_module.beta_fast,
            orig_module.beta_slow,
            orig_module.mscale,
            orig_module.mscale_all_dim,
        )
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def load(self):
        self.orig_module.__init__(
            self.orig_module.dim,
            self.orig_module.max_position_embeddings,
            self.orig_module.base,
            self.generate_device,
            self.orig_module.scaling_factor,
            self.orig_module.original_max_position_embeddings,
            self.orig_module.beta_fast,
            self.orig_module.beta_slow,
            self.orig_module.mscale,
            self.orig_module.mscale_all_dim,
        )

class DeepSeekV3YarnRotaryEmbedding(BaseInjectedModule, DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        #  device: str = "cuda",
        generate_device: str = "cuda",
        prefill_device: str = "cuda",
        **kwargs,
    ):
        BaseInjectedModule.__init__(
            self, key, gguf_loader, config, orig_module, generate_device, **kwargs
        )
        self.generate_device = generate_device
        self.prefill_device = prefill_device

    def load(self):
        # TODO support perlayer prefill
        self.orig_module.__init__(
            self.config,
            device=self.generate_device
        )
        return

class DynamicNTKScalingRotaryEmbedding(
    BaseInjectedModule, LlamaDynamicNTKScalingRotaryEmbedding
):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        **kwargs,
    ):
        BaseInjectedModule.__init__(
            self, key, gguf_loader, config, orig_module, device, **kwargs
        )
        self.orig_module.__init__(
            orig_module.dim,
            orig_module.max_position_embeddings,
            orig_module.base,
            None,  # device
            orig_module.scaling_factor,
            orig_module.rope_type,
            orig_module.config,
        )

    def load(self):
        self.orig_module.__init__(
            self.orig_module.dim,
            self.orig_module.max_position_embeddings,
            self.orig_module.base,
            self.orig_module.device,
            self.orig_module.scaling_factor,
            self.orig_module.rope_type,
            self.orig_module.config,
        )
