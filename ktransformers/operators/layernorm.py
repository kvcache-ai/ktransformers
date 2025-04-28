'''
Date: 2024-11-13 15:05:52
LastEditors: Xie Weiyu ervinxie@qq.com
LastEditTime: 2024-11-25 08:59:19
'''
"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig
import torch
import torch.nn as nn
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeRMSNorm
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeRMSNorm
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader
from flashinfer.norm import (
    fused_add_rmsnorm,
    rmsnorm,
)


logger = logging.getLogger(__name__)


class RMSNorm(DeepseekV3RMSNorm, BaseInjectedModule):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.hidden_size,
            orig_module.variance_epsilon)

    def forward(
        self,
        x: torch.Tensor,
        batch_size_tensor: torch.Tensor = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        #return self.forward_native(x, residual)
        if batch_size_tensor is None:
            return self.forward_native(x)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, batch_size_tensor, self.variance_epsilon)
            #residual = x + residual
            #out = rmsnorm(residual, self.weight.data, batch_size_tensor, self.variance_epsilon)
            return x, residual
        # print(x.shape, self.weight.data.shape, self.variance_epsilon, x.dtype, self.weight.data.dtype, x.device, self.weight.device, x.is_contiguous(), self.weight.data.is_contiguous())
        out = rmsnorm(x, self.weight.data, batch_size_tensor,self.variance_epsilon)
        return out

    def forward_native(
        self, hidden_states    
    ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class KQwen2MoeRMSNorm(Qwen2MoeRMSNorm, BaseInjectedModule):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(config.hidden_size,
            orig_module.variance_epsilon)

    def forward(
        self,
        x: torch.Tensor,
        batch_size_tensor: torch.Tensor = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        #return self.forward_native(x, residual)
        if batch_size_tensor is None:
            return self.forward_native(x)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, batch_size_tensor, self.variance_epsilon)
            #residual = x + residual
            #out = rmsnorm(residual, self.weight.data, batch_size_tensor, self.variance_epsilon)
            return x, residual
        # print(x.shape, self.weight.data.shape, self.variance_epsilon, x.dtype, self.weight.data.dtype, x.device, self.weight.device, x.is_contiguous(), self.weight.data.is_contiguous())
        out = rmsnorm(x, self.weight.data, batch_size_tensor,self.variance_epsilon)
        return out

    def forward_native(
        self, hidden_states    
    ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class KQwen3MoeRMSNorm(Qwen3MoeRMSNorm, BaseInjectedModule):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.hidden_size,
            orig_module.variance_epsilon)

    def forward(
        self,
        x: torch.Tensor,
        batch_size_tensor: torch.Tensor = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        #return self.forward_native(x, residual)
        bsz, hidden_size = x.shape
        x = x.view(-1, self.orig_module.hidden_size)
        if batch_size_tensor is None:
            return self.forward_native(x)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, batch_size_tensor, self.variance_epsilon)
            #residual = x + residual
            #out = rmsnorm(residual, self.weight.data, batch_size_tensor, self.variance_epsilon)
            return x, residual
        # print(x.shape, self.weight.data.shape, self.variance_epsilon, x.dtype, self.weight.data.dtype, x.device, self.weight.device, x.is_contiguous(), self.weight.data.is_contiguous())
        out = rmsnorm(x, self.weight.data, batch_size_tensor,self.variance_epsilon)
        out = out.view(bsz, hidden_size)
        return out

    def forward_native(
        self, hidden_states    
    ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
