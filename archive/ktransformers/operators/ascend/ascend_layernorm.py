# coding=utf-8
# Copyright (c) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import Optional, Union, Tuple

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeRMSNorm
from ktransformers.util import utils
from ktransformers.util.custom_loader import GGUFLoader


class KDeepseekV3RMSNormW8A8(BaseInjectedModule):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "npu",
                 generate_device: str = "npu",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.weight = nn.Parameter(torch.ones(self.orig_module.hidden_size))
        self.bias = nn.Parameter(torch.ones(self.orig_module.hidden_size))
        self.variance_epsilon = self.orig_module.variance_epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        out = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0] + self.bias
        return out.to(input_dtype)

    def load(self):
        self.weight = self.gguf_loader.safetensor_loader.load_tensor(self.key + ".weight").to(utils.get_current_device())
        self.bias = self.gguf_loader.safetensor_loader.load_tensor(self.key + ".bias").to(utils.get_current_device())

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.bias is not None:
            self.bias = None

class KQwen3MoeRMSNormW8A8(BaseInjectedModule):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "npu",
                 generate_device: str = "npu",
                 **kwargs):

        super().__init__(key, gguf_loader, config, orig_module,
                         prefill_device, generate_device, **kwargs)

        self.hidden_size = orig_module.hidden_size
        self.variance_epsilon = orig_module.variance_epsilon
        self.weight = nn.Parameter(orig_module.weight.data.clone())

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float16)
        gamma = self.weight.to(torch.float16)

        input_dtype = x.dtype
        out = torch_npu.npu_rms_norm(
            x,
            gamma,
            self.variance_epsilon
        )[0]

        return out.to(input_dtype)

    def load(self):
        device = utils.get_current_device()
        self.weight = self.gguf_loader.safetensor_loader.load_tensor(self.key + ".weight").to(device)

        try:
            self.bias = (
                self.gguf_loader.safetensor_loader
                .load_tensor(self.key + ".bias")
                .to(device)
            )
        except KeyError:
            self.bias = None

    def unload(self):
        self.weight = None
        self.bias = None

class KQwen3FinalRMSNormNPU(nn.Module):
    def __init__(self, orig_module: nn.Module):
        super().__init__()
        assert hasattr(orig_module, "weight"), "orig_module must have weight"
        self.variance_epsilon = getattr(orig_module, "variance_epsilon", 1e-6)

        w = orig_module.weight.detach()
        if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            w = w.to(torch.float16)
        else:
            if w.dtype == torch.float32:
                w = w.to(torch.float16)

        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x = x.contiguous()
        gamma = self.weight
        x_rms = x.to(dtype=gamma.dtype)

        out = torch_npu.npu_rms_norm(
            x_rms,
            gamma,
            self.variance_epsilon
        )[0]

        return out.to(input_dtype)