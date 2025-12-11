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

from abc import abstractmethod

import torch
import torch_npu
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig

from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.operators.linear import KLinearBase, LINEAR_MAP
from ktransformers.util import utils
from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.util.utils import InferenceState
from ktransformers.util.ascend.ascend_utils import get_safetensors_cut_weight, get_tensor_parallel_size, get_tensor_parallel_group
from ktransformers.util.custom_gguf import translate_name_to_gguf


class KLinearW8A8(KLinearBase):
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

    def load_weight(self, override_key: str | None = None, device: str | None = None):
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        fake_tensor = torch.tensor([1])
        for key in keys:
            if device is None:
                device = utils.get_current_device()
            
            key = translate_name_to_gguf(key)
            if key == "lm_head":
                key = "output"
                
            if key + ".weight" in self.gguf_loader.safetensor_loader.tensor_file_map:
                if key + ".deq_scale" in self.gguf_loader.safetensor_loader.tensor_file_map:
                    qweight = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight")
                    deq_scale = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.deq_scale")
                    quant_bias = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.quant_bias")
                    input_scale = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.input_scale")
                    input_offset = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.input_offset")
                    tensors = (qweight, deq_scale, quant_bias, input_scale, input_offset)
                    return tensors
                elif key + ".weight_scale" in self.gguf_loader.safetensor_loader.tensor_file_map:
                    if key.endswith("ffn_gate_shexp"):
                        parts = key.split(".")
                        layer = parts[1]
                        gate_weight = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_gate_shexp.weight")
                        gate_weight = get_safetensors_cut_weight(self.key, gate_weight).t()
                        up_weight = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_up_shexp.weight")
                        up_weight = get_safetensors_cut_weight(self.key, up_weight).t()
                        gate_scale = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_gate_shexp.weight_scale")
                        gate_scale = get_safetensors_cut_weight(self.key, gate_scale)
                        up_scale = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_up_shexp.weight_scale")
                        up_scale = get_safetensors_cut_weight(self.key, up_scale)
                        gate_up_weight = torch.cat((gate_weight, up_weight), 1)
                        gate_up_scale = torch.cat((gate_scale, up_scale), 0)
                        gate_offset = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_gate_shexp.weight_offset")
                        up_offset = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_up_shexp.weight_offset")
                        gate_up_offset = torch.cat((gate_offset, up_offset), 0)
                        tensors = (gate_up_weight, gate_up_scale, gate_up_offset)
                    elif key.endswith("ffn_up_shexp"):
                        return fake_tensor
                    else:
                        qweight = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight")
                        weight_scale = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight_scale")
                        weight_offset = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight_offset")
                        tensors = (qweight, weight_scale, weight_offset)
                    return tensors
                else:
                    weight = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight")
                    return weight
            else:
                raise FileNotFoundError(f"Weight file not found for key {key}")

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = "cuda"):
        pass

    @abstractmethod
    def unload(self):
        pass


class KLinearTorchW8A8A2(KLinearW8A8):
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
        self.input_scale = None
        self.input_offset = None
        self.quant_bias = None
        self.deq_scale = None
        self.weight_scale = None
        self.weight_offset = None

    def forward(self, x: torch.Tensor, bsz_tensor) -> torch.Tensor:
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return torch.matmul(x, self.weight)

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None):
        if device is None: device = utils.get_current_device()
        device = utils.CUR_DEVICE
        if w is None:
            w = self.load_weight()

        if isinstance(w, nn.Parameter):
            try:
                self.weight = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T.contiguous()
            except:
                self.weight = w.to(dtype=self.dtype).T.contiguous()
            self.weight = self.weight.to(device)
            if self.has_bias:
                self.bias = self.bias.to(device)
        elif isinstance(w, tuple):
            w_list = list(w)
            if len(w_list) == 3:
                self.weight = w_list[0]
                self.weight_scale = w_list[1].view(-1)
                self.weight_offset = w_list[2]
                self.weight = self.weight.to(utils.CUR_DEVICE)
                self.weight_scale = self.weight_scale.to(utils.CUR_DEVICE)
                if self.key.endswith("ffn_gate_shexp") is not True:
                    self.weight = get_safetensors_cut_weight(self.key, self.weight).t()
                    weight_scale = get_safetensors_cut_weight(self.key, self.weight_scale)
                    self.weight_scale = weight_scale.clone()
                    del weight_scale
            else:
                for i in range(len(w_list)):
                    w_list[i] = get_safetensors_cut_weight(self.key, w_list[i])
                    w_list[i] = w_list[i].to(utils.CUR_DEVICE)
                self.weight = w_list[0]
                self.deq_scale = w_list[1]
                self.quant_bias = w_list[2]
                if "attn_output" in self.key or "ffn_down" in self.key:
                    if torch.distributed.get_rank(get_tensor_parallel_group()) != 0:
                        self.quant_bias = torch.zeros_like(self.quant_bias, dtype=self.quant_bias.dtype, device=self.quant_bias.device)

                self.input_scale = w_list[3]
                self.input_offset = w_list[4]
        elif isinstance(w, torch.Tensor):
            self.weight = w.T.contiguous()
            self.weight = self.weight.to(device)
            if "kv_b" not in self.key and ("output" in  self.key or "eh_proj" in self.key):
                self.weight = torch_npu.npu_format_cast(self.weight, 29)
        else:
            raise ValueError(f"Invalid weight type {self.key=} {type(w)=}")

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None
        self.input_scale = None
        self.input_offset = None
        self.quant_bias = None
        self.deq_scale = None
        self.weight_scale = None
        self.weight_offset = None


LINEAR_MAP["KLinearTorchW8A8A2"] = KLinearTorchW8A8A2


class KTransformersLinearW8A8A2(BaseInjectedModule, KLinearW8A8):
    def __init__(
            self,
            key: str,
            gguf_loader: GGUFLoader,
            config: PretrainedConfig,
            orig_module: nn.Module,
            generate_device: str = "cuda",
            generate_op: str | None = "KLinearMarlin",
            prefill_device: str = "cuda",
            prefill_op: str | None = "KLinearTorch",
            **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KLinearW8A8.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
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

    def forward(self, x, bsz_tensor=None):
        if self.mode == InferenceState.PREFILL:
            assert self.prefill_linear is not None, "cpu linear is not initialized"
            y = self.prefill_linear.forward(x, bsz_tensor)
        else:
            assert self.generate_linear is not None, "gpu linear is not initialized"
            y = self.generate_linear.forward(x, bsz_tensor)
        return y

    def load(self, w: dict | nn.Parameter | tuple | None = None, mode: InferenceState = InferenceState.GENERATE):
        if not mode:
            mode = InferenceState.GENERATE
        # load to device
        if mode == InferenceState.PREFILL:
            self.generate_linear.unload()
            self.prefill_linear.load(w=w)
            self.device = self.prefill_linear.device
            self.weight = self.prefill_linear.weight  # modeling_xxx.py may use linear.weight
            self.input_scale = self.prefill_linear.input_scale
            self.input_offset = self.prefill_linear.input_offset
            self.quant_bias = self.prefill_linear.quant_bias
            self.deq_scale = self.prefill_linear.deq_scale
            self.weight_scale = self.prefill_linear.weight_scale
            self.weight_offset = self.prefill_linear.weight_offset
        elif mode == InferenceState.GENERATE:
            self.prefill_linear.unload()
            self.generate_linear.load(w=w)
            self.device = self.generate_linear.device
            self.weight = self.generate_linear.weight  # modeling_xxx.py may use linear.weight
            self.input_scale = self.generate_linear.input_scale
            self.input_offset = self.generate_linear.input_offset
            self.quant_bias = self.generate_linear.quant_bias
            self.deq_scale = self.generate_linear.deq_scale
            self.weight_scale = self.generate_linear.weight_scale
            self.weight_offset = self.generate_linear.weight_offset
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
