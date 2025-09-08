import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util import utils
from ktransformers.util.custom_gguf import GGUFLoader


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
