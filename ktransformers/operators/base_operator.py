'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
from typing import Any
from torch import nn, Tensor
from ktransformers.util.custom_gguf import GGUFLoader
from transformers.configuration_utils import PretrainedConfig
import ktransformers.util.utils as utils
class BaseInjectedModule(nn.Module):
    
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 **kwargs):
        nn.Module.__init__(self)
        nn.Module.__setattr__(self, "orig_module", orig_module)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "gguf_loader", gguf_loader)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "prefill_device", prefill_device)
        object.__setattr__(self, "generate_device", generate_device)
        object.__setattr__(self, "device", generate_device)
        
    def __getattr__(self, name: str) -> Any:
        # __getattr__ in nn.Module doesn't call super().__getattribute__ when name is not in nn.Module.__dict__,
        # but __setattr__ in nn.Module call super().__setattr__ in that case, there may be some attribute set 
        # but can't get using __getattr__, typically these attr is build in attr of the class, so class.attr does not
        # call __getattr__.
        # Example:
        # ...import torch
        # ...l=torch.nn.Linear(100,200)
        # ...l.out_features # 200
        # ...l.__getattr__("out_features") # AttributeError: 'Linear' object has no attribute 'out_features'
        try:
            return object.__getattribute__(self, name) # if this attr belongs to BaseInjectedModule
        except:
            if name == "orig_module":
                return nn.Module.__getattr__(self, "orig_module")
            try:
                return nn.Module.__getattr__(self, "orig_module").__getattr__(name) # if this attr belongs to orig_module
            except:
                return super(nn.Module, nn.Module.__getattr__(self, "orig_module")).__getattribute__(name) # if this attr belongs to orig_module but not in nn.Module.__dict__

    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:
        if name == "orig_module":
            return nn.Module.__setattr__(self, "orig_module", value)
        elif hasattr(self, name):
            return object.__setattr__(self, name, value)
        return nn.Module.__getattr__(self, "orig_module").__setattr__(name, value)
    
    def forward(self, *args, **kwargs):
        return self.orig_module.forward(*args, **kwargs)
    
    def load(self):
        for name, child in self._modules.items():
            utils.load_weights(child, self.gguf_loader, self.key+".")
