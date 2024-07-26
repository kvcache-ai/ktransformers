'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
from typing import Mapping, List
import torch
import yaml
import re
from torch import nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
# from operators import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader, translate_name_to_gguf
from ktransformers.util.utils import set_module, load_weights
import itertools

def inject(module, local_optimization_dict, model_config:AutoConfig ,gguf_loader:GGUFLoader, prefix=''):
    for name, child in module._modules.items():
        if child is not None:
            child_prefix = prefix + name
            if child_prefix in local_optimization_dict:
                inject_module_meta=local_optimization_dict[child_prefix]
                if isinstance(inject_module_meta, Mapping):
                    import_path = inject_module_meta["class"].split(".")
                    import_module_name = ".".join(import_path[:-1])
                    import_class_name = import_path[-1]
                    module_cls=getattr(__import__(import_module_name, fromlist=[""]), import_class_name)
                    print(f"Injecting {child_prefix} as", import_module_name, ".", import_class_name)
                    inject_module=module_cls(key = inject_module_meta["key"], gguf_loader = gguf_loader, config = model_config, orig_module=child, device = inject_module_meta["device"], **inject_module_meta["kwargs"])
                    set_module(module, name, inject_module)
                elif isinstance(inject_module_meta, str):
                    assert inject_module_meta=="default", "for str inject_module_meta, only support \"default\"."
                else:
                    raise Exception("inject_module_meta must be a dict or str")
                child_prefix += "."
                child_optimization_dict = {k: v for k, v in local_optimization_dict.items() if k.startswith(child_prefix)}
                inject(child, child_optimization_dict, model_config, gguf_loader, child_prefix)

def del_meta(module:nn.Module):
    #print("default loading weights", prefix)
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        if param.device == "meta" or param.device == torch.device("meta"):
            module.__delattr__(name)
    for name, child in module._modules.items():
        del_meta(child)

def gen_optimize_config(module: nn.Module, out_data: Mapping, rule_list: List, prefix: str="", default_device: str = "cuda:0"):
    module_name = prefix[:-1]
    translated_name = translate_name_to_gguf(prefix)[:-1]
    #print("gen_optimize_config", prefix, module_name, translated_name)
    recursive = True
    for rule in rule_list:
        #print(rule)
        match_meta = rule["match"]
        if "class" in match_meta:
            import_path = match_meta["class"].split(".")
            import_module_name = ".".join(import_path[:-1])
            import_class_name = import_path[-1]
            module_cls=getattr(__import__(import_module_name, fromlist=[""]), import_class_name)
            if not isinstance(module, module_cls):
                continue
        if "name" in match_meta:
            if re.search(match_meta["name"], module_name) is None:
                continue
        replace_meta = rule["replace"]
        out_data[module_name]={"key": translated_name,
                               "class": replace_meta["class"],
                               "device": replace_meta["device"] if "device" in replace_meta else default_device,
                               "kwargs": replace_meta["kwargs"] if "kwargs" in replace_meta else dict()}
        if "recursive" in rule:
            recursive = bool(rule["recursive"])
            
    if module_name not in out_data:
        out_data[module_name]="default"

    #print(out_data[module_name])
    #input()

    if recursive:
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                gen_optimize_config(child, out_data, rule_list, child_prefix)
    

def optimize_and_load_gguf(module: nn.Module, rule_file: str, gguf_path: str, model_config: PretrainedConfig, default_device: str = "cuda:0"):
    with open(rule_file, 'r', encoding='utf-8') as f:
        rule_list = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    optimize_config = dict()
    gen_optimize_config(module, optimize_config, rule_list, default_device = default_device)
    
    gguf_loader=GGUFLoader(gguf_path)
    with torch.device("meta"):
        inject(module, optimize_config, model_config, gguf_loader)
    load_weights(module, gguf_loader)
    del_meta(module)
