'''
Description  :  
Author       : Boxin Zhang, Azure-Tang
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
from ktransformers.util.custom_loader import GGUFLoader, ModelLoaderFactory
from ktransformers.util.utils import set_module, load_weights
import itertools
import copy

def inject(module, local_optimization_dict, model_config:AutoConfig ,gguf_loader:GGUFLoader, prefix=''):
    for name, child in module._modules.items():
        if child is not None:
            child_prefix = prefix + name
            if child_prefix in local_optimization_dict:
                inject_module_meta=local_optimization_dict[child_prefix]
                if inject_module_meta["class"] != "default":
                    import_path = inject_module_meta["class"].split(".")
                    import_module_name = ".".join(import_path[:-1])
                    gguf_loader.tensor_device_map[inject_module_meta["key"]] = inject_module_meta["kwargs"] if "kwargs" in inject_module_meta else dict()
                    import_class_name = import_path[-1]
                    module_cls=getattr(__import__(import_module_name, fromlist=[""]), import_class_name)
                    print(f"Injecting {child_prefix} as", import_module_name, ".", import_class_name)
                    inject_module=module_cls(key = inject_module_meta["key"], gguf_loader = gguf_loader, config = model_config, orig_module=child, **inject_module_meta["kwargs"])
                    set_module(module, name, inject_module)
                elif inject_module_meta["class"] == "default":
                    print(f"Injecting {child_prefix} as default")
                    gguf_loader.tensor_device_map[inject_module_meta["key"]] = inject_module_meta["kwargs"] if "kwargs" in inject_module_meta else dict()
                else:
                    raise Exception("inject_module_meta[\"class\"] must be \"default\" or a class path")
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
    # translated_name = translate_name_to_gguf(prefix)[:-1]
    #print("gen_optimize_config", prefix, module_name, translated_name)
    recursive = True
    for rule in rule_list:
        match_meta = rule["match"]
        if "class" not in match_meta and "name" not in match_meta:
            raise Exception("match must have at least one of \"class\" and \"name\"")
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
        if "replace" not in rule:
            raise Exception("replace must be in rule")
        if "replace" in rule:
            replace_meta = rule["replace"]
            if module_name not in out_data:
                out_data[module_name]={"key": module_name,
                                    "class": replace_meta["class"] if "class" in replace_meta else "default",
                                    # "device": replace_meta["device"] if "device" in replace_meta else default_device,
                                    "kwargs": copy.deepcopy(replace_meta["kwargs"]) if "kwargs" in replace_meta else dict()}
            else:
                if out_data[module_name]["class"] == "default":
                    out_data[module_name]["class"] = replace_meta["class"] if "class" in replace_meta else "default"
                out_data[module_name]["kwargs"].update(copy.deepcopy(replace_meta["kwargs"]) if "kwargs" in replace_meta else dict())
        if "recursive" in rule:
            recursive = bool(rule["recursive"])
        break
            
    if module_name not in out_data:
        out_data[module_name]= {
            "class": "default",
            "key": module_name,
            "kwargs": {"generate_device": default_device,
                       "prefill_device": default_device}
        }

    #print(out_data[module_name])
    #input()

    if recursive:
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                gen_optimize_config(child, out_data, rule_list, child_prefix, default_device = default_device)
    

def translate_model_config(model_config: PretrainedConfig):
    # for supporting some special model 
    if model_config.model_type == "mixtral":
        model_config.moe_intermediate_size = model_config.intermediate_size
    
    return model_config


def optimize_and_load_gguf(module: nn.Module, rule_file: str, gguf_path: str, model_config: PretrainedConfig, default_device: str = "cuda:0"):
    with open(rule_file, 'r', encoding='utf-8') as f:
        rule_list = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    optimize_config = dict()
    gen_optimize_config(module, optimize_config, rule_list, default_device = default_device)
    
    model_config = translate_model_config(model_config)

    weights_loader = ModelLoaderFactory.create_loader(gguf_path)
    with torch.device("meta"):
        inject(module, optimize_config, model_config, weights_loader)
    # pre load lm_head because its big inter result
    load_weights(module.lm_head, weights_loader, "lm_head.", device=default_device)
    load_weights(module, weights_loader, device=default_device)
    module.gguf_loader = weights_loader
    del_meta(module)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
