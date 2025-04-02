'''
Date: 2024-11-07 07:30:16
LastEditors: djw
LastEditTime: 2024-11-15 14:23:26
'''
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import yaml

import json
from typing import Optional

class ModelConfig:
    vocab_size: int = 32000
    n_layer: int = 1
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = 18944
    n_local_heads: int = 8
    head_dim: int = 128
    rope_base: float = 1000000.0
    norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    model_path: str
    gguf_path: str
    optimize_rule_path: str
    speculative_rule_path: str
            

    # quantize config
    quant_algorithm: Optional[str] = None
    quant_group_size: Optional[int] = None
    quant_num_bits: Optional[int] = None

    json_key_map = {
        "vocab_size": "vocab_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "dim": "hidden_size",
        "intermediate_size": "intermediate_size",
        "n_local_heads": "num_key_value_heads",
        "rope_base": "rope_theta",
        "norm_eps": "norm_eps",
        "rms_norm_eps": "rms_norm_eps",
        "hidden_act": "hidden_act",
    }

    def __init__(self, config):
        self.model_path = config["model"]["model_path"]
        self.gguf_path = config["model"]["gguf_path"]
        self.optimize_rule_path = config["model"]["optimize_rule_path"]
        if "speculative_rule_path" in config["model"]:
            self.speculative_rule_path =  config["model"]["speculative_rule_path"]
            self.speculative_gguf_path = config["model"]["speculative_gguf_path"]
            self.speculative_model_path = config["model"]["speculative_model_path"]
        self.quant_algorithm = config["model"]["quant"]["algorithm"]
        self.quant_group_size = config["model"]["quant"]["group_size"]
        self.quant_num_bits = config["model"]["quant"]["num_bits"]
        self.load_config()
        self.n_layer = config["model"]["n_layers"]

    def load_config(self):
        config_file = f"{self.model_path}/config.json"
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_file}")

        for attr, json_key in self.json_key_map.items():
            if json_key in config_data:
                setattr(self, attr, config_data[json_key])
            else:
                setattr(self, attr, getattr(self, attr))


    


class ParallelConfig:
    def __init__(
        self,
        config,
    ) -> None:
        self.pipeline_parallel_size = config["parallel"]["pp"]
        self.tensor_parallel_size = config["parallel"]["tp"]
        self.disable_custom_all_reduce = config["parallel"]["disable_custom_all_reduce"]
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size

class AttnConfig:
    page_size: int = 256
    block_num: int = 32
    max_batch_token : int = 256
    max_batch_size: int = 32

    def __init__(self, config):
        self.page_size = config["attn"]["page_size"]
        self.block_num = config["attn"]["block_num"]
        self.max_batch_token = config["attn"]["max_batch_token"]
        self.max_batch_size = config["attn"]["max_batch_size"]


class SamplerConfig():
	# Batched sampling params
    temperatures: float
    is_all_greedy: bool
	
    def __init__(self, config):
        self.temperatures = config["sample"]["temperature"]
        self.is_all_greedy = True


def load_yaml_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    



class LLMConfig:
    model_config: ModelConfig
    parallel_config: ParallelConfig
    attn_config: AttnConfig
    sample_config: SamplerConfig
    config_file: str

    def __init__(self, config_file):
        self.config_file = config_file
        config = load_yaml_config(config_file)
        self.model_config = ModelConfig(config)
        self.parallel_config = ParallelConfig(config)
        self.attn_config = AttnConfig(config)
        self.sample_config = SamplerConfig(config)

