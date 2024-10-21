"""
Description  :
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
"""

import asyncio
import os
import platform
import sys

from ktransformers.server.args import ArgumentParser

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.server.config.config import Config

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def local_chat():
    config = Config()
    arg_parser = ArgumentParser(config)
    # 初始化消息
    arg_parser.parse_args()
    if config.backend_type == "transformers":
        from ktransformers.server.backend.interfaces.transformers import TransformersInterface as BackendInterface
    elif config.backend_type == "exllamav2":
        from ktransformers.server.backend.interfaces.exllamav2 import ExllamaInterface as BackendInterface
    elif config.backend_type == "ktransformers":
        from ktransformers.server.backend.interfaces.ktransformers import KTransformersInterface as BackendInterface
    else:
        raise NotImplementedError(f"{config.backend_type} not implemented")
    interface = BackendInterface(config)

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    # add a history chat content
    his_content = []
    while True:
        content = input("Chat: ")
        if content.startswith('"""'):  # prefix """
            # multi lines input
            content = content[3:] + "\n"
            while True:
                line = input("")
                if line.endswith('"""'):
                    # end multi lines input
                    line = line[:-3]  # suffix """
                    if line:
                        content += line + "\n"
                    break
                else:
                    content += line + "\n"
        if content == "":
            if config.prompt_file == None or config.prompt_file == "":
                content = "Please write a piece of quicksort code in C++."
            else:
                content = open(config.prompt_file, "r").read()
        elif os.path.isfile(content):
            content = open(content, "r").read()
        messages = his_content + [{"role": "user", "content": content}]

        async def async_inference(messages):
            generated = ""
            async for token in interface.inference(messages, "local_chat"):
                generated += token
            return generated

        generated = asyncio.run(async_inference(messages))
        his_content += [
            {"role": "user", "content": content},
            {"role": "assitant", "content": generated},
        ]


if __name__ == "__main__":
    local_chat()
