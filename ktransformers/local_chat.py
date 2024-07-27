# Copyright 2024 Shaoyuan Chen
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
import platform
import sys
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.util.utils import prefill_and_generate
from ktransformers.server.config.config import Config

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
}

ktransformer_rules_dir = os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
default_optimize_rules ={
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
}

def local_chat(
    model_path: str,
    optimize_rule_path: str = None,
    gguf_path: str = None,
    max_new_tokens: int = 1000,
    cpu_infer: int = Config().cpu_infer
):
    torch.set_grad_enabled(False)
    
    Config().cpu_infer = cpu_infer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if "Qwen2Moe" in config.architectures[0]: # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_rule_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_rule_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_rule_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)

    model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if (system == u'Windows'):
        os.system('cls')
    else:
        os.system('clear')

    while True:
        content = input("Chat: ")
        # if content is num
        if content == "":
            content = "Please write a piece of quicksort code in C++." 

        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        torch.set_default_dtype(torch.bfloat16) # TODO: Remove this, replace dtype using config
        generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens)

if __name__ == "__main__":
    fire.Fire(local_chat)
