"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist

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
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate, get_compute_capability, xpu_fp16_model
from ktransformers.util.ascend.ascend_utils import get_absort_weight, setup_model_parallel, get_tensor_parallel_group
from ktransformers.util import utils, npu_graph_runner
from ktransformers.models.custom_cache import StaticCache
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}

try:
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
except:
    pass

import sys, signal, faulthandler
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)


def local_chat(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 1000,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = True,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
    chunk_size: int = 8192,
    device: str = "cuda",
    tp: int = 1,
):
    Config().cpu_infer = cpu_infer

    local_rank, world_size = setup_model_parallel(tp=tp)
    torch.set_grad_enabled(False)
    if utils.CUR_DEVICE is None:
        utils.CUR_DEVICE = f"npu:{torch.npu.current_device()}"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.chunk_size = chunk_size
    npu_graph_runner.LAYER_ID = config.num_hidden_layers
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0]) if local_rank == 0 else None
            optimize_config_path = default_optimize_rules[config.architectures[0]]
            print(f'{optimize_config_path=}') if local_rank == 0 else None
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    # 提前absorbed
    get_absort_weight(model, config)

    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if system == "Windows":
        os.system("cls") if local_rank == 0 else None
    else:
        os.system("clear") if local_rank == 0 else None

    print(f"{model=}") if local_rank == 0 else None

    batch_size, seq_length = 1, 16384  # default cache pool params
    device_map = model.gguf_loader.tensor_device_map
    static_cache = StaticCache(
        config = model.config, max_batch_size = batch_size, max_cache_len = seq_length + max_new_tokens, device = device_map, dtype = model.dtype
    )

    torch.distributed.barrier()
    while True:
        if local_rank == 0:
            try:
                content = input("Chat: \n").strip()
            except KeyboardInterrupt:
                dist.barrier()
                print('Exit all rank with KeyboardInterrupt!')
                sys.exit(0)
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
                if prompt_file != None:
                    content = open(prompt_file, "r").read()
                else:
                    continue
            elif os.path.isfile(content):
                f = open(content, "r")
                content = f.readlines()
                f.close()
            else:
                content = [f"{len(content)},{max_new_tokens},{content}"]
        else:
            content = [""]

        for line in content:
            content_tensor = torch.tensor(bytearray(line.encode()), dtype=torch.uint8).to(device=utils.CUR_DEVICE)
            if world_size > 1:
                content_size = torch.tensor(len(content_tensor), dtype=torch.int64).to(device=utils.CUR_DEVICE)
                all_content_sizes = [torch.zeros((1,), dtype=torch.int64).to(device=utils.CUR_DEVICE) for _ in range(world_size)]
                dist.all_gather(all_content_sizes, content_size)
                max_content_size = max([size.item() for size in all_content_sizes])

                padded_content_tensor = torch.zeros((max_content_size,), dtype=torch.uint8).to(device=utils.CUR_DEVICE)
                padded_content_tensor[:len(content_tensor)] = content_tensor

                all_content_tensors = [torch.zeros((max_content_size,), dtype=torch.uint8).to(device=utils.CUR_DEVICE) for _ in range(world_size)]
                dist.all_gather(all_content_tensors, padded_content_tensor)
                content_tensor = all_content_tensors[0][:all_content_sizes[0].item()]
                line = bytes(content_tensor.cpu().numpy()).decode()

            parts = line.split(",")
            input_tokens = int(parts[0])
            max_new_tokens = int(parts[1])
            line = line[line.index(",", line.index(",") + 1) + 1:]
            
            messages = [{"role": "user", "content": line}]
            input_tensor = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if force_think:
                token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
                input_tensor = torch.cat(
                    [input_tensor, token_thinks], dim=1
                )
            if mode == 'long_context':
                assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
                "please change max_seq_len in  ~/.ktransformers/config.yaml"

            if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8 and device_manager.gpu_vendor == GPUVendor.NVIDIA:
                generated = prefill_and_generate(
                    model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,
                    use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim,
                    static_cache=static_cache
                )
            else:
                generated = prefill_and_generate(
                    model, tokenizer, input_tensor.to(device=utils.CUR_DEVICE), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,
                    static_cache=static_cache
                )


if __name__ == "__main__":
    fire.Fire(local_chat)
