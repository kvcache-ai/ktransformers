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
import argparse
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
    EvalPrediction,
)
import json
from pathlib import Path
from tqdm import tqdm
from torchviz import make_dot
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import load_weights, prefill_and_generate, prefill_and_generate_capture, get_compute_capability, xpu_fp16_model
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor
from ktransformers.sft.lora import inject_lora_layer, lora_and_load_adapter
from ktransformers.util.custom_loader import GGUFLoader, SafeTensorLoader
from ktransformers.util.globals import GLOBAL_CONFIG
from ktransformers.sft.metrics import ComputeSimilarity
from ktransformers.sft.monkey_patch_torch_module import install_patch, restore_patch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# for debug
def print_module_tree(module, indent=0):
    print(" " + f"{module.__class__.__name__}(training={module.training})")
    for name, child in module.named_children():
        print(" " + f"└─{name}: ", end="")
        print_module_tree(child, indent + 4)

# for debug
def write_to_file(content, file_path: str = 'ktransformers/mark_content.txt', mode: str = 'a', encoding: str = 'utf-8') -> None:
    """
    将字符串写入指定文件 
    :param content: 要写入的字符串内容 
    :param file_path: 目标文件路径 
    :param mode: 文件打开模式（默认'w'为覆盖写入，可选'a'追加写入）
    :param encoding: 文件编码（默认utf-8）
    """
    with open(file_path, mode, encoding=encoding) as f:
        f.write(content) 

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


def local_chat(
    model_path: str | None = None,
    model_config_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 1000,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = True, # modify to false if using KExpertsTorch
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
    chunk_size: int = 8192,
    device: str = "cuda",
    is_sft: bool = False,
    sft_data_path: str | None = None,
    save_adapter_path: str | None = None,
    use_adapter: bool = False,
    use_adapter_path: str | None = None,
    is_test_data: bool = False,
    test_data_path: str | None = None,
    output_dir: str | None = None,
):

    if not is_sft:
        torch.set_grad_enabled(False)
        
    if is_sft == True or use_adapter == True:
        GLOBAL_CONFIG._config["mod"] = "sft"
    else:
        GLOBAL_CONFIG._config["mod"] = "infer"

    Config().cpu_infer = cpu_infer
    Config().chunk_size = chunk_size
    if torch.xpu.is_available():
        use_cuda_graph = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if model_config_path == None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_config_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    elif xpu_fp16_model(config):
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
            if torch.xpu.is_available():
                config._attn_implementation = "eager"
            model = custom_models[config.architectures[0]](config)
        else:
            if torch.xpu.is_available():
                attn_implementation = "eager"
            else:
                attn_implementation = "flash_attention_2"
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation=attn_implementation
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
        
    GLOBAL_CONFIG._config["mod"] = "infer"
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)

    model.train()

    if is_sft == True:
        if use_adapter == True or is_test_data == True:
            raise AttributeError("We do not support to run sft and inference at the same time.")
        GLOBAL_CONFIG._config["mod"] = "sft"
        print(f"sft with lora in dataset: {sft_data_path} ...")
        print(f"use_cuda_graph:{use_cuda_graph}")
        lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path)

    if use_adapter == True:
        GLOBAL_CONFIG._config["mod"] = "sft"
        if is_sft == True:
            raise AttributeError("We do not support more than one adapter up to now...")
        
        if use_adapter_path.endswith('.gguf'):
            inject_lora_layer(model, use_adapter_path)
            adapter_gguf_loader = GGUFLoader(use_adapter_path)
            load_weights(model, adapter_gguf_loader, adapter_gguf=True)
            model.train()
        else:
            inject_lora_layer(model, use_adapter_path)
            
            adapter_loader = SafeTensorLoader(use_adapter_path)
            device = next(model.parameters()).device
            
            # for name, param in model.named_parameters():
            #     print(name, param.shape)

            for key in adapter_loader.tensor_file_map.keys():
                try:
                    tensor = adapter_loader.load_tensor(key, device=device)
                    
                    model_key = key.replace("base_model.model.", "")
                    model_key = model_key.replace(".weight", ".default.weight")
                    
                    param = model.get_parameter(model_key)
                    param.data.copy_(tensor.data)
                    
                    print(f"Loaded adapter weight: {key} -> {model_key}")
                except AttributeError as e:
                    print(f"Skipping {key}: not a model parameter")
                except KeyError as e:
                    print(f"Key not found in model: {model_key} (original: {key})")
            

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
    
    # @torch.no_grad()
    # def first_token_argmax_baseline(model, tokenizer, prompt_text, device):
    #     model.eval()
    #     enc = tokenizer.apply_chat_template([{"role":"user","content":prompt_text}],
    #                                         add_generation_prompt=True, return_tensors="pt")
    #     x = enc.to(device)
    #     logits = model(input_ids=x, use_cache=False, return_dict=False)[0]
    #     return int(torch.argmax(logits[:, -1, :], dim=-1)[0])

    # try:
    #     device_map = model.gguf_loader.tensor_device_map
    #     from ktransformers.util.utils import get_device, torch_device_mapping
    #     torch_device = get_device('model.layers.0.self_attn', device_map)
    #     torch_device = torch_device_mapping.get(torch_device, torch_device)
    #     print(f"[FIRST-TOKEN PROBE] argmax id = {probe_id} ({tokenizer.decode([probe_id])!r})")
    # except Exception as e:
    #     print("[FIRST-TOKEN PROBE] failed:", e)
    #     return

    system = platform.system()
    # for debug
    # if system == "Windows":
    #     os.system("cls")
    # else:
    #     os.system("clear")
    
    if GLOBAL_CONFIG._config["mod"] == "sft" :
        model.model.embed_tokens.to("cpu")
        
    if is_test_data:
        data_path = Path(test_data_path)
        with data_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        preds, refs = [], []

        for sample in tqdm(dataset, desc="Processing samples"):
            inst = sample.get("instruction", "")
            prompt = sample.get("input", "")
            prompt = prompt+inst
            # print(f"prompt: {prompt}")
            label = sample.get("output", "")
   
            messages = [{"role": "user", "content": prompt}]
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
                prediction = prefill_and_generate_capture(
                    model, tokenizer, input_tensor.to(device), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,
                    use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim, echo_stream=False
                )
            else:
                prediction = prefill_and_generate_capture(
                    model, tokenizer, input_tensor.to(device), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,echo_stream=False,
                )
            # print(f"prediction:{prediction}")
            sample["label"] = label
            sample["prediction"] = prediction
            sample.pop("output", None)

            preds.append(prediction)
            refs.append(label)

        pred_file = Path(output_dir) / 'predictions.json'
        pred_file.parent.mkdir(parents=True, exist_ok=True)
        
        with pred_file.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        compute_metrics = ComputeSimilarity(tokenizer)
        # print(f"metrics:{metrics}")
        
        enc_pred = tokenizer(preds, add_special_tokens=False, padding=True, return_tensors="np")
        enc_ref  = tokenizer(refs,  add_special_tokens=False, padding=True, return_tensors="np")

        ep = EvalPrediction(
            predictions=enc_pred["input_ids"],
            label_ids=enc_ref["input_ids"]
        )

        metrics = compute_metrics(ep, compute_result=True)

        metric_file = Path(output_dir) / 'metrics.json'
        with metric_file.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
            
        print(f"Results of predictions saved in {pred_file}")
        print(f"Results of metrics saved in {metric_file}")

    while not is_test_data:
        GLOBAL_CONFIG._config["mod"] = "infer"
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
            if prompt_file != None:
                content = open(prompt_file, "r").read()
            else:
                content = "Please write a piece of quicksort code in C++."
        elif os.path.isfile(content):
            content = open(content, "r").read()
            
        messages = [{"role": "user", "content": content}]
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
                model, tokenizer, input_tensor.to(device), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,
                use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
            )
        else:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.to(device), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,
            )


if __name__ == "__main__":
    install_patch()
    IS_DEBUG = True

    if IS_DEBUG == False:
        parser = argparse.ArgumentParser()

        parser.add_argument("--model_path", required=True)
        parser.add_argument("--model_config_path", default=None)
        parser.add_argument("--gguf_path", required=True)
        parser.add_argument("--cpu_infer", type=int, default=32)
        parser.add_argument("--max_new_tokens", type=int, default=1000)
        parser.add_argument("--force_think", action="store_true")
        parser.add_argument("--optimize_config_path", required=True)
        parser.add_argument("--is_sft", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--sft_data_path", default=None)
        parser.add_argument("--save_adapter_path", default=None)
        parser.add_argument("--use_adapter", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--use_adapter_path", default=None)
        parser.add_argument("--is_test_data", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--test_data_path", default=None)
        parser.add_argument("--output_dir", default=None)

        args = parser.parse_args()

        local_chat(
            model_path=args.model_path,
            model_config_path=args.model_config_path,
            gguf_path=args.gguf_path,
            cpu_infer=args.cpu_infer,
            max_new_tokens=args.max_new_tokens,
            force_think=args.force_think,
            optimize_config_path=args.optimize_config_path,
            is_sft=args.is_sft,
            sft_data_path=args.sft_data_path,
            save_adapter_path=args.save_adapter_path,
            use_adapter=args.use_adapter,
            use_adapter_path=args.use_adapter_path,
            is_test_data=args.is_test_data,
            test_data_path=args.test_data_path,
            output_dir= args.output_dir
        )

    else:
        local_chat(
            # model_path="/mnt/data/data/DeepSeek-V3-671B-BF16",
            # model_config_path="/mnt/data/data/DeepSeek-V3-671B-BF16",
            # gguf_path="/mnt/data/data/DeepSeek-V3-671B-BF16",
            model_path="/mnt/data/models/DeepSeek-V2-Lite-Chat",
            model_config_path="/mnt/data/models/DeepSeek-V2-Lite-Chat",
            gguf_path="/mnt/data/models/DeepSeek-V2-Lite-Chat",
            cpu_infer=32,
            max_new_tokens=1000,
            force_think=False,
            # optimize_config_path="ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml",
            optimize_config_path="ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml",
            is_sft=True,
            sft_data_path="test_adapter/western_train.json",
            # sft_data_path="test_adapter/western_train.json",
            # sft_data_path="test_adapter/500token_test.json",
            save_adapter_path="/mnt/data/lpl/test_adapter/Kwhl_test_py312_torch28_DeepSeekV2_WEST",
            use_adapter=False,
            use_adapter_path="/mnt/data/lpl/test_adapter/Kllama_deepseekV2_AfriMed_mcq",
            is_test_data=False,
            test_data_path="/home/lpl/LLaMA-Factory-KT/data/mcq_test.json",
            output_dir="/mnt/data/lpl/test_adapter/Kllama_deepseekV2_AfriMed_mcq/baselines",
        )
        