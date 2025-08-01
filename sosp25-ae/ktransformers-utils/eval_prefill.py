import os
import random
import time
import json
import psutil
from tqdm import tqdm
from datasets import load_dataset

import torch
import fire
from transformers import AutoTokenizer, AutoConfig
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.server.config.config import Config
from ktransformers.util.utils import get_device

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
}

def eval_prefill(
    model_path: str,
    optimize_rule_path: str,
    config_id: str,
    seed: int = 2025,
    prompt_length_list: list = [32,64,128,256,512,1024,2048,4096,8192],
    num_trials: int = 3,
    use_cuda_graph: bool = True,
    output_file: str = "prefill_perf.jsonl"
):
    random.seed(seed)
    torch.set_grad_enabled(False)

    Config().chunk_size = max(prompt_length_list)
    Config().cpu_infer = psutil.cpu_count(logical=False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if "Qwen2Moe" in config.architectures[0]:
            config._attn_implementation = "sdpa"
        model = custom_models[config.architectures[0]](config)

    gguf_path = model_path
    optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)
    model.eval()

    device_map = model.gguf_loader.tensor_device_map
    input_device = get_device('blk.0.self_attn', device_map)
    input_device = "cuda:0" if input_device == "cuda" else input_device

    # USE WIKITEXT-2-RAW-V1 AS TEST DATASET
    test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_input_ids = tokenizer("".join(test_dataset['text']), return_tensors='pt').input_ids.to(input_device)

    # ========== Warmup ==========
    start_idx = random.randint(0, len(test_input_ids[0]) - prompt_length_list[-1])
    prompt = test_input_ids[:, start_idx:start_idx+prompt_length_list[-1]]
    _ = model(prompt, use_cache=False)

    # ========== Evaluation ==========
    results = {}
    for prompt_length in prompt_length_list:
        times = []
        for _ in tqdm(range(num_trials), desc=f"Eval Prefill: Prompt Length = {prompt_length}"):
            start_idx = random.randint(0, len(test_input_ids[0]) - prompt_length)
            prompt = test_input_ids[:, start_idx:start_idx+prompt_length]

            start = time.perf_counter()
            _ = model(prompt, use_cache=False)
            end = time.perf_counter()

            elapsed_time = end - start
            tokens_per_sec = prompt_length / elapsed_time
            times.append(tokens_per_sec)

        results[prompt_length] = round(sum(times) / len(times), 2)

    current_record = {
        "config_id": config_id,
        "results": results,
    }

    # dump result
    with open(output_file, "a") as f:
        f.write(json.dumps(current_record, separators=(",", ":")) + "\n")

    print(f"✅ Results saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(eval_prefill)