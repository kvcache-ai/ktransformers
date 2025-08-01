import os
import random
import time
import json
import psutil
from tqdm import tqdm
from datasets import load_dataset

from ktransformers.operators.flashinfer_wrapper import MLAWrapperSingleton
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
)
import fire
from ktransformers.models.custom_cache import StaticCache
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

from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.utils import get_all_used_cuda_device, get_device

def eval_decode(
    model_path: str,
    optimize_rule_path: str,
    config_id: str,
    seed: int = 2025,
    prompt_length: int = 32,
    decode_length: int = 512,
    num_trials: int = 3,
    use_cuda_graph: bool = True,
    output_file: str = "decode_perf.jsonl"
):
    random.seed(seed)
    torch.set_grad_enabled(False)

    Config().chunk_size = prompt_length
    Config().cpu_infer = min(64, psutil.cpu_count(logical=False))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)

    use_flashinfer_mla = "Deepseek" in config.architectures[0]
    with torch.device("meta"):
        if "Qwen2Moe" in config.architectures[0]:
            config._attn_implementation = "sdpa"
        model = custom_models[config.architectures[0]](config)

    gguf_path = model_path
    optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)
    model.eval()

    # USE WIKITEXT-2-RAW-V1 AS TEST DATASET
    test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_input_ids = tokenizer("".join(test_dataset['text']), return_tensors='pt').input_ids

    device_map = model.gguf_loader.tensor_device_map
    input_device = get_device('blk.0.self_attn', device_map)
    input_device = "cuda:0" if input_device == "cuda" else input_device
    all_cuda_device = get_all_used_cuda_device(device_map)

    def decode_one_token(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values):
        if use_flashinfer_mla:
            MLAWrapperSingleton.plan_all(None,None,None,position_ids.squeeze(1)+1,None,
                                            config.num_attention_heads, config.kv_lora_rank, config.qk_rope_head_dim, past_key_values.page_size,
                                            model.model.layers[0].self_attn.softmax_scale, torch.bfloat16, torch.bfloat16)
        if cuda_graph_runner is not None:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            torch.cuda.set_device(input_device)
            inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(input_device)
            logits=model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=False, use_cache=True)[0]
        if past_key_values != None:
            past_key_values.change_seq_length(1)
        for device in all_cuda_device:
            torch.cuda.synchronize(device)
        return torch.argmax(logits[:, -1, :], dim=-1)
    
    torch.cuda.set_device(input_device)
        
    past_key_values = StaticCache(
        config = model.config, max_batch_size = 1, max_cache_len = prompt_length + decode_length, device = device_map, dtype = model.dtype
    )
    if use_flashinfer_mla:
        MLAWrapperSingleton.update_buffer(past_key_values.max_pages)
        MLAWrapperSingleton.need_plan_all()

    cuda_graph_runner = None
    
    # ========== Warmup ==========
    cur_token = torch.tensor([[0]], device=input_device, dtype=torch.long)
    cache_position = torch.tensor([0], device=input_device, dtype=torch.int32)
    position_ids = cache_position.unsqueeze(0)
    _ = decode_one_token(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values).to(input_device)
    if use_cuda_graph:
        cuda_graph_runner = CUDAGraphRunner()
        cuda_graph_runner.capture(model, cur_token, position_ids, cache_position, past_key_values, input_device, return_dict=False, use_cache=True)
        _ = decode_one_token(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values).to(input_device)
    
    # ========== Evaluation ==========
    times = []
    for _ in tqdm(range(num_trials), desc=f"Eval Decode: Prompt Length = {prompt_length}, Decode Length = {decode_length}"):
        past_key_values.reset()
        if use_flashinfer_mla:
            MLAWrapperSingleton.reset_buffer()
        start_idx = random.randint(0, len(test_input_ids[0]) - prompt_length)
        prompt = test_input_ids[:, start_idx:start_idx+prompt_length].to(input_device)
        inputs_embeds = model.model.embed_tokens(prompt.to("cpu")).to(input_device)
        cache_position = torch.arange(prompt_length, device=input_device, dtype=torch.int32)
        logits = model(inputs_embeds=inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False,use_cache=True)[0]
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        
        start = time.perf_counter()
        for i in range(decode_length):
            cur_token = next_token.unsqueeze(0)
            cache_position = torch.tensor([prompt_length + i], device=input_device, dtype=torch.int32)
            position_ids = cache_position.unsqueeze(0)
            next_token = decode_one_token(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values).to(input_device)
        end = time.perf_counter()

        elapsed_time = end - start
        tokens_per_sec = decode_length / elapsed_time
        times.append(tokens_per_sec)

    result = round(sum(times) / len(times), 2)
    current_record = {
        "config_id": config_id,
        "result": result,
    }

    # dump result
    with open(output_file, "a") as f:
        f.write(json.dumps(current_record, separators=(",", ":")) + "\n")
        
    print(f"✅ Results saved to {output_file}")

if __name__ == "__main__":
    
    fire.Fire(eval_decode)