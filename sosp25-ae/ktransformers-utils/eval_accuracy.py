import json
import os
from tqdm import tqdm

from ktransformers.operators.flashinfer_wrapper import MLAWrapperSingleton
import torch
from torch import nn
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
from ktransformers.util.utils import get_device, tf_logits_warper
    
class HumanEvalCriteria:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(
        self, tokens
    ) -> bool:
        if len(tokens) >= 2 and self.tokenizer.decode(tokens[-2]).endswith('\n') and not self.tokenizer.decode(tokens[-1]).startswith(' ') and not self.tokenizer.decode(tokens[-1]).startswith('\n'):
            tokens[-1] = self.tokenizer.eos_token_id
            return True
        return False
        
class MBPPCriteria:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(
        self, tokens
    ) -> bool:
        if len(tokens) >= 3 and self.tokenizer.decode(tokens[-3]).startswith('[') and self.tokenizer.decode(tokens[-2]).endswith('ONE') and self.tokenizer.decode(tokens[-1]).startswith(']'):
            return True
        return False
    
class GSM8KCriteria:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(
        self, tokens
    ) -> bool:
        if len(tokens) >= 2 and self.tokenizer.decode(tokens[-2]).endswith('\n') and self.tokenizer.decode(tokens[-1]).startswith('Question'):
            tokens[-1] = self.tokenizer.eos_token_id
            return True
        return False


custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
}

from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.utils import get_all_used_cuda_device, get_device

def eval_accuracy(
    model_path: str,
    optimize_rule_path: str,
    task_name: str,
    seed: int = 2025,
    max_new_tokens: int = 1024,
    do_sample = True,
    temperature: float = 0.3,
    top_p: float = 0.95,
    num_trials: int = 10,
    output_file: str = "accuracy_result.jsonl",
):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    Config().chunk_size = 8192

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

    input_file = os.path.join(os.path.dirname(__file__), 'data', task_name, 'input.jsonl')
    with open(input_file) as f:
        prompt_lst = [
            {**json.loads(line.strip()), "idx": idx}
            for idx, line in enumerate(f)
        ]

    if task_name == "mbpp":
        stopping_criteria = MBPPCriteria(tokenizer)
    elif task_name == "human_eval":
        stopping_criteria = HumanEvalCriteria(tokenizer)
    else:
        stopping_criteria = GSM8KCriteria(tokenizer)

    def generate(model, tokenizer, inputs):
        _, seq_length = inputs.shape
        device_map = model.gguf_loader.tensor_device_map
        input_device = get_device('blk.0.self_attn', device_map)
        input_device = "cuda:0" if input_device == "cuda" else input_device
        all_cuda_device = get_all_used_cuda_device(device_map)

        def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config):
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
            next_token_scores = logits_warper(inputs, logits[:, -1, :])
            if generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_scores, dim=-1)
            return next_token
        
        torch.cuda.set_device(input_device)
            
        past_key_values = StaticCache(
            config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = device_map, dtype = model.dtype
        )
        if use_flashinfer_mla:
            MLAWrapperSingleton.update_buffer(past_key_values.max_pages)
            MLAWrapperSingleton.need_plan_all()

        generation_config, _ = model._prepare_generation_config(
            None, do_sample=do_sample, top_p=top_p, temperature=temperature
        )
        logits_warper = (
            tf_logits_warper(generation_config)
        )

        cache_position = torch.arange(seq_length, device=input_device, dtype=torch.int32)
        inputs_embeds = model.model.embed_tokens(inputs.to("cpu")).to(input_device)
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0].to(input_device)

        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        
        tokens = [int(next_token)]
        cache_position = torch.tensor([seq_length], device=input_device, dtype=torch.int32)
        position_ids = cache_position.unsqueeze(0)
        
        cuda_graph_runner = None

        for i in range(1, max_new_tokens):
            if i == 2:
                cuda_graph_runner = CUDAGraphRunner()
                cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, input_device, return_dict=False, use_cache=True)
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, generation_config).to(input_device)
            tokens.append(int(next_token))
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token.tolist()) == '<|im_end|>' or tokenizer.decode(next_token.tolist()) == '<|endoftext|>':
                break
            if stopping_criteria is not None and stopping_criteria(tokens):
                break
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)

        return tokens

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for trial in range(num_trials):
        for prompt in tqdm(prompt_lst, desc=f"Eval Accuracy: {task_name} Trial {trial + 1}/{num_trials}"):
            prompt_text = f"{prompt['instructions']}"
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
            output_tokens: list = generate(model, tokenizer, input_ids)
            generation_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            json_str = json.dumps(
                {
                    "idx": prompt["idx"],
                    "completion": generation_text.strip(),
                }
            )
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json_str + "\n")

    print(f"✅ Results saved to {output_file}")

if __name__ == "__main__":
    
    fire.Fire(eval_accuracy)