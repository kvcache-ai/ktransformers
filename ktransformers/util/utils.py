#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from torch import nn
import itertools
import time
import enum
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.operators import base_operator
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        if hasattr(cur_mod, s):
            cur_mod = getattr(cur_mod, s)
        else: # nn.ModuleList or nn.ModuleList
            cur_mod=cur_mod[int(s)]
    if hasattr(cur_mod, tokens[-1]):
        setattr(cur_mod, tokens[-1], module)
    else: # nn.ModuleList or nn.ModuleList
        cur_mod[int(tokens[-1])] = module

def set_param(module: nn.Module, name: str, weights: torch.Tensor):
    
    param=nn.parameter.Parameter(weights, requires_grad=False)
    if isinstance(module, nn.Linear) and len(weights.shape)==1:
        param.unsqueeze_(0)
    setattr(module, name, param)

def load_cur_state_dict(module: nn.Module, gguf_loader: GGUFLoader, prefix: str = ""):
    prefix = prefix.replace("orig_module.", "")
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        key = prefix + name
        translated_key = translate_name_to_gguf(key)
        print("default loading weights", key, translated_key)
        if translated_key in gguf_loader.tensor_file_map:
            target_dtype = torch.get_default_dtype()
            device = "cpu" if "embd" in translated_key else "cuda"
            weights = gguf_loader.load_gguf_tensor(translated_key, device = device).to(dtype = target_dtype)
            set_param(module, name, weights)
            del weights
        else:
            #print(load_config.tensor_file_map.keys())
            raise Exception(f"can't fand {translated_key} in GGUF file!")
        
def load_weights(module:nn.Module, gguf_loader:GGUFLoader, prefix='', return_when_injected:bool = False, only_load_injected:bool = False):
    # print(f"recursively loading weights {prefix},{return_when_injected=}, {only_load_injected=}")
    if not isinstance(module, base_operator.BaseInjectedModule):
        load_cur_state_dict(module, gguf_loader, prefix)
        for name, child in module._modules.items():
            load_weights(child, gguf_loader, prefix+name+".")
    else:
        module.load()
    
def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape
    torch_device = inputs.device
    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values):
        logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        past_key_values.change_seq_length(1)
        """
        with torch.cuda.stream(custom_stream):
            logits=model(cur_token, 
                         position_ids=position_ids,
                         cache_position=cache_position,
                         past_key_values=past_key_values,
                         return_dict=False, use_cache=True)[0]
        #"""            
        torch.cuda.synchronize()
        #print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token

    with torch.no_grad():
        stream = TextStreamer(tokenizer)
        past_key_values = StaticCache(
            config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = torch_device, dtype = model.dtype
        )
        cache_position = torch.arange(seq_length, device=torch_device)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        past_key_values.cur_idx=cache_position
        start_time = time.time()
        #custom_stream = torch.cuda.Stream()

        inputs_embeds = model.model.embed_tokens(inputs.to("cpu")).to("cuda")
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone()
        generation_config, model_kwargs = model._prepare_generation_config(
            None, max_length=max_new_tokens,
            do_sample=True, top_k=5, top_p=0.85, temperature=0.1 # change this to modify generate config
        )
        try: # transformers==4.43
            logits_warper = (
                model._get_logits_warper(generation_config,device=inputs.device) if generation_config.do_sample else None
            )
        except: 
            logits_warper = (
                model._get_logits_warper(generation_config) if generation_config.do_sample else None
            )
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        first_token_time = time.time() - start_time

        prefill_count = seq_length
        prefill_time = first_token_time

        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(next_token)
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=torch_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1

        cuda_graph_runner = CUDAGraphRunner()
        cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, return_dict=False, use_cache=True)
        start_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id:
                print(stream.end(), end="", flush=True)
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)

    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print("")

    print(f"prompt eval count:    {prefill_count} token(s)")
    print(f"prompt eval duration: {prefill_time}s")
    print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    print(f"eval count:           {tokens_generated} token(s)")
    print(f"eval duration:        {total_time}s")
    print(f"eval rate:            {tokens_per_second} tokens/s")

    return tokens

class InferenceState(enum.Enum):
    UNLOAD = 0
    PREFILL = 1
    GENERATE = 2
    RESTORE = 3
