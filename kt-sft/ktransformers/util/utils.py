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
from typing import Any, List, Optional, Set
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
)
from torchviz import make_dot
# from ktransformers.sft.peft_utils.lora_layer import KTransformersLinearLora
from ktransformers.util.custom_loader import ModelLoaderFactory, ModelLoader, SafeTensorLoader, GGUFLoader, translate_name_to_gguf, translate_adapter_name_to_gguf
from ktransformers.operators import base_operator
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer
from ktransformers.util.globals import GLOBAL_CONFIG
if not torch.xpu.is_available():
    from ktransformers.operators.flashinfer_wrapper import MLAWrapperSingleton
import socket

from transformers.generation.logits_process import LogitsProcessor
# from transformers import TextStreamer # !!! this will override the TextStreamer from ktransformers.util.textstream

class NoEosUntil(LogitsProcessor):
    def __init__(self, prompt_len: int, min_gen_len: int, eos_ids):
        super().__init__()
        self.start_len = int(prompt_len)
        self.min_len   = self.start_len + int(min_gen_len)
        self.eos_ids   = list(eos_ids) if isinstance(eos_ids,(list,tuple)) else [int(eos_ids)]

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] < self.min_len:
            scores[..., self.eos_ids] = -float("inf")
        return scores

class SilentCaptureStreamer(TextStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self._buf: List[str] = []

    def _append_piece(self, piece: Optional[str]):
        if piece:
            self._buf.append(piece)

    def put(self, value) -> str:
        tokens: List[int] = []
        if isinstance(value, int):
            tokens = [value]
        else:
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    tokens = list(map(int, value.view(-1).tolist()))
                elif isinstance(value, (list, tuple)) and all(isinstance(x, int) for x in value):
                    tokens = list(value)
                else:
                    raise ValueError("Unsupported value type for SilentCaptureStreamer.put")
            except Exception:
                if isinstance(value, (list, tuple)) and all(isinstance(x, int) for x in value):
                    tokens = list(value)
                else:
                    raise ValueError("Unsupported value type for SilentCaptureStreamer.put")
        for t in tokens:
            piece = super().put(t)
            self._append_piece(piece)
        return ""

    def end(self) -> str:
        piece = super().end()
        self._append_piece(piece)
        return ""

    def getvalue(self) -> str:
        return "".join(self._buf)

    def clear(self):
        self._buf.clear()

warm_uped = False

def get_free_ports(n: int, continue_prot: list):
    sockets = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0)) 
        port = s.getsockname()[1]
        if port in continue_prot:
            s.close()
            continue
        ports.append(port)
        sockets.append(s)
    for s in sockets:
        s.close()
    return ports

def get_compute_capability(device:torch.device = None):
    if torch.cuda.is_available():
        if device is None:
            num_gpus = torch.cuda.device_count()
            min_compute_capability_major = 100
            for gpu_id in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(gpu_id)
                min_compute_capability_major = min(min_compute_capability_major, gpu_props.major)
            return min_compute_capability_major
        else:
            return torch.cuda.get_device_properties(device)
    else:
        return 0

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
    
    param=nn.parameter.Parameter(weights, requires_grad=True)
    if isinstance(module, nn.Linear) and len(weights.shape)==1:
        param.unsqueeze_(0)
    setattr(module, name, param)

def get_device(gguf_module_key:str, device_map:dict):
    if gguf_module_key in device_map:
        return device_map[gguf_module_key]["generate_device"]
    elif gguf_module_key.replace("model.layers", "blk") in device_map:
        return device_map[gguf_module_key.replace("model.layer", "blk")]["generate_device"]
    else:
        return "cuda"

def get_all_used_cuda_device(device_map:dict):
    all_device_list = set()
    for key in device_map:
        all_device_list.add(device_map[key]["generate_device"]) if "generate_device" in device_map[key] else None
        all_device_list.add(device_map[key]["prefill_device"]) if "prefill_device" in device_map[key] else None
    if "cpu" in all_device_list:
        all_device_list.remove("cpu")
    all_device_list = list(all_device_list)
    return all_device_list

def load_cur_state_dict(module: nn.Module, gguf_loader: ModelLoader, prefix: str = "", device="cuda", adapter_gguf: bool = False):
    if GLOBAL_CONFIG._config["mod"] == 'sft':
        prefix = prefix.replace("orig_module.", "")
        persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
        local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            translated_key = translate_name_to_gguf(key)
            if adapter_gguf == True:
                translated_adapter_key = translate_adapter_name_to_gguf(key)

            # TODO: Merge all loader.
            # I know this is ugly but lets do it for now.
            if gguf_loader.safetensor_loader is not None:
                load_dequantized_tensor = gguf_loader.safetensor_loader.load_dequantized_tensor
                tensor_file_map = gguf_loader.safetensor_loader.tensor_file_map
            else:
                load_dequantized_tensor = gguf_loader.load_gguf_tensor
                tensor_file_map = gguf_loader.tensor_file_map
            # print(f"tensor_file_map:{tensor_file_map}")
            # We allow some key not be used in GGUF
            if translated_key in tensor_file_map:
                target_dtype = torch.get_default_dtype()
                device = get_device(translated_key[:translated_key.rfind(".")], gguf_loader.tensor_device_map)
                print(f"loading {translated_key} to {device}")
                torch.cuda.empty_cache()
                weights = load_dequantized_tensor(translated_key, device=device).to(dtype=target_dtype)
                set_param(module, name, weights)
                del weights
            else:
                if adapter_gguf == True: # Not all module should be reload in lora adapter
                    for single_tensor_file_map in tensor_file_map:
                        if translated_adapter_key in single_tensor_file_map:
                            target_dtype = torch.get_default_dtype()
                            device = get_device(single_tensor_file_map[:single_tensor_file_map.rfind(".")], gguf_loader.tensor_device_map)
                            print(f"loading {single_tensor_file_map} to {device}")
                            torch.cuda.empty_cache()
                            weights = load_dequantized_tensor(single_tensor_file_map, device=device).to(dtype=target_dtype)
                            set_param(module, name, weights)
                            del weights

                else:
                    #print(load_config.tensor_file_map.keys())
                    raise Exception(f"can't find {translated_key} in GGUF file!")
    elif GLOBAL_CONFIG._config["mod"] == 'infer':
        prefix = prefix.replace("orig_module.", "")
        persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
        local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            translated_key = key
            
            # TODO: Merge all loader.
            # I know this is ugly but lets do it for now.
            if isinstance(gguf_loader, SafeTensorLoader):
                load_dequantized_tensor = gguf_loader.load_dequantized_tensor
            else:
                load_dequantized_tensor = gguf_loader.load_gguf_tensor
                tensor_file_map = gguf_loader.tensor_file_map
            
            if gguf_loader.has_tensor(translated_key) or "kv_b_proj" in translated_key:
                target_dtype = torch.get_default_dtype()
                device = get_device(translated_key[:translated_key.rfind(".")], gguf_loader.tensor_device_map)
                print(f"loading {translated_key} to {device}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.xpu.is_available():
                    torch.xpu.empty_cache()
                if "kv_b_proj" in translated_key and not gguf_loader.has_tensor(translated_key):
                    attn_k_b = load_dequantized_tensor(translated_key.replace("self_attn.kv_b_proj", "attn_k_b"), device=device).to(dtype=target_dtype)
                    attn_k_b = attn_k_b.transpose(1, 2).contiguous()
                    attn_v_b = load_dequantized_tensor(translated_key.replace("self_attn.kv_b_proj", "attn_v_b"), device=device).to(dtype=target_dtype)
                    kv_b_proj = torch.cat((attn_k_b, attn_v_b), dim=1)
                    kv_b_proj = kv_b_proj.contiguous() if kv_b_proj.ndim == 2 else kv_b_proj.flatten(0, 1).contiguous()
                    set_param(module, name, kv_b_proj)
                    del attn_k_b
                    del attn_v_b
                else:
                    weights = load_dequantized_tensor(translated_key, device=device).to(dtype=target_dtype)
                    set_param(module, name, weights)
                    del weights
            else:
                #print(load_config.tensor_file_map.keys())
                raise Exception(f"can't find {translated_key} in GGUF file!")
        

def sync_all_device(all_device_list):
    for device in all_device_list:
        if "cuda" in device.lower():
            torch.cuda.synchronize(device)
        elif "xpu" in device.lower():
            torch.xpu.synchronize(device)
        else:
            raise RuntimeError("The device {} is not available".format(device))

torch_device_mapping ={"cuda": "cuda:0", "xpu": "xpu:0"}

def xpu_fp16_model(config):
    # This function is to check if we run this model on XPU with FP16 dtype
    if not torch.xpu.is_available():
        return False
    if config.architectures[0] == "DeepseekV3ForCausalLM":
        return True
    if config.architectures[0] == "Qwen3MoeForCausalLM" and config.hidden_size == 4096:
        # Qwen3-30B seems have precision issue with FP16
        # so we only use FP16 for Qwen3-235B now
        return True
    return False

def load_weights(module:nn.Module, gguf_loader:ModelLoader, prefix='', device="cuda", adapter_gguf=False):
    #print(f"recursively loading weights {prefix}")
    if not isinstance(module, base_operator.BaseInjectedModule):
        load_cur_state_dict(module, gguf_loader, prefix, device=device, adapter_gguf=adapter_gguf, )
        for name, child in module._modules.items():
            load_weights(child, gguf_loader, prefix+name+".", device=device, adapter_gguf=adapter_gguf, )
    else:
        if adapter_gguf == True:
            # TODO: This is not the best choice, because we should change the value of gguf_loader in BaseInjectModule, but up to now, it can still work
            try: # for other class inherit from BaseInjectModule, but not inherit from KTLinear
                module.load(gguf_loader=gguf_loader, adapter_gguf=adapter_gguf)
            except: # for only KTLinear up to now
                module.load()
        else:
            module.load()

def tf_logits_warper(generation_config):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config._eos_token_tensor, list):
                min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
            elif isinstance(generation_config._eos_token_tensor, torch.Tensor):
                min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.min_p is not None:
            # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
            warpers.append(MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
               EtaLogitsWarper(
                    epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep, device=device
                )
            )
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = True,
                         mode = 'normal', force_think: bool = False, chunk_size = 16384, use_flashinfer_mla = False,
                         num_heads = None, head_dim_ckv = None, head_dim_kpe = None, q_head_dim = None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape
    device_map = model.gguf_loader.tensor_device_map
    torch_device = get_device('model.layers.0.self_attn', device_map)
    # torch_device = "cuda:0" if torch_device == "cuda" else torch_device
    torch_device = torch_device_mapping[torch_device] if torch_device in torch_device_mapping else torch_device
    inputs = inputs.to(torch_device)
    all_cuda_device = get_all_used_cuda_device(device_map)

    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph: bool = True):
        if cuda_graph_runner is None:
            use_cuda_graph = False
        if use_cuda_graph:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            # custom_stream = torch.cuda.Stream()
            if torch.cuda.is_available():
                torch.cuda.set_device(torch_device)
            elif torch.xpu.is_available():
                torch.xpu.set_device(torch_device)
            else:
                raise RuntimeError(f"The device: {torch_device} is not available")
            inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(torch_device)
            # with torch.cuda.stream(custom_stream):
            logits=model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=False, use_cache=True)[0]
        if past_key_values != None and isinstance(past_key_values, StaticCache):
            past_key_values.change_seq_length(1)
        sync_all_device(all_cuda_device)
        # print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token
    
    # TODO: use CUDA Graph for chunk prefill, may get small improvement
    def chunk_prefill(inputs, cache_position, past_key_values):
        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            print(f"torch_device:{torch_device}")
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu")).to(torch_device)
        if use_flashinfer_mla:
            MLAWrapperSingleton.update_buffer(past_key_values.max_pages)
            MLAWrapperSingleton.need_plan_all()
            
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        
        return logits
    
    if torch.cuda.is_available():
        torch.cuda.set_device(torch_device)
    elif torch.xpu.is_available():
        torch.xpu.set_device(torch_device)
    else:
        raise RuntimeError(f"The device: {torch_device} is not available")
    with torch.no_grad():
        
        stream = TextStreamer(tokenizer)
        if torch.xpu.is_available():
            from ipex_llm.transformers.kv import DynamicUnbalancedFp8Cache, DynamicNormalCache
            if model.config.architectures[0] in ["DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"]:
                past_key_values = DynamicUnbalancedFp8Cache.from_legacy_cache(None)
            else:
                past_key_values = DynamicNormalCache.from_legacy_cache(None)
        elif mode != 'long_context':
            past_key_values = StaticCache(
                config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = device_map, dtype = model.dtype
            )
        else:
            past_key_values = None
        
        generation_config, model_kwargs = model._prepare_generation_config(
            None, do_sample=True
            # change this to modify generate config
            #top_k=5, top_p=0.85, temperature=0.1
        )

        logits_warper = tf_logits_warper(generation_config)

        cache_position = torch.arange(seq_length, device=torch_device, dtype=torch.int32)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        start_time = time.time()

        chunk_start = 0
        while chunk_start < seq_length:
            chunk_end = min(chunk_start + chunk_size, seq_length)
            if past_key_values != None:
                past_key_values.cur_idx=cache_position[chunk_start:chunk_end]
            logits = chunk_prefill(inputs[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end], past_key_values)
            chunk_start += chunk_size

        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
            
        # decoded_first = tokenizer.decode(next_token)
        # print(f"\n[DEBUG] first token id={next_token.item()} decoded='{decoded_first}'\n")

        first_token_time = time.time() - start_time
        
        if use_flashinfer_mla:
            MLAWrapperSingleton.reset_buffer()

        prefill_count = seq_length
        prefill_time = first_token_time
        if force_think:
            print("<think>")
        print(stream.put(next_token.item()), end="", flush=True)
        # stream.put(next_token.item())
        generated_ids[:, seq_length] = next_token
        tokens.append(int(next_token))
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=torch_device, dtype=torch.int32)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        cuda_graph_runner = None
            
        start_time = time.time()
        for i in range(1, max_new_tokens):
            if use_flashinfer_mla:
                MLAWrapperSingleton.plan_all(None,None,None,position_ids.squeeze(1)+1,None,
                                             num_heads, head_dim_ckv, head_dim_kpe, past_key_values.page_size,
                                             model.model.layers[0].self_attn.softmax_scale, torch.bfloat16, torch.bfloat16)
            global warm_uped
            if use_cuda_graph and ( (warm_uped == True and int(i) == 1) or (warm_uped == False and int(i) == 2) ):
                warm_uped = True
                cuda_graph_runner = CUDAGraphRunner()
                cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, torch_device, return_dict=False, use_cache=True)
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph).to(torch_device)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(int(next_token))
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token.tolist()) == '<|im_end|>':
                # print(stream.end(), end="", flush=True)
                stream.end()
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)
                # stream.put(next_token.item())
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

def prefill_and_generate_capture(
    model, tokenizer, inputs,
    max_new_tokens=10000, use_cuda_graph: bool = True,
    mode='normal', force_think: bool = False, chunk_size=16384,
    use_flashinfer_mla=False, num_heads=None,
    head_dim_ckv=None, head_dim_kpe=None, q_head_dim=None,
    echo_stream: bool = True,
):
    """
    echo_stream=False 时，将不会在终端输出，只写入返回值。
    """
    import os, time, torch, torch.nn as nn
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape
    device_map = model.gguf_loader.tensor_device_map
    torch_device = get_device('model.layers.0.self_attn', device_map)
    torch_device = torch_device_mapping.get(torch_device, torch_device)
    inputs = inputs.to(torch_device)
    all_cuda_device = get_all_used_cuda_device(device_map)
    tokens = []

    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph: bool = True):
        if cuda_graph_runner is None:
            use_cuda_graph = False
        if use_cuda_graph:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            # custom_stream = torch.cuda.Stream()
            if torch.cuda.is_available():
                torch.cuda.set_device(torch_device)
            elif torch.xpu.is_available():
                torch.xpu.set_device(torch_device)
            else:
                raise RuntimeError(f"The device: {torch_device} is not available")
            inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(torch_device)
            # with torch.cuda.stream(custom_stream):
            logits=model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=False, use_cache=True)[0]
        if past_key_values != None and isinstance(past_key_values, StaticCache):
            past_key_values.change_seq_length(1)
        sync_all_device(all_cuda_device)
        # print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token
    
    # TODO: use CUDA Graph for chunk prefill, may get small improvement
    def chunk_prefill(inputs, cache_position, past_key_values):
        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu")).to(torch_device)
        if use_flashinfer_mla:
            MLAWrapperSingleton.update_buffer(past_key_values.max_pages)
            MLAWrapperSingleton.need_plan_all()
            
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        
        return logits

    if torch.cuda.is_available():
        torch.cuda.set_device(torch_device)
    elif torch.xpu.is_available():
        torch.xpu.set_device(torch_device)
    else:
        raise RuntimeError(f"The device: {torch_device} is not available")

    with torch.no_grad():
        stream = SilentCaptureStreamer(tokenizer)

        if torch.xpu.is_available():
            from ipex_llm.transformers.kv import DynamicUnbalancedFp8Cache, DynamicNormalCache
            if model.config.architectures[0] in ["DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"]:
                past_key_values = DynamicUnbalancedFp8Cache.from_legacy_cache(None)
            else:
                past_key_values = DynamicNormalCache.from_legacy_cache(None)
        elif mode != 'long_context':
            past_key_values = StaticCache(
                config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = device_map, dtype = model.dtype
            )
        else:
            past_key_values = None
        
        generation_config, model_kwargs = model._prepare_generation_config(
            None, do_sample=True
            # change this to modify generate config
            #top_k=5, top_p=0.85, temperature=0.1
        )

        logits_warper = tf_logits_warper(generation_config)

        cache_position = torch.arange(seq_length, device=torch_device, dtype=torch.int32)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        start_time = time.time()

        chunk_start = 0
        while chunk_start < seq_length:
            chunk_end = min(chunk_start + chunk_size, seq_length)
            if past_key_values != None:
                past_key_values.cur_idx=cache_position[chunk_start:chunk_end]
            logits = chunk_prefill(inputs[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end], past_key_values)
            chunk_start += chunk_size

        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
            
        # decoded_first = tokenizer.decode(next_token)
        # print(f"\n[DEBUG] first token id={next_token.item()} decoded='{decoded_first}'\n")

        first_token_time = time.time() - start_time
        
        if use_flashinfer_mla:
            MLAWrapperSingleton.reset_buffer()

        prefill_count = seq_length
        prefill_time = first_token_time
        if force_think:
            print("<think>")
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(int(next_token))
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=torch_device, dtype=torch.int32)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        cuda_graph_runner = None
            
        start_time = time.time()
        for i in range(1, max_new_tokens):
            if use_flashinfer_mla:
                MLAWrapperSingleton.plan_all(None,None,None,position_ids.squeeze(1)+1,None,
                                             num_heads, head_dim_ckv, head_dim_kpe, past_key_values.page_size,
                                             model.model.layers[0].self_attn.softmax_scale, torch.bfloat16, torch.bfloat16)
            global warm_uped
            if use_cuda_graph and ( (warm_uped == True and int(i) == 1) or (warm_uped == False and int(i) == 2) ):
                warm_uped = True
                cuda_graph_runner = CUDAGraphRunner()
                cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, torch_device, return_dict=False, use_cache=True)
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph).to(torch_device)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(int(next_token))
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token.tolist()) == '<|im_end|>':
                print(stream.end(), end="", flush=True)
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)

        stream.end()
        return stream.getvalue()
