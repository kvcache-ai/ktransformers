#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import re
import sys
import threading

import torch
import torch_npu
import torch.distributed as dist
from torch import nn
import itertools
import time
import enum
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
from ktransformers.util.ascend.ascend_utils import get_tensor_parallel_size
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.custom_loader import ModelLoaderFactory, ModelLoader, SafeTensorLoader
from ktransformers.operators import base_operator
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer
if not torch.xpu.is_available():
    from ktransformers.operators.flashinfer_wrapper import MLAWrapperSingleton
# from ktransformers.operators.flashinfer_wrapper import MLAWrapperSingleton
import socket

warm_uped = False
warm_up_cnt = [1, 1]  # skip warm up profiling[prefill, decode]
CUR_DEVICE = None
W8A8_ENABLE = False
Q4_GGUF_LODER = None
_USE_NPU_GRAPH = False
_MAX_DECODE_PROFILE = 1
WARM_UP_SKIP_CNT = [1, 1]
_IS_PREFILL = None
_SPECULATE_STEP = 1

def get_use_npu_graph():
    assert _USE_NPU_GRAPH is not None, "use npu graph is not setting"
    return _USE_NPU_GRAPH

from enum import StrEnum

class StatKey(StrEnum):
    Embedding = "Embedding"
    GraphCapture = "GraphCapture"
    GraphReplay = "GraphReplay"
    ExpertsForward1 = "ExpertsForward1"
    ExpertsForward2 = "ExpertsForward2"
    CPUExperts = "CPUExperts"
    GraphDestroy = "GraphDestroy"
    DecodeOneTokenPost = "DecodeOneTokenPost"
    DecodeOneToken = "DecodeOneToken"
    GraphInit = "GraphInit"

class TimeStat:
    def __init__(self):
        # open_status = os.environ["KT_PERF_STAT"] if "KT_PERF_STAT" in os.environ else "0"
        # if open_status == "0":
        #     self.on = False
        # else:
        #     self.on = True
        self.on = True
        self.prefill_stats = dict()
        self.decode_stats = dict()
        for key in StatKey:
            self.prefill_stats[key] = StatItem()
            self.decode_stats[key] = StatItem()
        self.reset_all()

    def record_start_time(self):
        start_time = time.time_ns()
        return start_time

    def add_time_stat(self, key: StatKey, time_ns, is_prefill):
        if not key:
            return
        # torch.cuda.synchronize()
        cost = time.time_ns() - time_ns
        if is_prefill:
            item = self.prefill_stats[key]
        else:
            item = self.decode_stats[key]
        item.add_item(cost)

    def print_all(self):
        # rank = f"[rank:{torch.distributed.get_rank()}]"
        rank = f"[rank:0]"
        msg = f"\n{rank} Prefill Time Stat\n"
        msg += rank + " {:27}{:>15}{:>15}{:>15}{:>15}{:>15}\n".format("", "min(ms)", "max(ms)", "avg(ms)", "count", "total(ms)")
        for key, value in self.prefill_stats.items():
            msg += rank + f" {key.value:<25}:{value.get_stat()}\n"
        msg += f"\n{rank} Decode Time Stat\n"
        msg += rank + " {:27}{:>15}{:>15}{:>15}{:>15}{:>15}\n".format("", "min(ms)", "max(ms)", "avg(ms)", "count", "total(ms)")
        for key, value in self.decode_stats.items():
            msg += rank + f" {key.value:<25}:{value.get_stat()}\n"
        print(msg)

    def reset_all(self):
        for _, value in self.prefill_stats.items():
            value.reset()
        for _, value in self.decode_stats.items():
            value.reset()


class StatItem:
    def __init__(self):
        self.min_time = 100000000
        self.max_time = 0
        self.total_time_ns = 0
        self.count = 0

    def add_item(self, cost_time_ns):
        self.count += 1
        self.total_time_ns += cost_time_ns
        self.min_time = min(self.min_time, cost_time_ns)
        self.max_time = max(self.max_time, cost_time_ns)

    def reset(self):
        self.min_time = 100000000
        self.max_time = 0
        self.total_time_ns = 0
        self.count = 0

    def get_stat(self):
        min_time = self.min_time / 1000 / 1000
        max_time = self.max_time / 1000 / 1000
        if self.count != 0:
            avg_time = self.total_time_ns / self.count / 1000 / 1000
        else:
            avg_time = 0
        total = self.total_time_ns / 1000 / 1000
        return f"{min_time:15.2f}{max_time:15.2f}{avg_time:15.2f}{self.count:15}{total:15.2f}"


timeStat = TimeStat()


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

def get_current_device():
    return f"npu:{torch.npu.current_device()}"

def get_compute_capability(device:torch.device = None):
    try:
        if torch_npu.npu.is_available():
            return 0
    expect:
        pass
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

def get_device(gguf_module_key:str, device_map:dict):
    if gguf_module_key in device_map:
        return device_map[gguf_module_key]["generate_device"]
    else:
        return "cuda"

def get_all_used_cuda_device(device_map:dict):
    all_device_list = set()
    for key in device_map:
        all_device_list.add(device_map[key]["generate_device"]) if "generate_device" in device_map[key] else None
        all_device_list.add(device_map[key]["prefill_device"]) if "prefill_device" in device_map[key] else None
    if "cpu" in all_device_list:
        all_device_list.remove("cpu")
    all_device_list = set([device.replace('cuda', 'npu') for device in all_device_list])
    all_device_list = list(all_device_list)
    return all_device_list

def load_cur_state_dict(module: nn.Module, gguf_loader: GGUFLoader, prefix: str = ""):
    prefix = prefix.replace("orig_module.", "")
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        key = prefix + name
        translated_key = translate_name_to_gguf(key)
        # TODO: Merge all loader.
        # I know this is ugly but lets do it for now.
        if gguf_loader.safetensor_loader is not None:
            load_dequantized_tensor = gguf_loader.safetensor_loader.load_dequantized_tensor
            tensor_file_map = gguf_loader.safetensor_loader.tensor_file_map
        else:
            load_dequantized_tensor = gguf_loader.load_gguf_tensor
            tensor_file_map = gguf_loader.tensor_file_map
        
        if translated_key in tensor_file_map:
            target_dtype = torch.get_default_dtype()
            device = get_device(translated_key[:translated_key.rfind(".")], gguf_loader.tensor_device_map)
            # Todo need fix
            device = "cpu" if "embd" in translated_key else get_current_device()
            print(f"loading layer {translated_key} to {device}") if torch.distributed.get_rank() == 0 else None
            torch.cuda.empty_cache()
            weights = load_dequantized_tensor(translated_key, device=device).to(dtype=target_dtype)
            set_param(module, name, weights)
            del weights
        else:
            #print(load_config.tensor_file_map.keys())
            raise Exception(f"can't find {translated_key} in GGUF file!")
        
def load_weights(module:nn.Module, gguf_loader:GGUFLoader, prefix=''):
    #print(f"recursively loading weights {prefix}")
    if not isinstance(module, base_operator.BaseInjectedModule):
        load_cur_state_dict(module, gguf_loader, prefix)
        for name, child in module._modules.items():
            load_weights(child, gguf_loader, prefix+name+".")
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
                         num_heads = None, head_dim_ckv = None, head_dim_kpe = None, q_head_dim = None,
                         static_cache = None, draft_model=None, draft_cache=None):
    import os
    CUR_DEVICE = f"npu:{torch.npu.current_device()}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape
    device_map = model.gguf_loader.tensor_device_map
    # torch_device = get_device('blk.0.self_attn', device_map)
    # torch_device = "cuda:0" if torch_device == "cuda" else torch_device
    vocabulary_size = model.config.vocab_size
    topp = torch.tensor([[model.generation_config.top_p]], dtype=torch.float16).npu()
    topk = torch.tensor([[model.generation_config.top_k]], dtype=torch.int32).npu()
    temperature = torch.tensor([[model.generation_config.temperature]], dtype=torch.float16).npu()
    next_token_fake = torch.tensor([[1]], dtype=torch.int32).npu()
    next_token_probs = torch.tensor([[1.0]], dtype=torch.float16).npu()
    torch_device = torch.npu.current_device()
    inputs = inputs.to(torch_device)
    all_cuda_device = get_all_used_cuda_device(device_map)

    tokens = []

    def decode_one_tokens(cuda_graph_runner, next_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph: bool = True):
        # if draft_model is not None:
        #     nonlocal global_acc_counts
        #     nonlocal global_verify_counts
        # next_token, mtp_hidden_states = decode_model_token(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph)
        if draft_model is not None:
            nonlocal draft_token
            nonlocal global_acc_counts
            nonlocal global_verify_counts
        if cuda_graph_runner is None:
            use_cuda_graph = False
        if use_cuda_graph:
            if timeStat.on:
                start_time = timeStat.record_start_time()
            spec_cur_tokens = next_token
            model_inputs_embeds = model.model.embed_tokens(spec_cur_tokens.to('cpu')).to(torch_device)
            model_position_ids = position_ids
            model_cache_position = cache_position
            if timeStat.on:
                timeStat.add_time_stat(StatKey.Embedding, start_time, False)
                start_time = timeStat.record_start_time()
            if cuda_graph_runner.model_capture:
                if timeStat.on:
                    start_time = timeStat.record_start_time()
                cuda_graph_runner.capture(model, spec_cur_tokens, model_position_ids, model_cache_position, past_key_values, CUR_DEVICE, return_dict=False, use_cache=True)
                cuda_graph_runner.model_capture = False
                if timeStat.on:
                    timeStat.add_time_stat(StatKey.GraphCapture, start_time, False)
            if timeStat.on:
                start_time = timeStat.record_start_time()
            ret = cuda_graph_runner(model_inputs_embeds, model_position_ids, model_cache_position)
            if timeStat.on:
                timeStat.add_time_stat(StatKey.GraphReplay, start_time, False)
            logits = ret[0]
            next_token = torch.argmax(logits, dim=-1)
        else:
            torch_npu.npu.set_device(torch_device)
            #inputs_embeds = model.model.embed_tokens(next_token.to("cpu")).to(torch_device)
            if draft_model is not None:
                spec_cur_tokens = torch.cat((next_token[0], torch.tensor(draft_token))).unsqueeze(0)
                model_position_ids = torch.cat((position_ids[0], position_ids[0] + 1)).unsqueeze(0)
                model_cache_position = torch.cat((cache_position, cache_position + 1))
                model_inputs_embeds = model.model.embed_tokens(spec_cur_tokens.to("cpu")).to(torch_device)
            else:
                model_inputs_embeds = model.model.embed_tokens(next_token.to('cpu')).to(torch_device)
                model_position_ids = position_ids
                model_cache_position = cache_position
            # with torch.cuda.stream(custom_stream):
            ret = model(inputs_embeds=model_inputs_embeds,
                       position_ids=model_position_ids,
                       cache_position=model_cache_position,
                       past_key_values=past_key_values,
                       return_dict=False, use_cache=True, is_prefill=False)
            logits = ret[0]

        accept_token_num = 1
        if draft_model is not None:
            global_verify_counts += 1
            if draft_token == next_token[0][0]:   # 接受
                if timeStat.on:
                    start_time = timeStat.record_start_time()
                past_key_values.position[0] += 2
                draft_cache.position[0] += 2
                global_acc_counts += 1
                accept_token_num = 2
                position_ids = model_position_ids + 1
                position_ids = position_ids.squeeze(0)
            else: # 拒绝
                if timeStat.on:
                    start_time = timeStat.record_start_time()
                past_key_values.position[0] += 1
                draft_cache.position[0] += 1
                position_ids = torch.tensor([model_position_ids[0][1]]).to(torch_device)
            if timeStat.on:
                start_time = timeStat.record_start_time()
            if use_cuda_graph:
                if accept_token_num == 1:
                    if timeStat.on:
                        start_time = timeStat.record_start_time()
                else:
                    if timeStat.on:
                        start_time = timeStat.record_start_time()
        else:
            if generation_config.do_sample:
                logits = logits / temperature
                torch.manual_seed(0)
                probs = logits.view(batch_size, vocabulary_size)
                sm = nn.Softmax(dim=-1)
                probs = sm(probs).half().npu()
                next_token = next_token_fake
                torch_npu._npu_topk_topp_sampling(probs, topk, topp, next_token, next_token_probs)
                next_token = next_token.squeeze(-1)
            else:
                next_token_scores = logits_warper(inputs, logits[:, -1, :])
                next_token = torch.argmax(next_token_scores, dim=-1)
        if past_key_values != None:
            past_key_values.change_seq_length(accept_token_num)
        return next_token, accept_token_num

    # TODO: use CUDA Graph for chunk prefill, may get small improvement
    def chunk_prefill(inputs, cache_position, past_key_values):
        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu")).to(torch_device)
            # inputs_embeds = torch_npu.npu_format_cast_(inputs_embeds, 29)
        if use_flashinfer_mla:
            MLAWrapperSingleton.update_buffer(past_key_values.max_pages)
            MLAWrapperSingleton.need_plan_all()

        ret = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True, is_prefill=True
        )
        logits = ret[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        hidden_states = ret[-1]

        return logits, hidden_states

    def decode_wrapper(next_token, position_ids, cache_position, cuda_graph_runner, past_key_values, inputs, seq_length, prof=None):
        global warm_uped
        global _USE_NPU_GRAPH
        nonlocal draft_token
        if use_cuda_graph:
            start_time = timeStat.record_start_time()
            from ktransformers.util.npu_graph_runner import get_or_create_runner
            np_graph_runner = get_or_create_runner(CUR_DEVICE)
            np_graph_runner.init(batch_size, seq_length)
            timeStat.add_time_stat(StatKey.GraphInit, start_time, False)
            
            with torch_npu.npu.stream(np_graph_runner.main_stream):
                gen_num_tokens = 1
                while gen_num_tokens < max_new_tokens:
                    start_time = timeStat.record_start_time()
                    if use_flashinfer_mla:
                        MLAWrapperSingleton.plan_all(None,None,None,position_ids.squeeze(1)+1,None,
                                                    num_heads, head_dim_ckv, head_dim_kpe, past_key_values.page_size,
                                                    model.model.layers[0].self_attn.softmax_scale, torch.bfloat16, torch.bfloat16)
                    if gen_num_tokens == 1:
                        warm_uped = True
                        _USE_NPU_GRAPH = True
                        #np_graph_runner.capture(model, draft_model, next_token, torch.tensor(draft_token), position_ids, cache_position, past_key_values, draft_cache, torch_device, return_dict=False, use_cache=True)
                        cuda_graph_runner = np_graph_runner
                    next_token, cur_gen_num = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph)
                    timeStat.add_time_stat(StatKey.DecodeOneToken, start_time, False)
                    start_time = timeStat.record_start_time()
                    next_token = next_token.to(torch_device)
                    inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
                    generated_ids[:, cache_position] = next_token[0].int()
                    tokens.append(int(next_token[0]))
                    if cur_gen_num == 2:
                        generated_ids[:, (cache_position + 1)] = next_token[1].int()
                        tokens.append(int(next_token[1]))
                    seq_length += cur_gen_num

                    if next_token[-1].item() == tokenizer.eos_token_id or tokenizer.decode(next_token[-1].tolist()) == '<|im_end|>':
                        print(stream.end(), end="", flush=True)
                        break
                    else:
                        if torch.distributed.get_rank() % get_tensor_parallel_size() == 0:
                            print(stream.put(next_token[0].item()), end="", flush=True)
                        if cur_gen_num == 2:
                            print(stream.put(next_token[1].item()), end="", flush=True)

                    cache_position += cur_gen_num
                    past_key_values.position[0] += 1
                    position_ids = cache_position.unsqueeze(0)
                    gen_num_tokens += cur_gen_num
                    if cur_gen_num == 2:
                        next_token = torch.tensor(next_token[-1]).unsqueeze(0)
                    else:
                        next_token = torch.tensor(next_token[0]).unsqueeze(0)
                    timeStat.add_time_stat(StatKey.DecodeOneTokenPost, start_time, False)
                start_time = timeStat.record_start_time()
                np_graph_runner.destroy()
                timeStat.add_time_stat(StatKey.GraphDestroy, start_time, False)
                _USE_NPU_GRAPH = False
        else:
            gen_num_tokens = 1
            while gen_num_tokens < max_new_tokens:
                if use_flashinfer_mla:
                    MLAWrapperSingleton.plan_all(None,None,None,position_ids.squeeze(1)+1,None,
                                                num_heads, head_dim_ckv, head_dim_kpe, past_key_values.page_size,
                                                model.model.layers[0].self_attn.softmax_scale, torch.bfloat16, torch.bfloat16)
                next_token, cur_gen_num = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph)
                next_token = next_token.to(torch_device)
                inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
                generated_ids[:, cache_position] = next_token[0].int()
                tokens.append(int(next_token[0]))
                if cur_gen_num == 2:
                    generated_ids[:, (cache_position + 1)] = next_token[1].int()
                    tokens.append(int(next_token[1]))
                seq_length += cur_gen_num

                if next_token[-1].item() == tokenizer.eos_token_id or tokenizer.decode(next_token.tolist()) == '<|im_end|>':
                    print(stream.end(), end="", flush=True)
                    break
                else:
                    if torch.distributed.get_rank() % get_tensor_parallel_size() == 0:
                        print(stream.put(next_token[0].item()), end="", flush=True)
                    if cur_gen_num == 2:
                        print(stream.put(next_token[1].item()), end="", flush=True)

                cache_position += cur_gen_num
                past_key_values.position[0] += 1
                position_ids = cache_position.unsqueeze(0)
                gen_num_tokens += cur_gen_num
                if cur_gen_num == 2:
                    next_token = torch.tensor(next_token[-1]).unsqueeze(0)
                else:
                    next_token = torch.tensor(next_token[0]).unsqueeze(0)
    

    # torch.cuda.set_device(torch_device)
    torch_npu.npu.set_device(torch_device)
    global warm_up_cnt
    with torch.no_grad():

        stream = TextStreamer(tokenizer)
        if static_cache:
            assert isinstance(static_cache, StaticCache), '[ERROR] static_cache format not equal to StaticCache'
            past_key_values = static_cache
            if past_key_values.max_batch_size < batch_size or past_key_values.max_cache_len < seq_length + max_new_tokens:
                print('[WARN] current staticCache size exceeded, try create new staticCache...')
                past_key_values = StaticCache(
                    config=model.config, max_batch_size=1, max_cache_len=seq_length + max_new_tokens, device=device_map, dtype=model.dtype
                )
            else:
                past_key_values.reset()
        elif mode != 'long_context':
            past_key_values = StaticCache(
                config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = device_map, dtype = model.dtype
            )
        else:
            past_key_values = None

        generation_config, model_kwargs = model._prepare_generation_config(
            None, do_sample=False
            # change this to modify generate config
            #top_k=5, top_p=0.85, temperature=0.1
        )
        try: # transformers==4.43
            logits_warper = (
                model._get_logits_warper(generation_config,device=inputs.device)
            )
        except: 
            logits_warper = (
                model._get_logits_warper(generation_config)
            )

        cache_position = torch.arange(seq_length, device=torch_device, dtype=torch.int32)
        past_key_values.position[0] = seq_length + 1
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        start_time = time.time()
        logits = None
        if draft_model is not None:
            draft_cache.reset()
            past_key_values.position[0] = seq_length + 2
            draft_cache.position[0] = seq_length
            draft_token = None
            global_acc_counts = 0
            global_verify_counts = 0

        def prefill_wrapper(prof=None):
            nonlocal logits
            chunk_start = 0
            while chunk_start < seq_length:
                chunk_end = min(chunk_start + chunk_size, seq_length)
                if past_key_values != None:
                    past_key_values.cur_idx=cache_position[chunk_start:chunk_end]
                logits, _ = chunk_prefill(inputs[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end], past_key_values)
                chunk_start += chunk_size
                if prof is not None:
                    prof.step()
            if prof is not None:
                prof.stop()
            if logits is None:
                raise ValueError('logits cannot be None')

        global WARM_UP_SKIP_CNT
        global _IS_PREFILL
        _IS_PREFILL = True
        prof_prefill = os.environ["PROF_PREFILL"] if "PROF_PREFILL" in os.environ else "0"
        if prof_prefill == "1" and WARM_UP_SKIP_CNT[0] <= 0:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
            )
            with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU
                    ],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=8, repeat=1, skip_first=0),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prefill_prof"),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config) as prof:
                prefill_wrapper(prof)
        else:
            prefill_wrapper()
        WARM_UP_SKIP_CNT[0] -= 1

        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)

        first_token_time = time.time() - start_time

        # print(f"------------------------------------- prefill next_token {next_token}  draft_token {draft_token} ")
        if use_flashinfer_mla:
            MLAWrapperSingleton.reset_buffer()

        prefill_count = seq_length
        prefill_time = first_token_time
        if torch.distributed.get_rank() % get_tensor_parallel_size() == 0:
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
        
        _IS_PREFILL = False
        start_time = time.time()
        prof_decode = os.environ["PROF_DECODE"] if "PROF_DECODE" in os.environ else "0"
        if prof_decode == "1" and warm_up_cnt[1] <= 0:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
            )
            with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU
                    ],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=_MAX_DECODE_PROFILE, repeat=0, skip_first=0),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./decode_prof"),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,  #
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config) as prof:
                decode_wrapper(next_token, position_ids, cache_position, cuda_graph_runner, past_key_values, inputs, seq_length, prof)
        else:
            decode_wrapper(next_token, position_ids, cache_position, cuda_graph_runner, past_key_values, inputs, seq_length)
        warm_up_cnt[1] -= 1

    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    tp_size = get_tensor_parallel_size()
    if torch.distributed.get_rank() % tp_size == 0:
        rank = f"[rank:{torch.distributed.get_rank()}]"
        msg = f"\n{rank} Eval Time\n"
        msg += rank + f"prompt eval count    {prefill_count} token(s)\n"
        msg += rank + f"prompt eval duration {prefill_time:.9f}s\n"
        msg += rank + f"prompt eval rate     {prefill_count/prefill_time:.9f} tokens/s\n"
        msg += rank + f"eval count           {tokens_generated} token(s)\n"
        msg += rank + f"eval duration        {total_time:.9f}s\n"
        msg += rank + f"eval rate            {tokens_per_second:.9f} tokens/s\n"
        print(msg)
    if draft_model is not None:
        print(f"mtp accept rate:      {global_acc_counts}/{global_verify_counts} = {global_acc_counts * 100 / global_verify_counts} %")
    if timeStat.on:
        timeStat.print_all()
        timeStat.reset_all()
    return tokens

class InferenceState(enum.Enum):
    UNLOAD = 0
    PREFILL = 1
    GENERATE = 2
    RESTORE = 3
