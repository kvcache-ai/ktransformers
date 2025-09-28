"""
Date: 2024-11-07 07:02:20
LastEditors: djw
LastEditTime: 2024-12-10 08:48:32
"""
import os.path
import threading

import torch
import torch_npu
from torch import nn
import queue
import signal
import queue
from typing import AsyncIterable
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import asyncio
import multiprocessing
import time
import torch.multiprocessing as mp
import random
import torch.distributed as dist
import zmq
import copy
import tempfile
from ktransformers.server.balance_serve.inference.forward_batch import (
    ForwardBatchInput, ForwardBatchOutput, ForwardMiniBatchCombine, ForwardMiniBatchSplit)
from ktransformers.util import utils
from ktransformers.models.custom_cache import KVC2StaticCache

from ktransformers.server.config.config import Config
from ktransformers.models.custom_modeling_deepseek_v3 import KDeepseekV3ForCausalLM
from ktransformers.models.custom_modeling_deepseek_v2 import KDeepseekV2ForCausalLM
from ktransformers.models.custom_modeling_qwen2_moe import KQwen2MoeForCausalLM
from ktransformers.models.custom_modeling_qwen3_moe import KQwen3MoeForCausalLM
from ktransformers.models.custom_modeling_smallthinker import KSmallThinkerForCausalLM
from ktransformers.models.custom_modeling_glm4_moe import KGlm4MoeForCausalLM
from ktransformers.models.ascend.custom_ascend_modeling_deepseek_v3 import KNPUDeepseekV3ForCausalLM
from ktransformers.server.balance_serve.inference.query_manager import QueryManager
from ktransformers.server.balance_serve.settings import sched_ext

try:
    import torch_npu
    use_torch_npu = torch_npu.npu.is_available()
except:
    use_torch_npu = False


def pad_num_tokens(num_tokens):
    return (num_tokens + 63) // 64 * 64

def deduplicate_and_sort(lst):
    return sorted(set(lst))
def generate_cuda_graphs(chunk_size: int) -> list:
    # 如果输入不符合要求，assert掉
    assert chunk_size <= 1024 or chunk_size % 1024 == 0, "chunk_size must <= 1024 or a multiple of 1024"
    base_list = [1, 2, 3, Config().max_batch_size, 64, 256, 512, chunk_size]

    if chunk_size <= 1024:
        return deduplicate_and_sort(base_list)

    multiples = [i for i in range(1024, chunk_size + 1, 1024)]

    return deduplicate_and_sort(base_list + multiples)
class ModelRunner:
    """A CudaGraphRunner runs the forward pass of a model with CUDA graph and torch.compile."""

    model: KDeepseekV3ForCausalLM  | KQwen2MoeForCausalLM | KQwen3MoeForCausalLM | KSmallThinkerForCausalLM | KGlm4MoeForCausalLM | KNPUDeepseekV3ForCausalLM
    input: ForwardBatchInput | list[ForwardBatchInput]
    output: ForwardBatchOutput
    cache: KVC2StaticCache

    def __init__(self, model = None, cache = None, device = None, use_cuda_graph = False, max_decode_batch_size = 1, max_chunk_size = 4096, num_mini_batches: int = 1, page_size = 256, block_num = 8):
        
        # 先注释掉
        self.model = model  # Compile and move model to the specified device
        if use_torch_npu:
            self.stream = torch.npu.Stream(device=device)
            self.stream_scope = torch.npu.stream
            self.input_decode = []
            max_batch_size = 1 if Config().max_batch_size <= 1 else Config().max_batch_size
            self.npu_graphs = sorted(set([i for i in range(1, max_batch_size + 1)]))
            self.model.stream = self.stream  # npu do not support multi stream like this
            if use_cuda_graph:
                torch_npu.npu._subscribe_report(self.stream)

            self.start_model_event = torch.npu.Event(enable_timing=True)
            self.end_model_event = torch.npu.Event(enable_timing=True)
        else:
            self.stream = torch.cuda.Stream(device=device)
            self.cuda_graphs = generate_cuda_graphs(Config().chunk_size)

            self.start_model_event = torch.cuda.Event(enable_timing=True)
            self.end_model_event = torch.cuda.Event(enable_timing=True)
 
        self.device = device
        self.input = None
        self.features_buf = None
        self.output = None
        self.graph_memory_pool = None
        self.cache = cache
        self.use_cuda_graph = use_cuda_graph
        self.debug = False

        self.model_time = 0
        self.page_size = page_size
        self.block_num = block_num

        if 'cuda' in device:
            self.graphs = [torch.cuda.CUDAGraph() for _ in range(len(self.cuda_graphs))]
            self.page_idx_buf = [torch.zeros([self.cuda_graphs[i]], dtype=torch.int32, device = self.device) for i in range(len(self.cuda_graphs))]
            self.page_offset_buf = [torch.zeros([self.cuda_graphs[i]], dtype=torch.int32, device = self.device) for i in range(len(self.cuda_graphs))]
        elif 'npu' in device:
            self.workspace = [None for _ in range(len(self.npu_graphs))]
            self.graphs = [torch.npu.NPUGraph() for _ in range(len(self.npu_graphs))]
            self.page_idx_buf = [torch.zeros((self.npu_graphs[i], 1), dtype=torch.int32, device = self.device) for i in range(len(self.npu_graphs))]
            self.page_offset_buf = [torch.zeros((self.npu_graphs[i], 1), dtype=torch.int32, device = self.device) for i in range(len(self.npu_graphs))]
        else:
            self.graphs, self.page_idx_buf, self.page_offset_buf = None, None, None
        self.num_mini_batches = num_mini_batches

        self.max_chunk_size = max_chunk_size

        self.bsz_tensor_buf = torch.empty((1, ),dtype=torch.int32, device=device)
        self.num_tokens_tensor_buf = torch.empty((1, ),dtype=torch.int32, device=device)

    def model_attn_plan(self, batch, cuda_graph_idx=0):
        if isinstance(self.model, KDeepseekV3ForCausalLM):
            self.model.flash_infer_attn_plan(batch, self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                             num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                             head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.model.cache.page_size, causal=True,
                                             sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        elif isinstance(self.model, KQwen2MoeForCausalLM) or isinstance(self.model, KQwen3MoeForCausalLM) or isinstance(self.model, KSmallThinkerForCausalLM) or isinstance(self.model, KGlm4MoeForCausalLM):
            self.model.flash_infer_attn_plan(batch, self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                             num_q_heads=self.model.config.num_attention_heads, num_kv_heads=self.model.config.num_key_value_heads,
                                             head_dim=self.model.config.head_dim if hasattr(self.model.config, 'head_dim') else self.model.config.hidden_size // self.model.config.num_attention_heads, 
                                             page_size=self.model.cache.page_size, causal=True,
                                             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, cuda_graph_idx=cuda_graph_idx)
        else:
            assert False, "model type not supported"


    def warmup(self):

        def capture_graphs(cuda_graph_idx):
            with torch.cuda.graph(self.graphs[cuda_graph_idx], pool=self.graph_memory_pool, stream=self.stream):
                self.outputs_buf[cuda_graph_idx] = self.model(self.input[cuda_graph_idx], self.features_buf[cuda_graph_idx], self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[cuda_graph_idx], self.page_offset_buf[cuda_graph_idx], cuda_graph_idx=cuda_graph_idx)   
            self.graph_memory_pool = self.graphs[cuda_graph_idx].pool()

        self.input = []
        self.features_buf = []
        self.outputs_buf = []
        self.bsz_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.num_tokens_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
        for i in range(len(self.cuda_graphs)):
            prefill_query_length = (self.cuda_graphs[i] - Config().max_decode_batch_size) // Config().max_prefill_batch_size if self.cuda_graphs[i] > Config().max_decode_batch_size else 0  #@TODO only supprot 2 prefill batch
            self.input.append(ForwardBatchInput.gen_max_forward_batch(device=self.device, num_mini_batches = self.num_mini_batches, prefill_query_length=prefill_query_length, prefill_active_length=prefill_query_length, page_size=self.page_size, cuda_lens=self.cuda_graphs[i]))

            self.features_buf.append(self.model.batch_embeddings(self.input[i]))
            batch_size = self.input[i].minibatch.q_indptr.size(0)-1
            num_tokens = self.features_buf[i][0].size(0)
            print("capturing cuda graph", batch_size, num_tokens)

            if isinstance(self.model, KQwen2MoeForCausalLM) or isinstance(self.model, KQwen3MoeForCausalLM) or isinstance(self.model, KSmallThinkerForCausalLM) or isinstance(self.model, KGlm4MoeForCausalLM):
                self.model.init_wrapper(self.use_cuda_graph, self.device, num_tokens ,batch_size, self.block_num, i) # TODO: 1024 is a magic number(max_batch_tokens)

            self.bsz_tensor_buf[0] = batch_size
            self.num_tokens_tensor_buf[0] = num_tokens

            self.model_attn_plan(self.input[i], i)
        
            page_idx, page_offset = self.model.cache.get_page_table(self.input[i].minibatch.position_ids, self.input[i].minibatch.q_indptr, self.input[i].minibatch.kv_indptr, self.input[i].minibatch.kv_indices, self.num_tokens_tensor_buf)

            
            self.page_idx_buf[i][:num_tokens].copy_(page_idx[:num_tokens])
            self.page_offset_buf[i][:num_tokens].copy_(page_offset[:num_tokens])

            self.page_idx_buf[i][num_tokens:].fill_(self.model.cache.max_cache_len // self.model.cache.page_size -1) 
        
            self.outputs_buf.append(None)
        
            torch.cuda.synchronize()
            for warm_up_iters in range(11):
                with torch.cuda.stream(self.stream):
                    self.outputs_buf[i] = self.model(self.input[i], self.features_buf[i], self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[i], self.page_offset_buf[i], cuda_graph_idx=i)
            torch.cuda.synchronize()

            self.outputs_buf[i].num_batchs = batch_size

            capture_graphs(i)

            with torch.cuda.stream(self.stream):
                self.graphs[i].replay()

            self.sync(calc_time=False)
            print(f"cuda_graph: {i+1}/{len(self.cuda_graphs)}, warmup finished.")

    def warmup_npu(self):
        # npu 当前使用PD分离
        # 当前只支持 decode 阶段的图下沉
        # 多batch 场景下只支持 1 2 3 4 5 6 7 8
        def capture_graphs(npu_graph_idx):
            utils._USE_NPU_GRAPH = True
            print("self.features_buf[npu_graph_idx] is ", self.features_buf[npu_graph_idx])
            with torch.npu.graph(self.graphs[npu_graph_idx], pool=self.graph_memory_pool, stream=self.stream, auto_dispatch_capture=True):
                self.outputs_buf[npu_graph_idx] = self.model(self.input_decode[npu_graph_idx], self.features_buf[npu_graph_idx], self.cache, None, None, self.page_idx_buf[npu_graph_idx], self.page_offset_buf[npu_graph_idx], self.position_ids_buf[npu_graph_idx], self.block_tables_buf[npu_graph_idx], cuda_graph_idx=npu_graph_idx, is_prefill=False)
            self.graph_memory_pool = self.graphs[npu_graph_idx].pool()
            utils._USE_NPU_GRAPH = False

        self.features_buf = []
        self.outputs_buf = []
        self.position_ids_buf = []
        self.block_tables_buf = []
        self.bsz_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.num_tokens_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
        for i in range(len(self.npu_graphs)):
            prefill_query_length = (self.npu_graphs[i] - Config().max_decode_batch_size) // Config().max_prefill_batch_size if self.npu_graphs[i] > Config().max_decode_batch_size else 0  #@TODO only supprot 2 prefill batch
            self.input_decode.append(ForwardBatchInput.gen_max_forward_batch(device=self.device, num_mini_batches = self.num_mini_batches, decode_batch_size=self.npu_graphs[i], prefill_active_length=1, page_size=self.page_size, cuda_lens = self.npu_graphs[i]))
            self.features_buf.append(self.model.batch_embeddings(self.input_decode[i], device=self.device, is_prefill=False))

            batch_size = self.npu_graphs[i]
            num_tokens = batch_size
            self.bsz_tensor_buf[0] = batch_size
            self.num_tokens_tensor_buf[0] = num_tokens
            
            page_idx, page_offset = self.cache.get_page_table(self.input_decode[i].minibatch, self.num_tokens_tensor_buf, is_prefill=False)

            self.position_ids_buf.append(self.input_decode[i].minibatch.d_position_ids.clone())
            self.block_tables_buf.append(self.input_decode[i].minibatch.d_block_tables.clone())


            self.page_idx_buf[i][:num_tokens].copy_(page_idx[:num_tokens][0])
            page_offset = page_offset.view(self.page_offset_buf[i].size())
            self.page_offset_buf[i][:num_tokens].copy_(page_offset[:num_tokens])
            self.page_idx_buf[i][num_tokens:].fill_(self.cache.max_cache_len // self.cache.page_size -1)
            self.outputs_buf.append(None)

            torch.npu.synchronize()
            for warm_up_iters in range(11):
                with torch.npu.stream(self.stream):
                    self.outputs_buf[i] = self.model(self.input_decode[i], self.features_buf[i], self.cache, self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[i], self.page_offset_buf[i], self.position_ids_buf[i], self.block_tables_buf[i], is_prefill=False)
            torch.npu.synchronize()
            capture_graphs(i)
            self.replay(i)
            self.sync(calc_time=False)
            print(f"npu_graph: {i+1}/{len(self.npu_graphs)}, warmup finished.")


    def run(self, batch: sched_ext.BatchQueryTodo = None, query_manager: QueryManager = None):
        with torch.cuda.stream(self.stream):

            batch_size = len(batch.prefill_mini_batches) # TODO: calc this
            num_tokens = 0
            for i in range(len(batch.decode_mini_batches)):
                batch_size += len(batch.decode_mini_batches[i])
                num_tokens += len(batch.decode_mini_batches[i])
                print(f'decode_batch_i: {len(batch.decode_mini_batches[i])},')

            for i in range(len(batch.prefill_mini_batches)):
                num_tokens += batch.prefill_mini_batches[i][2]
                print(f'prefill_batch_i: {batch.prefill_mini_batches[i][2]},')



            # cuda graph idx equal to min idx i in self.cuda_graphs, that self.cuda_graphs[i] > num_tokens
            cuda_graph_idx = next((i for i, token in enumerate(self.cuda_graphs) if token >= num_tokens), len(self.cuda_graphs))
            if not self.use_cuda_graph:
                cuda_graph_idx = 0
    
            if self.use_cuda_graph:
                self.input[cuda_graph_idx].fill(batch, query_manager, self.page_size)
            else:
                self.input = [ForwardBatchInput(batch=batch, query_manager=query_manager, device=self.device)]
        

            if self.use_cuda_graph:
                self.features = self.model.batch_embeddings(self.input[cuda_graph_idx], device=self.device)

            self.bsz_tensor_buf.copy_(batch_size)
            self.num_tokens_tensor_buf.copy_(torch.tensor([num_tokens], dtype=torch.int32, device=self.device))

            if self.use_cuda_graph:
                self.features_buf[cuda_graph_idx][0].copy_(self.features[0], non_blocking=True)

            self.model_attn_plan(self.input[cuda_graph_idx], cuda_graph_idx)
            self.start_model_event.record(self.stream)

            if self.use_cuda_graph:
                self.model.flash_infer_attn_plan(self.input[cuda_graph_idx], self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                            num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                                head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.cache.page_size, causal=True,
                                                sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                self.start_model_event.record(self.stream)
                page_idx, page_offset = self.cache.get_page_table(self.input[cuda_graph_idx].minibatch, self.bsz_tensor_buf)

                self.page_idx_buf[cuda_graph_idx][:num_tokens].copy_(page_idx[:num_tokens])
                self.page_offset_buf[cuda_graph_idx][:num_tokens].copy_(page_offset[:num_tokens])
                self.page_idx_buf[cuda_graph_idx][num_tokens:].fill_(self.cache.max_cache_len // self.cache.page_size - 1)
                self.replay(cuda_graph_idx)
                self.output = ForwardBatchOutput()
                
                self.output.top_ps.append(self.input[cuda_graph_idx].minibatch.top_ps)
                self.output.temperatures.append(self.input[cuda_graph_idx].minibatch.temperatures)
                self.output.logits.append(self.outputs_buf[cuda_graph_idx].logits[0][self.input[cuda_graph_idx].minibatch.logits_start].clone())

                self.end_model_event.record(self.stream)
            else:
                self.model.flash_infer_attn_plan(self.input, self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                            num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                                head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.cache.page_size, causal=True,
                                                sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                self.start_model_event.record(self.stream)
                page_idx, page_offset = self.cache.get_page_table(self.input[cuda_graph_idx].minibatch, self.bsz_tensor_buf)

                self.output = self.model(self.input, self.features, self.bsz_tensor_buf, self.num_tokens_tensor_buf, page_idx, page_offset)
                self.output.logits[0] = self.output.logits[0][self.input.minibatch.logits_start]
                self.output.top_ps.append(self.input.minibatch.top_ps)
                self.output.temperatures.append(self.input.minibatch.temperatures)

                self.end_model_event.record(self.stream)

        if not self.use_cuda_graph:
            self.output.num_batchs = self.input.batch_size
        else:
            self.output.num_batchs = self.input[cuda_graph_idx].batch_size

    def run_split(self, batch: sched_ext.BatchQueryTodo = None, query_manager: QueryManager = None):
        """running without flashinfer and prefill & decode split infer"""
        def _run_infer_stage(is_prefill=True):
            if "npu" in self.device:
                cuda_graph_idx = batch_size_decode
                # print("batch_size is ", batch_size)
            if is_prefill == False:
                if cuda_graph_idx != -1 and self.use_cuda_graph:
                    self.features = self.model.batch_embeddings(self.input_decode[cuda_graph_idx], device=self.device, is_prefill=is_prefill)
                else:
                    self.features = self.model.batch_embeddings(self.input, device=self.device, is_prefill=is_prefill)

                self.bsz_tensor_buf.copy_(batch_size_decode)

                if self.use_cuda_graph:
                    if cuda_graph_idx != -1:
                        self.features_buf[cuda_graph_idx].copy_(self.features)
                    else:
                        self.features_buf.copy_(self.features)
            else:
                self.features = self.model.batch_embeddings(self.input, device=self.device, is_prefill=is_prefill)
                self.bsz_tensor_buf.copy_(batch_size_decode)

            if cuda_graph_idx != -1 and self.use_cuda_graph and is_prefill == False:
                num_tokens = batch_size_decode + 1
                self.start_model_event.record(self.stream) if self.start_model_event else None
                page_idx, page_offset = self.cache.get_page_table(self.input_decode[cuda_graph_idx].minibatch, self.bsz_tensor_buf, is_prefill=is_prefill)
                self.position_ids_buf[cuda_graph_idx].copy_(self.input_tmp.minibatch.d_position_ids)
                self.block_tables_buf[cuda_graph_idx].copy_(self.input_tmp.minibatch.d_block_tables)
                self.page_idx_buf[cuda_graph_idx][:num_tokens].copy_(page_idx[:num_tokens])
                self.page_offset_buf[cuda_graph_idx][:num_tokens].copy_(page_offset[:num_tokens])
                self.page_idx_buf[cuda_graph_idx][num_tokens:].fill_(self.cache.max_cache_len // self.cache.page_size - 1)

                self.replay(cuda_graph_idx)
                new_output = ForwardBatchOutput()
                # bsz = self.outputs_buf[cuda_graph_idx].logits[0][self.input_decode[cuda_graph_idx].minibatch.d_logits_start].size(0)
                for i in range(num_tokens):
                    new_output.top_ps.append(self.input_decode[cuda_graph_idx].minibatch.d_top_ps[i])
                    new_output.temperatures.append(self.input_decode[cuda_graph_idx].minibatch.d_temperatures[i])
                    new_output.logits.append(self.outputs_buf[cuda_graph_idx].logits[i].clone())  # TODO support MTP
                self.end_model_event.record(self.stream) if self.start_model_event else None

                if self.output is None:
                    self.output = copy.deepcopy(new_output)
                else:
                    self.output.merge(new_output)

            else:
                self.start_model_event.record(self.stream) if self.start_model_event else None
                page_idx, page_offset = self.cache.get_page_table(self.input.minibatch, self.num_tokens_tensor_buf, is_prefill=is_prefill)
                new_output = self.model(self.input, self.features, self.cache, None, None, page_idx, page_offset, None, None, is_prefill=is_prefill)
                bsz = len(new_output.logits)
                if is_prefill:
                    for i in range(bsz):
                        # new_output.logits[i] = new_output.logits[i][self.input.minibatch.p_logits_start[i]:, :]  # slice prefill seq[-1]
                        new_output.logits[i] = new_output.logits[i][-1:, :]  # batched tensor do not need location
                        new_output.top_ps.append(self.input.minibatch.p_top_ps[i])
                        new_output.temperatures.append(self.input.minibatch.p_temperatures[i])
                else:
                    for i in range(bsz):
                        # new_output.logits[i] = new_output.logits[i][self.input.minibatch.d_logits_start[i]:, :]
                        new_output.top_ps.append(self.input.minibatch.d_top_ps[i])
                        new_output.temperatures.append(self.input.minibatch.d_temperatures[i])

                if self.output is None:
                    self.output = copy.deepcopy(new_output)
                else:
                    self.output.merge(new_output)
                self.end_model_event.record(self.stream) if self.end_model_event else None

        with self.stream_scope(self.stream):

            batch_size = len(batch.prefill_mini_batches) # TODO: calc this
            num_d_tokens, num_p_tokens = 0, 0
            for i in range(len(batch.decode_mini_batches)):
                batch_size += len(batch.decode_mini_batches[i])
                num_d_tokens += len(batch.decode_mini_batches[i])
                if self.debug:
                    print(f'decode_batch_i: {len(batch.decode_mini_batches[i])}, token_num: {len(batch.decode_mini_batches[i])} ,batch_size: {batch_size}')

            for i in range(len(batch.prefill_mini_batches)):
                num_p_tokens += batch.prefill_mini_batches[i][2]
                if self.debug:
                    print(f'prefill_batch_i: {batch.prefill_mini_batches[i][2]}, token_num: {batch.prefill_mini_batches[i][2]}')

            # batch info holder both in graph mode & kernel mode
            self.input_tmp = ForwardBatchInput(batch=batch, query_manager=query_manager, device=self.device)
            batch_size_decode = self.input_tmp.minibatch.decode_batch - 1
            idx = self.input_tmp.minibatch.decode_batch - 1
            cuda_graph_idx = batch_size_decode
            self.output = None  # clear last step output

            if self.input_tmp.minibatch.decode_batch > 0:
                if self.use_cuda_graph and len(self.input_decode) > 0:
                    self.input_decode[idx].fill(batch, query_manager, self.page_size)
                else:
                    self.input = self.input_tmp
                    assert isinstance(self.input.minibatch, ForwardMiniBatchSplit), 'split batch input type must be ForwardMiniBatchSplit'
                    print(self.input.minibatch) if self.debug else None

            if self.input_tmp.minibatch.prefill_batch > 0:
                self.input = self.input_tmp
                assert isinstance(self.input.minibatch, ForwardMiniBatchSplit), 'split batch input type must be ForwardMiniBatchSplit'
                print(self.input.minibatch) if self.debug else None

            # ++++++++++++++++++++++++++++++++++++++++++ Prefill Stage ++++++++++++++++++++++++++++++++++++++++++++++++
            if self.input_tmp.minibatch.prefill_batch > 0:
                _run_infer_stage(is_prefill=True)
                self.output.num_batchs = self.input.minibatch.batch_size
            # ++++++++++++++++++++++++++++++++++++++++++ Decode Stage ++++++++++++++++++++++++++++++++++++++++++++++++
            if self.input_tmp.minibatch.decode_batch > 0:
                if self.use_cuda_graph:
                    _run_infer_stage(is_prefill=False)
                    self.output.num_batchs = self.input_decode[idx].minibatch.batch_size
                else:
                    _run_infer_stage(is_prefill=False)
                    self.output.num_batchs = self.input.minibatch.batch_size

            print(self.output) if self.debug else None

    def replay(self, cuda_graph_idx=-1):
        if use_torch_npu:
            thread = threading.Thread(target=self.graphs[cuda_graph_idx].update, kwargs={"cpu_update_input": [{"actual_seq_lengths_kv": self.input_decode[cuda_graph_idx].minibatch.d_kv_len_list}]})
            thread.start()
            torch_npu.npu.synchronize()

        with torch.cuda.stream(self.stream):
            if cuda_graph_idx != -1:
                self.graphs[cuda_graph_idx].replay()
            else:
                self.graphs.replay()


    def sync(self, calc_time = True):
        self.stream.synchronize()
        if calc_time:
            self.model_time = self.start_model_event.elapsed_time(self.end_model_event)  # In ms


def get_or_create_model_runner(model=None, cache=None, device=None, use_cuda_graph=None, page_size=None):
    from ktransformers.server.balance_serve.inference.config import model_runner_dict
    runner = model_runner_dict.get(device)
    if runner is None:
        print("[WARN] the new ModelRunner and deviceId is ", device)
        runner = ModelRunner(model, cache, device, use_cuda_graph, page_size)
        model_runner_dict[device] = runner
    return runner
