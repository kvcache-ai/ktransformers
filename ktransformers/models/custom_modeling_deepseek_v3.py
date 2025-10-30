"""
Date: 2024-11-06 10:05:11
LastEditors: djw
LastEditTime: 2024-11-13 07:50:51
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
from ktransformers.models.custom_cache import KDeepSeekV3Cache
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Model,  DeepseekV3PreTrainedModel
from ktransformers.models.configuration_deepseek_v3 import DeepseekV3Config


torch.set_grad_enabled(False)
torch.set_default_dtype(torch.bfloat16)
import flashinfer

class KDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):

    cache: KDeepSeekV3Cache
    use_cuda_graph = False
    def __init__(
        self,
        config: DeepseekV3Config,
        cache,
    ):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.config = config
        self.cache = cache
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def init_wrapper(self, use_cuda_graph, device, max_batch_size, max_pages):
        self.use_cuda_graph = use_cuda_graph
        # Increase buffer sizes to be safe
        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8).to(0)
        self.qo_indptr_buf = torch.empty((max_batch_size+2,), dtype=torch.int32, device=device)
        self.paged_kv_indptr_buf = torch.empty((max_batch_size+2,), dtype=torch.int32, device=device)
        # Make sure this buffer is large enough
        self.paged_kv_indices_buf = torch.empty((max_pages * 2,), dtype=torch.int32, device=device)
        self.paged_kv_len_buf = torch.empty((max_batch_size+1,), dtype=torch.int32, device=device)
        self.bsz_tensor_buf = torch.empty((1, ), dtype=torch.int32, device=device)
		

        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, use_cuda_graph=use_cuda_graph,
            qo_indptr=self.qo_indptr_buf,
            kv_indptr=self.paged_kv_indptr_buf,
            kv_indices=self.paged_kv_indices_buf,
            kv_len_arr=self.paged_kv_len_buf,
            bsz_tensor=self.bsz_tensor_buf,
            backend = "fa2",
        )

    def batch_embeddings(self, batch: ForwardBatchInput, device="cuda:0"):
        features = []
        for i in range(batch.batch_size):
            tokens = batch.minibatch.tokens.contiguous()
            feature = (
                self.model.embed_tokens(tokens.to(torch.device('cpu')))
                .to(torch.bfloat16)
                .to(device=device)
            )
            features.append(feature)

        return features


    def forward(
        self,
        batch: ForwardBatchInput | None = None,
        features: List[torch.Tensor] | None = None,
        bsz_tensors: torch.Tensor | None = None,
        num_tokens_tensors: torch.Tensor | None = None,
        page_idx: torch.Tensor | None = None,
        page_offset: torch.Tensor | None = None,
        cuda_graph_idx: int | None = -1
    ) -> ForwardBatchOutput:
        current_stream = torch.cuda.current_stream()

        forward_batch_output = ForwardBatchOutput()

        
        hidden_states = features[0]

        with torch.cuda.stream(current_stream):
            residual = torch.zeros_like(hidden_states)
            for i, decode_layer in enumerate(self.model.layers):
                # can't use now, only one flashinfer wrapper
                if self.model.transfer_map is not None and i in self.model.transfer_map:
                    prev_stream = torch.cuda.current_stream()
                    cur_device = self.model.transfer_map[i]
                    if cur_device not in self.model.stream_device_map:
                        self.model.stream_device_map[cur_device] = torch.cuda.Stream(cur_device)
                    torch.cuda.set_device(cur_device)
                    self.model.stream_device_map[cur_device].wait_stream(prev_stream)
                    torch.cuda.set_stream(self.model.stream_device_map[cur_device])
                    hidden_states = hidden_states.to(
                        self.model.transfer_map[i], non_blocking=True
                    )

                    batch.minibatch.position_ids = (
                        batch.minibatch.position_ids.to(self.model.transfer_map[i], non_blocking=True)
                        if batch.minibatch.position_ids is not None
                        else None
                    )
                hidden_states, residual = decode_layer.input_layernorm(hidden_states, num_tokens_tensors, residual)
                hidden_states = decode_layer.self_attn(hidden_states, self.cache, 
                                                       position_ids=batch.minibatch.position_ids, 
                                                       wrapper=self.wrapper, num_tokens_tensors=num_tokens_tensors, 
                                                       page_idx=page_idx,
                                                       page_offset=page_offset
                                                       )

                hidden_states, residual = decode_layer.post_attention_layernorm(hidden_states, num_tokens_tensors, residual)
                if i < self.config.first_k_dense_replace:
                    hidden_states = decode_layer.mlp(hidden_states, num_tokens_tensors)
                else:
                    hidden_states = decode_layer.mlp(hidden_states.unsqueeze(0), num_tokens_tensors, cuda_graph_idx)
                    hidden_states = hidden_states.squeeze(0)
        forward_batch_output = ForwardBatchOutput()
        with torch.cuda.stream(current_stream):
            local_logit = self.lm_head(self.model.norm(hidden_states, num_tokens_tensors, residual)[0], num_tokens_tensors)
            forward_batch_output.logits.append(local_logit)

        return forward_batch_output
    

               
    def flash_infer_attn_plan(self, batch: ForwardBatchInput, bsz_tensors, num_tokens_tensors,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,):
        minibatch = batch.minibatch
        self.wrapper.plan(minibatch.q_indptr, minibatch.kv_indptr, minibatch.kv_indices, 
                          minibatch.kv_len, num_heads, head_dim_ckv, head_dim_kpe, page_size, causal, sm_scale, q_data_type, kv_data_type, bsz_tensors)
        
