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
from ktransformers.models.custom_cache import KGQACache
from ktransformers.models.modeling_glm4_moe import Glm4MoeModel,  Glm4MoePreTrainedModel
from ktransformers.models.configuration_glm4_moe import Glm4MoeConfig
from ktransformers.operators.flashinfer_batch_prefill_wrapper import flashInferAttn

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.bfloat16)
import flashinfer

class KGlm4MoeForCausalLM(Glm4MoePreTrainedModel):

    cache: KGQACache
    use_cuda_graph = False
    def __init__(
        self,
        config: Glm4MoeConfig,
        cache,
    ):

        super().__init__(config)
        self.model = Glm4MoeModel(config)
        self.config = config
        self.cache = cache
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.attn = [None] * 100
        
    def init_wrapper(self, use_cuda_graph, device, max_batch_token, max_batch_size, max_pages, cuda_graph_idx = 0):
        self.attn[cuda_graph_idx] = flashInferAttn(use_cuda_graph=use_cuda_graph, max_batch_token=max_batch_token, max_batch_size=max_batch_size, max_pages=max_pages, device=device)


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
        cuda_graph_idx: int | None = 0
    ) -> ForwardBatchOutput:
        current_stream = torch.cuda.current_stream()

        forward_batch_output = ForwardBatchOutput()

        
        hidden_states = features[0]
        self.attn[cuda_graph_idx].calc_batch_indices(hidden_states.shape[0])

        freqs_cis = self.model.rotary_emb(hidden_states.unsqueeze(0), batch.minibatch.position_ids.unsqueeze(0))


        with torch.cuda.stream(current_stream):
            residual = torch.zeros_like(hidden_states)
            for i, decode_layer in enumerate(self.model.layers):

                hidden_states, residual = decode_layer.input_layernorm(hidden_states, num_tokens_tensors, residual)
                hidden_states = decode_layer.self_attn(hidden_states, self.cache, 
                                                       freqs_cis,
                                                       wrapper=self.attn[cuda_graph_idx], bsz_tensors=num_tokens_tensors, 
                                                       position_ids=batch.minibatch.position_ids
                                                       )

                hidden_states, residual = decode_layer.post_attention_layernorm(hidden_states, num_tokens_tensors, residual)
                if i < self.model.config.first_k_dense_replace:
                    hidden_states = decode_layer.mlp(hidden_states, num_tokens_tensors)
                else:
                    hidden_states = decode_layer.mlp(hidden_states, num_tokens_tensors, cuda_graph_idx)
                    # hidden_states = hidden_states.squeeze(0)

        forward_batch_output = ForwardBatchOutput()
        with torch.cuda.stream(current_stream):
            local_logit = self.lm_head(self.model.norm(hidden_states, num_tokens_tensors, residual)[0], num_tokens_tensors)
            forward_batch_output.logits.append(local_logit)

        return forward_batch_output
    

               
    def flash_infer_attn_plan(self, batch: ForwardBatchInput, bsz_tensors, num_tokens_tensors,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        causal: bool,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        cuda_graph_idx: int = 0
        ):
        minibatch = batch.minibatch
        self.attn[cuda_graph_idx].plan(minibatch.q_indptr, minibatch.kv_indptr, minibatch.kv_indices, 
                          minibatch.kv_last_page_len, bsz_tensors, num_tokens_tensors, num_q_heads, num_kv_heads, head_dim, page_size, causal=causal, q_data_type=q_data_type, kv_data_type=kv_data_type)
        