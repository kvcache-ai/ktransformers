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
import torch_npu
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ktransformers.server.config.config import Config

from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
from ktransformers.models.custom_cache import KVC2StaticCache
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Model,  DeepseekV3PreTrainedModel
from ktransformers.models.configuration_deepseek_v3 import DeepseekV3Config
import ktransformers.util.utils as utils


torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float16)


class KNPUDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):

    # cache: KVC2StaticCache
    use_cuda_graph = False

    def __init__(
        self,
        config: DeepseekV3Config,
        stream = None,
        default_type=torch.float16
    ):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.config = config
        self.config.backend_type = "balance_serve"
        # self.cache = cache
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.default_type = default_type
        self.stream = torch_npu.npu.current_stream() if stream is None else stream
        self.para_stream = torch_npu.npu.Stream()
        self.call_stream = torch_npu.npu.Stream()
        
    def init_wrapper(self, use_cuda_graph, device, max_batch_size, max_pages):
        print('[WARN] this custom modeling do not support flash infer, skip this part...')

    def batch_embeddings(self, batch: ForwardBatchInput, device="npu:0", is_prefill=True):
        features = []
        if is_prefill:
            start_ids = 0
            seq_lens = []
            for i in range(batch.minibatch.prefill_batch):
                assert batch.minibatch.p_kv_len[i] == batch.minibatch.p_q_len[i], \
                    "[ERROR] current prefill do not support chunk or prefix cache"
                tokens = batch.minibatch.p_tokens[start_ids: start_ids+batch.minibatch.p_q_len[i]].contiguous()
                start_ids += batch.minibatch.p_q_len[i]
                feature = (
                    self.model.embed_tokens(tokens.to(torch.device('cpu')))
                    .to(self.default_type)
                    .to(device=device)
                )
                features.append(feature)
                seq_lens.append(feature.shape[0])

            max_seq_len = max(seq_lens) if seq_lens else 0

            padded_features = []
            for feat in features:
                curr_len = feat.shape[0]
                if curr_len < max_seq_len:
                    pad_len = max_seq_len - curr_len
                    padded_feat = torch.nn.functional.pad(
                        feat,
                        (0, 0, 0, pad_len),
                        mode='constant',
                        value=0.0
                    )
                    padded_features.append(padded_feat)
                else:
                    padded_features.append(feat)

            features_t = torch.stack(padded_features)

        else:
            for i in range(batch.minibatch.decode_batch):
                if batch.minibatch.d_tokens.dim() == 1:
                    tokens = batch.minibatch.d_tokens.contiguous()
                else:
                    tokens = batch.minibatch.d_tokens[i].contiguous()

                feature = (
                    self.model.embed_tokens(tokens.to(torch.device('cpu')))
                    .to(self.default_type)
                    .to(device=device)
                )
                features.append(feature)

            features_t = torch.stack(features)
        return features_t

    def print_callback(self, param):
        with torch.npu.stream(self.call_stream):
            hidden_states = param[0]
            print("########################################")
            print("hidden_states is ", hidden_states)
            print("########################################")
        # with torch.npu.stream(self.call_stream):
        #     position_ids, page_idx, page_offset, block_tables, hidden_states, bsz, q_len, hidden_size = param
        #     print("########################################")
        #     print("position_ids is ", position_ids)
        #     print("page_idx is ", page_idx)
        #     print("page_offset is ", page_offset)
        #     print("block_tables is ", block_tables)
        #     print("hidden_states is ", hidden_states)
        #     print("#########################################")


    def forward(
        self,
        batch: ForwardBatchInput | None = None,
        features: torch.Tensor | None = None,
        past_key_value: KVC2StaticCache | None = None,
        bsz_tensors: torch.Tensor | None = None,
        num_tokens_tensors: torch.Tensor | None = None,
        page_idx: torch.Tensor | None = None,
        page_offset: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        cuda_graph_idx: int | None = -1,
        is_prefill: bool = True
    ) -> ForwardBatchOutput:
        # NPU use direct block table from ForwardBatchInput instead of page_idx & page_offset

        if features.ndim == 2:
            hidden_states = features.unsqueeze(0)
        elif features.ndim == 1:
            hidden_states = features.unsqueeze(0).unsqueeze(0)  # (bsz, seqlen, hidden)
        else:
            hidden_states = features

        (bsz, q_len, hidden_size) = hidden_states.shape

        if is_prefill:
            position_ids = -1 * torch.ones(bsz, q_len).to(batch.minibatch.p_position_ids.device)
            bsz_real = torch.zeros(bsz).to(batch.minibatch.p_position_ids.device)
            # convert merged into batched
            start_ids = 0
            for i, qlen in enumerate(batch.minibatch.p_q_len):
                position_ids[i, 0:qlen] = batch.minibatch.p_position_ids[start_ids:start_ids+qlen]
                start_ids += qlen
                bsz_real[i] = qlen
            block_tables = batch.minibatch.p_block_tables
            kv_len = batch.minibatch.p_kv_len[0]
            q_len_raw = batch.minibatch.p_q_len
            kv_len_raw = batch.minibatch.p_kv_len
        else:
            position_ids = batch.minibatch.d_position_ids
            if len(position_ids.shape) == 1:
                position_ids = position_ids.unsqueeze(0)
            block_tables = batch.minibatch.d_block_tables
            kv_len = batch.minibatch.d_kv_len[0]
            q_len_raw = None
            kv_len_raw = batch.minibatch.d_kv_len_list
            bsz_real = None
            # if utils._USE_NPU_GRAPH:
            #     from libgraph_capture import graph_capture_launch_callback
            #     param = (position_ids, page_idx, page_offset, block_tables, hidden_states, bsz, q_len, hidden_size)
            #     graph_capture_launch_callback(self.print_callback, param, 1, self.stream.npu_stream)
            # else:
            #     param = (position_ids, page_idx, page_offset, block_tables, hidden_states, bsz, q_len, hidden_size)
            #     self.print_callback(param)


        # with torch_npu.npu.stream(self.stream):
        # print_ex("####: before decode layer...")
        for i, decode_layer in enumerate(self.model.layers):
            # if not is_prefill:
            #     if utils._USE_NPU_GRAPH:
            #         from libgraph_capture import graph_capture_launch_callback
            #         param = (hidden_states, )
            #         graph_capture_launch_callback(self.print_callback, param, 1, self.stream.npu_stream)
            #     else:
            #         param = (hidden_states, )
            #         self.print_callback(param)
            # attn
            residual = hidden_states
            hidden_states = decode_layer.input_layernorm(hidden_states)

            # generate chunk_mask automatically.
            if is_prefill:
                attn_mask = -65504.0 * torch.triu(torch.ones(q_len, kv_len, device=hidden_states.device), diagonal=1)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # (bsz, 1, q_len, kv_len)
                if bsz > 1:
                    attn_mask = attn_mask.expand(bsz, attn_mask.shape[1], attn_mask.shape[2], attn_mask.shape[3])
            else:
                attn_mask = None
            # print_ex(f"####: before self_attn of layer {i}...")
            hidden_states, _, _ = decode_layer.self_attn(hidden_states,
                                                            position_ids=position_ids,
                                                            attention_mask=attn_mask,
                                                            past_key_value=past_key_value,
                                                            num_tokens_tensors=num_tokens_tensors,
                                                            page_idx=page_idx,
                                                            page_offset=page_offset,
                                                            block_table=block_tables,
                                                            q_len_raw=q_len_raw,
                                                            kv_len_raw=kv_len_raw,
                                                            is_prefill=is_prefill,
                                                            stream = self.stream,
                                                            )
            hidden_states = residual + hidden_states
            # mlp
            residual = hidden_states
            hidden_states = decode_layer.post_attention_layernorm(hidden_states)
            # print_ex(f"####: before mlp of layer {i}...")
            hidden_states = decode_layer.mlp(hidden_states, self.stream, self.para_stream)
            hidden_states = hidden_states.squeeze(0)
            hidden_states = residual + hidden_states
        # print_ex(f"####: fill output...")
        forward_batch_output = ForwardBatchOutput()
        # with torch_npu.npu.stream(self.stream):
        hidden_states_without_norm = hidden_states.clone()
        local_logit = self.lm_head(self.model.norm(hidden_states))
        for bsz in range(local_logit.size(0)):
            if bsz_real is not None:
                index = int(bsz_real[bsz].item())
                result = local_logit[bsz][:index]
            else:
                result = local_logit[bsz]
            forward_batch_output.logits.append(result)
            forward_batch_output.pre_hidden_states.append(hidden_states_without_norm[bsz])
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
        print('[WARN] this custom modeling do not support flash infer, skip this part...')
