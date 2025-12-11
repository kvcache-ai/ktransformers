# coding=utf-8
# Copyright (c) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_npu
from dataclasses import dataclass
from torch.nn import functional as F
import torch.utils.checkpoint

from ktransformers.server.config.config import Config
from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
from ktransformers.models.custom_cache import KVC2Qwen3Cache
from ktransformers.models.modeling_qwen3_moe import Qwen3MoePreTrainedModel, Qwen3MoeModel
from ktransformers.models.configuration_qwen3_moe import Qwen3MoeConfig
import ktransformers.util.utils as utils
from ktransformers.operators.ascend.ascend_layernorm import KQwen3FinalRMSNormNPU

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float16)

class KNPUQwen3MoeForCausalLM(Qwen3MoePreTrainedModel):

    cache: "KVC2Qwen3Cache"
    use_cuda_graph = False

    def __init__(
        self,
        config: "Qwen3MoeConfig",
        cache: "KVC2Qwen3Cache",
        stream: Optional["torch_npu.npu.Stream"] = None,
        default_type: torch.dtype = torch.float16,
    ):
        super().__init__(config)

        self.model = Qwen3MoeModel(config)
        self.config = config
        self.config.backend_type = "balance_serve" 
        self.cache = cache
        self.vocab_size = config.vocab_size

        self.model.to(torch.float16)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.default_type = default_type

        self.stream = torch_npu.npu.current_stream() if stream is None else stream
        self.para_stream = torch_npu.npu.Stream()
        self.call_stream = torch_npu.npu.Stream()

        if hasattr(self.model, "embed_tokens"):
            self.model.embed_tokens.weight.data = self.model.embed_tokens.weight.data.to(torch.float16)

        if hasattr(self.model, "norm"):
            self.model.norm.weight.data = self.model.norm.weight.data.to(torch.float16)
            if getattr(self.model.norm, "bias", None) is not None:
                self.model.norm.bias.data = self.model.norm.bias.data.to(torch.float16)


        try:
            orig_norm = self.model.norm
            self.model.norm = KQwen3FinalRMSNormNPU(orig_norm)
        except Exception as e:
            print(f"[INIT][WARN] replace model.norm failed: {e}", flush=True)

    def init_wrapper(self):
        print("[WARN] KNPUQwen3MoeForCausalLM does not use flashinfer wrapper on NPU, skip init_wrapper...")

    # ---------------------------------------------------
    # Embedding：support prefill / decode modes
    # ---------------------------------------------------
    def batch_embeddings(
        self,
        batch: "ForwardBatchInput",
        device: str = "npu:0",
        is_prefill: bool = True,
    ) -> torch.Tensor:
        features = []

        if is_prefill:
            start_ids = 0
            seq_lens = []

            for i in range(batch.minibatch.prefill_batch):
                qlen = int(batch.minibatch.p_q_len[i])
                kvlen = int(batch.minibatch.p_kv_len[i])

                if kvlen < qlen:
                    raise AssertionError(
                        f"[ERROR] p_kv_len({kvlen}) < p_q_len({qlen}) "
                        f"for prefill idx={i}, this should not happen"
                    )

                tokens = batch.minibatch.p_tokens[start_ids: start_ids + qlen].contiguous()
                start_ids += qlen

                feat = (
                    self.model.embed_tokens(tokens.to(torch.device("cpu")))
                    .to(self.default_type)
                    .to(device=device)
                )

                features.append(feat)
                seq_lens.append(qlen)

            max_seq_len = max(seq_lens) if seq_lens else 0

            # Pad the current chunk to the maximum q_len with [bsz, max_q_len, hidden].
            padded_features = []
            for feat in features:
                curr_len = feat.shape[0]
                if curr_len < max_seq_len:
                    pad_len = max_seq_len - curr_len
                    padded_feat = torch.nn.functional.pad(
                        feat,
                        (0, 0, 0, pad_len),
                        mode="constant",
                        value=0.0,
                    )
                    padded_features.append(padded_feat)
                else:
                    padded_features.append(feat)
            features_t = torch.stack(padded_features, dim=0)  # [bsz, max_seq_len, hidden]
        else:
            for i in range(batch.minibatch.decode_batch):
                if batch.minibatch.d_tokens.dim() == 1:
                    tokens = batch.minibatch.d_tokens.contiguous()
                else:
                    tokens = batch.minibatch.d_tokens[i].contiguous()
                feature = (
                    self.model.embed_tokens(tokens.to(torch.device("cpu")))
                    .to(self.default_type)
                    .to(device=device)
                )
                features.append(feature)
            features_t = torch.stack(features)  # [decode_bsz, decode_q_len, hidden]

        return features_t

    def forward(
            self,
            batch: Optional["ForwardBatchInput"] = None,
            features: torch.Tensor | None = None,
            cache=None,
            bsz_tensors: torch.Tensor | None = None,
            num_tokens_tensors: torch.Tensor | None = None,
            page_idx: torch.Tensor | None = None,
            page_offset: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            block_tables: torch.Tensor | None = None,
            cuda_graph_idx: int | None = 0,
            is_prefill: bool = True,
        ) -> "ForwardBatchOutput":
        try:
            is_capturing = torch.npu.is_current_stream_capturing()
        except Exception:
            is_capturing = False
        # features: [bsz, q_len, hidden]
        if features.ndim == 2:
            hidden_states = features.unsqueeze(0)
        elif features.ndim == 1:
            hidden_states = features.unsqueeze(0).unsqueeze(0)
        else:
            hidden_states = features
        bsz, q_len, hidden_size = hidden_states.shape
        minibatch = batch.minibatch
        if is_prefill:
            device_pos = minibatch.p_position_ids.device
            position_ids = -1 * torch.ones(
                bsz,
                q_len,
                dtype=minibatch.p_position_ids.dtype,
                device=device_pos,
            )
            bsz_real = torch.zeros(bsz, dtype=torch.int32, device=device_pos)
            start_ids = 0
            for i, qlen in enumerate(minibatch.p_q_len):
                position_ids[i, :qlen] = minibatch.p_position_ids[start_ids:start_ids + qlen]
                start_ids += int(qlen.item())
                bsz_real[i] = qlen
            block_tables = minibatch.p_block_tables
            kv_len = minibatch.p_kv_len[0]
            q_len_raw = minibatch.p_q_len
            kv_len_raw = minibatch.p_kv_len
            kv_len_tensor = kv_len_raw
        else:
            position_ids = minibatch.d_position_ids
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            block_tables = minibatch.d_block_tables
            kv_len = minibatch.d_kv_len[0]
            q_len_raw = None
            kv_len_tensor = minibatch.d_kv_len_list
            bsz_real = None

        # ==================== layer loop ====================
        for i, decode_layer in enumerate(self.model.layers):
            # ---------- Attention Block ----------
            attn_residual = hidden_states

            hidden_states = decode_layer.input_layernorm(hidden_states)

            attn_out = decode_layer.self_attn(
                hidden_states,
                past_key_value=self.cache,
                position_ids=position_ids,
                num_tokens_tensors=num_tokens_tensors,
                page_idx=page_idx,
                page_offset=page_offset,
                block_table=block_tables,
                q_len_raw=q_len_raw,
                kv_len_raw=kv_len_tensor,
                is_prefill=is_prefill,
                stream=self.stream,
            )

            hidden_states = attn_residual + attn_out
            # ---------- MLP Block ----------
            mlp_residual = hidden_states
            hidden_states = decode_layer.post_attention_layernorm(hidden_states)
            mlp_in = hidden_states
            mlp_out = decode_layer.mlp(
                mlp_in,
                num_tokens_tensors,
                cuda_graph_idx,
            )

            if isinstance(mlp_out, tuple):
                moe_y = mlp_out[0]
            else:
                moe_y = mlp_out

            hidden_states = mlp_residual + moe_y
        forward_batch_output = ForwardBatchOutput()

        hidden_states_without_norm = hidden_states.clone()

        normed = self.model.norm(hidden_states)

        local_logit = self.lm_head(normed)
        B_out = local_logit.size(0)
        for b in range(B_out):
            if (bsz_real is not None) and (not is_capturing):
                valid_len = int(bsz_real[b].item())
                result = local_logit[b, :valid_len]
                pre_h = hidden_states_without_norm[b, :valid_len]
            else:
                result = local_logit[b]
                pre_h = hidden_states_without_norm[b]

            forward_batch_output.logits.append(result)
            forward_batch_output.pre_hidden_states.append(pre_h)
        return forward_batch_output



    def flash_infer_attn_plan(
        self,
        batch: "ForwardBatchInput",
        bsz_tensors: torch.Tensor,
        num_tokens_tensors: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        causal: bool,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        cuda_graph_idx: int = 0,
    ):
        print("[WARN] KNPUQwen3MoeForCausalLM on NPU does not support flashinfer, skip flash_infer_attn_plan...")
