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

import re
import os
from typing import Optional

import bisect
import torch
import numpy as np
from torch import nn
import torch_npu
from transformers import PretrainedConfig
import torch.nn.functional as F

from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.util.ascend.ascend_utils import get_tensor_parallel_size, get_tensor_parallel_group
from ktransformers.operators.experts import cuda_graphs, KExpertsBase, KExpertsCPU, KTransformersExperts, EXPERTS_MAP, KDeepseekV3MoE
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3MoE
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.utils import CUR_DEVICE, get_use_npu_graph, InferenceState
from ktransformers.operators.experts import cuda_graphs as npu_graphs
from ktransformers.util import utils

class KExpertsCPUW8A8(KExpertsCPU):

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=None, use_npu_graph=False):
        if use_npu_graph:
            seq_len = input_tensor.size(0)
            cuda_graph_idx = seq_len - 1 if cuda_graph_idx is None else cuda_graph_idx # input_tensor is seq & batch merged
            self.cpu_infer.submit(self.moe.forward(KExpertsCPU.expert_ids_cpu[cuda_graph_idx][0].size(0),
                                                    KExpertsCPU.expert_ids_cpu[cuda_graph_idx][0].size(1),
                                                    KExpertsCPU.expert_ids_cpu[cuda_graph_idx][0].data_ptr(),
                                                    KExpertsCPU.weights_cpu[cuda_graph_idx][0].data_ptr(),
                                                    KExpertsCPU.input_tensor_cpu[cuda_graph_idx][0].data_ptr(),
                                                    KExpertsCPU.output_cpu[cuda_graph_idx][0].data_ptr(),
                                                    KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx][0].data_ptr()
                                                    ))
            self.cpu_infer.sync()
        else:
            if bsz_tensor is None:
                bsz_tensor = torch.tensor([input_tensor.size(0)], device=input_tensor.device, dtype=torch.int32)
            # if torch.cuda.is_current_stream_capturing():
            org_type = input_tensor.dtype
            input_tensor = input_tensor.contiguous().cpu()
            input_tensor = input_tensor.to(torch.bfloat16)
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            bsz_tensor = bsz_tensor.contiguous().cpu()
            output = torch.empty_like(input_tensor).contiguous()
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output.data_ptr(), bsz_tensor.data_ptr()))
            self.cpu_infer.sync()
            return output.to(org_type).to(device=utils.get_current_device())

EXPERTS_MAP["KExpertsCPUW8A8"] = KExpertsCPUW8A8

class KTransformersExpertsW8A8(KTransformersExperts):
    def forward(self, input_tensor, expert_ids, weights, cuda_graph_idx=None, use_npu_graph=False):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights, cuda_graph_idx=cuda_graph_idx, use_npu_graph=use_npu_graph)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights, cuda_graph_idx=cuda_graph_idx, use_npu_graph=use_npu_graph)
        else:
            raise ValueError("load or set_inference_mode before forward")


class KDeepseekV3MoEW8A8(KDeepseekV3MoE):
    def forward(self, hidden_states, stream=None, para_stream=None):
        tp_size = get_tensor_parallel_size()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        identity = hidden_states
        orig_shape = hidden_states.shape

        def share_experts_forward():
            if self.config.n_shared_experts is not None:
                return self.shared_experts(identity).squeeze(0)

        if rank == 0:
            topk_idx, topk_weight = self.gate(hidden_states)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if get_use_npu_graph():
                org_type = hidden_states.dtype
                if hasattr(self.config, "backend_type"):
                    if self.config.backend_type == "ktransformers":
                        from ktransformers.util.npu_graph_runner import get_or_create_runner
                        npu_graph_runner = get_or_create_runner(utils.get_current_device())
                        stream = npu_graph_runner.main_stream
                        para_stream = npu_graph_runner.share_experts_stream
                    event = torch.npu.Event()
                    event.record(stream)
                    with torch.npu.stream(para_stream):
                        event.wait(para_stream)
                        y_ = share_experts_forward() if share_experts_forward is not None else None
                        event.record(para_stream)
            
                    input_tensor = hidden_states.to(torch.bfloat16)
                    topk_weight = topk_weight.contiguous().to(torch.float32)
                    cuda_graph_idx = orig_shape[0] - 1
                    self.moe_kexperts_param = (hidden_states, topk_idx, topk_weight, cuda_graph_idx, True)
                    if cuda_graph_idx < len(npu_graphs):
                        expert_ids = topk_idx
                        KExpertsCPU.input_tensor_cpu[cuda_graph_idx][0].copy_(input_tensor, non_blocking = True)
                        KExpertsCPU.expert_ids_cpu[cuda_graph_idx][0].copy_(expert_ids, non_blocking = True)
                        KExpertsCPU.weights_cpu[cuda_graph_idx][0].copy_(topk_weight, non_blocking = True)
                        torch_npu.npu._launch_host_func(stream, self.cpu_moe_kexperts, self.moe_kexperts_param)

                        y = self.experts.generate_experts.output_cpu[cuda_graph_idx][0].to(utils.get_current_device(), non_blocking = True)
                        y = y.view(*orig_shape).to(device=hidden_states.device)
                        y = y.to(org_type)
                    event.wait(stream)
                else:
                    from ktransformers.util.npu_graph_runner import get_or_create_runner
                    npu_graph_runner = get_or_create_runner(utils.get_current_device())
                    event = torch.npu.Event()
                    event.record(npu_graph_runner.main_stream)
                    with torch.npu.stream(npu_graph_runner.share_experts_stream):
                        event.wait(npu_graph_runner.share_experts_stream)
                        y_ = share_experts_forward() if share_experts_forward is not None else None
                        event.record(npu_graph_runner.share_experts_stream)
                    topk_weight = topk_weight.contiguous().to(torch.float32)
                    self.moe_kexperts_param = (hidden_states, topk_idx, topk_weight, None, True)

                    org_type = hidden_states.dtype
                    input_tensor = hidden_states.to(torch.bfloat16)

                    cuda_graph_idx = bisect.bisect_left(npu_graphs, 1)
                    if cuda_graph_idx < len(npu_graphs):

                        immediate_expert_ids = topk_idx
                        KExpertsCPU.input_tensor_cpu[cuda_graph_idx][0].copy_(input_tensor, non_blocking = True)
                        KExpertsCPU.expert_ids_cpu[cuda_graph_idx][0].copy_(immediate_expert_ids, non_blocking = True)
                        KExpertsCPU.weights_cpu[cuda_graph_idx][0].copy_(topk_weight, non_blocking = True)

                        npu_graph_runner.launch_callback(
                            self.cpu_moe_kexperts,
                            self.moe_kexperts_param,
                            1, npu_graph_runner.main_stream)
                        y = self.experts.generate_experts.output_cpu[cuda_graph_idx][0].to(utils.get_current_device(), non_blocking = True)

                        y = y.to(org_type)
                        y = y.view(*orig_shape).to(device=hidden_states.device)
                    event.wait(npu_graph_runner.main_stream)
            else:
                y = self.moe_kexperts(hidden_states, topk_idx, topk_weight)
                y_ = share_experts_forward() if share_experts_forward is not None else None
                y = y.view(*orig_shape).to(device=hidden_states.device)
                y_ = y_.view(*orig_shape)
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            y = torch.zeros(orig_shape, dtype=torch.float16, device=CUR_DEVICE)
            y_ = share_experts_forward() if share_experts_forward is not None else None

        if tp_size > 1 and world_size == tp_size:
            torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM, group=get_tensor_parallel_group())
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def cpu_moe_kexperts(self, moe_kexperts_param) -> torch.Tensor:
        x, topk_ids, topk_weight, cuda_graph_idx, use_npu_graph = moe_kexperts_param
        _ = self.experts(x, topk_ids, topk_weight, cuda_graph_idx=cuda_graph_idx, use_npu_graph=use_npu_graph)

    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

class KQwen3MoeSparseMoeBlockW8A8(BaseInjectedModule):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        prefill_device: str = "npu",
        generate_device: str = "npu",
        **kwargs,
    ):
        super().__init__(
            key,
            gguf_loader,
            config,
            orig_module,
            prefill_device=prefill_device,
            generate_device=generate_device,
            **kwargs,
        )

        self.gate = orig_module.gate
        self.top_k = orig_module.top_k
        self.norm_topk_prob = orig_module.norm_topk_prob
        self.output_router_logits = getattr(orig_module, "output_router_logits", False)

        experts_key = f"{key}.experts"

        print(f"[NPU-MOE][INIT] build experts at key={experts_key}", flush=True)
        self.experts = KTransformersExpertsW8A8(
            key=experts_key,
            gguf_loader=gguf_loader,
            config=config,
            orig_module=orig_module.experts,
            prefill_device=prefill_device,
            prefill_op="KExpertsTorch",
            generate_device="cpu",
            generate_op="KExpertsCPUW8A8",
            out_device=prefill_device,
        )

    def set_inference_mode(self, mode: InferenceState):
        if isinstance(self.experts, KExpertsBase):
            self.experts.set_inference_mode(mode)

    @torch.no_grad()
    def cpu_moe_kexperts(self, moe_kexperts_param):
        x, topk_ids, topk_weight, cuda_graph_idx, use_npu_graph = moe_kexperts_param
        _ = self.experts(
            x,
            topk_ids,
            topk_weight,
            cuda_graph_idx=cuda_graph_idx,
            use_npu_graph=use_npu_graph,
        )

    @torch.no_grad()
    def moe_kexperts(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        bsz_tensor: torch.Tensor = None,
        cuda_graph_idx: int = 0,
        use_npu_graph: bool = False,
    ) -> torch.Tensor:
        outs = self.experts(
            x,
            topk_ids,
            topk_weight,
            cuda_graph_idx=cuda_graph_idx,
            use_npu_graph=use_npu_graph,
        )
        return outs

    def forward(
        self,
        hidden_states: torch.Tensor,
        bsz_tensor: torch.Tensor = None,
        cuda_graph_idx: int = 0,
        *args,
        **kwargs,
    ):

        if hidden_states.dim() == 3:
            B, S, H = hidden_states.shape
        else:
            orig_shape = hidden_states.shape
            hidden_states = hidden_states.view(1, -1, orig_shape[-1])
            B, S, H = hidden_states.shape

        orig_device = hidden_states.device
        orig_shape = (B, S, H)

        output_router_logits_flag = kwargs.pop("output_router_logits", False)
        need_router_logits = output_router_logits_flag or self.output_router_logits

        # ===== 1) flatten =====
        hidden_states_flat = hidden_states.view(-1, H)
        T = hidden_states_flat.shape[0]

        # ===== 2) gate =====
        router_logits = self.gate(hidden_states_flat)
        try:
            router_logits_bs = router_logits.view(B, S, -1)
        except Exception:
            router_logits_bs = router_logits
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            rw_sum = routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights / rw_sum

        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        # ===== 3) MoE experts =====
        use_npu_graph = get_use_npu_graph()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_size = get_tensor_parallel_size()
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            tp_size = 1
            world_size = 1
            rank = 0
        y = None
        if isinstance(self.experts, KExpertsBase):
            if getattr(self.experts, "mode", None) == InferenceState.UNLOAD:
                self.experts.set_inference_mode(InferenceState.GENERATE)

            if rank == 0:
                if use_npu_graph:
                    org_type = hidden_states_flat.dtype
                    input_tensor = hidden_states_flat.to(torch.bfloat16)
                    topk_weight_f32 = routing_weights.contiguous().to(torch.float32)
                    self.moe_kexperts_param = (
                        hidden_states_flat,
                        selected_experts,
                        topk_weight_f32,
                        cuda_graph_idx,
                        True,
                    )
                    if cuda_graph_idx < len(npu_graphs):
                        KExpertsCPU.input_tensor_cpu[cuda_graph_idx][0].copy_(input_tensor, non_blocking=True)
                        KExpertsCPU.expert_ids_cpu[cuda_graph_idx][0].copy_(selected_experts, non_blocking=True)
                        KExpertsCPU.weights_cpu[cuda_graph_idx][0].copy_(topk_weight_f32, non_blocking=True)

                        stream = torch.npu.current_stream()
                        torch_npu.npu._launch_host_func(
                            stream,
                            self.cpu_moe_kexperts,
                            self.moe_kexperts_param,
                        )

                        y_flat = self.experts.generate_experts.output_cpu[cuda_graph_idx][0].to(
                            utils.get_current_device(),
                            non_blocking=True,
                        )
                        y_flat = y_flat.to(org_type)
                        y = y_flat.view(*orig_shape).to(device=orig_device)
                    else:
                        tmp_bsz_tensor = torch.tensor([B], dtype=torch.int32, device=orig_device)
                        y_flat = self.moe_kexperts(
                            hidden_states_flat,
                            selected_experts,
                            routing_weights,
                            bsz_tensor=tmp_bsz_tensor,
                            cuda_graph_idx=cuda_graph_idx,
                            use_npu_graph=False,
                        )
                        y = y_flat.view(*orig_shape).to(device=orig_device)
                else:
                    if bsz_tensor is None:
                        bsz_tensor = torch.tensor(
                            [B],
                            dtype=torch.int32,
                            device=orig_device,
                        )

                    y_flat = self.moe_kexperts(
                        hidden_states_flat,
                        selected_experts,
                        routing_weights,
                        bsz_tensor=bsz_tensor,
                        cuda_graph_idx=cuda_graph_idx,
                        use_npu_graph=False,
                    )
                    y = y_flat.view(*orig_shape).to(device=orig_device)
            else:
                y = torch.zeros(orig_shape, dtype=hidden_states.dtype, device=orig_device)
        else:
            y = hidden_states

        if tp_size > 1 and world_size == tp_size:
            torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM, group=get_tensor_parallel_group())
        # print("================ [NPU-MOE] EXIT MLP =======================\n")
        if need_router_logits:
            num_experts = router_logits.shape[-1]
            router_logits_bs = router_logits.view(B, S, num_experts)
            return y, router_logits_bs


        return y
