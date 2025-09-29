import bisect
import torch
import numpy as np
from torch import nn
import torch_npu
from transformers import PretrainedConfig

from ktransformers.util.ascend.ascend_utils import get_tensor_parallel_size, get_tensor_parallel_group
from ktransformers.util.custom_loader import GGUFLoader

from ktransformers.operators.experts import cuda_graphs, KExpertsBase, KExpertsCPU, KTransformersExperts, EXPERTS_MAP, \
                                            KDeepseekV3MoE
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
                    # todo 同步kt_mb写法
                    npu_graph_runner = get_or_create_runner(utils.get_current_device())
                    event = torch.npu.Event()
                    event.record(npu_graph_runner.main_stream)
                    with torch.npu.stream(npu_graph_runner.share_experts_stream):
                        event.wait(npu_graph_runner.share_experts_stream)
                        y_ = share_experts_forward() if share_experts_forward is not None else None
                        event.record(npu_graph_runner.share_experts_stream)
                    topk_weight = topk_weight.contiguous().to(torch.float32)
                    # todo 同步kt_mb，新增参数个数
                    self.moe_kexperts_param = (hidden_states, topk_idx, topk_weight, None, True)

                    # todo 比kt_mb 多，下面转换类型使用
                    org_type = hidden_states.dtype
                    # todo 和 kt_mb 区别，kt_mb直接使用的 hidden_states
                    input_tensor = hidden_states.to(torch.bfloat16)

                    # todo 同步kt_mb
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
