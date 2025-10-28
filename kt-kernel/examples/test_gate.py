import math
import os,sys 
import time
from typing import Optional
os.environ["BLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
from cpuinfer_ext.kvcache import ggml_type

import torch
from torch import nn
import torch.nn.functional as F
# from modeling_deepseek_v3 import MoEGate
from configuration_deepseek_v3 import DeepseekV3Config

seed = 42  # 你可以选择任何整数作为种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

seqlen = 64

config = DeepseekV3Config()

hidden_size = config.hidden_size
num_experts_per_token = config.num_experts_per_tok
n_routed_experts = config.n_routed_experts
n_group = config.n_group
topk_group = config.topk_group
routed_scaling_factor = config.routed_scaling_factor

weights = torch.randn((n_routed_experts, hidden_size), dtype=torch.float32).to('cpu').contiguous()
bias = torch.randn((n_routed_experts,), dtype=torch.float32).to('cpu').contiguous()
# weights = torch.randn((n_routed_experts, hidden_size), dtype=torch.float16).to('cpu').contiguous  ()
def load_fp32_tensor(file_path, shape):
    return torch.zeros(shape, dtype=torch.float32).to('cpu').contiguous()
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float32)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)

        h_to_check = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug/gate_input',(seq_len,h))
        diff = (h_to_check - hidden_states).abs().max()
        # print("hidden_states diff:", diff)
        # assert diff<0.02


        bias_to_check = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug/bias',(n_routed_experts))
        diff = (bias - bias_to_check).abs().max()
        # print('bias diff:',diff)
        # assert diff < 0.02


        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )

        logits_to_check = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug/gate_logits',(seq_len,n_routed_experts))
        diff = (logits_to_check - logits).abs().max()
        # print("logits diff:", diff)
        # assert diff < 0.02


        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            # assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)

            scores_to_check = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug/scores_to_choice',(seq_len,n_routed_experts))
            diff = (scores_for_choice - scores_to_check).abs().max()
            print(f'score for choice diff = {diff}')


            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]

            group_scores_to_check = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug/group_scores',(seq_len,n_group))
            diff = (group_scores - group_scores_to_check).abs().max()
            print(f'group scores diff = {diff}')


            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
            tmp_scores_to_check = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug/gate_logits_toped',(seq_len,n_routed_experts))
            is_close = torch.isclose(tmp_scores, tmp_scores_to_check, rtol=1e-2, atol=1e-2, equal_nan=True)
            print(f'tmp_score ok {is_close.all()}')


            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor

        return topk_idx, topk_weight



def torch_gate(hidden_states):
    hidden_states.unsqueeze_(0)
    gate = MoEGate(config)
    gate.weight.data = weights
    gate.e_score_correction_bias.data = bias
    y = gate(hidden_states)
    # print(y)
    return y


def cpuinfer_gate(hidden_states):
    config = cpuinfer_ext.gate.GateConfig(
    hidden_size,
    num_experts_per_token,
    n_routed_experts,
    n_group,
    topk_group,
    )

    CPUInfer = cpuinfer_ext.CPUInfer(64)
    config.routed_scaling_factor = routed_scaling_factor

    config.pool = CPUInfer.backend_
    config.weight = weights.data_ptr()
    config.weight_type = ggml_type.FP32
    config.e_score_correction_bias = bias.data_ptr()
    config.e_score_correction_bias_type = ggml_type.FP32

    gate = cpuinfer_ext.gate.MoEGate(config) 



    expert_ids = torch.zeros((seqlen, num_experts_per_token), dtype=torch.int64).to('cpu').contiguous()
    expert_weights = torch.zeros((seqlen, num_experts_per_token), dtype=torch.float32).to('cpu').contiguous()

    gate.forward(seqlen,hidden_states.data_ptr(),expert_ids.data_ptr(), expert_weights.data_ptr())

    # print(expert_ids,expert_weights)
    return expert_ids, expert_weights

input = torch.randn(seqlen, hidden_size, dtype=torch.float32).to('cpu').contiguous()
# print(input)
ids,we = cpuinfer_gate(input)
idx = torch.argsort(ids, dim=-1, descending=True)
ids = torch.gather(ids,dim=-1,index=idx)
we = torch.gather(we,dim=-1,index=idx)



std_ids,std_we= torch_gate(input)
idx = torch.argsort(std_ids, dim=-1, descending=True)
std_we = torch.gather(std_we,dim=-1,index=idx)
std_ids = torch.gather(std_ids,dim=-1,index=idx)



# print("ids diff:", torch.abs(std_ids - ids).max())
# print("weights diff:", torch.abs(std_we - we).max())
assert torch.abs(std_ids - ids).max() == 0, "Expert IDs do not match!"
assert torch.abs(std_we - we).max() < 1e-2, "Expert Weights do not match!"
print("Expert IDs and Weights match successfully!")

























