import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from ktransformers.operators.gate import KMoEGate


class KDeepseekV3GateA2(KMoEGate):
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None):
        if device is None:
            device = self.device
        if w is None:
            w = self.load_weights(device=device)

        if isinstance(w, dict):
            self.weight_type = w["weight_type"]
            self.e_score_correction_bias_type = w["e_score_correction_bias_type"]
            self.orig_module.weight = nn.Parameter(w["weight"])
            self.orig_module.e_score_correction_bias = nn.Parameter(w["e_score_correction_bias"])
        else:
            raise ValueError("Invalid weight type")
        self.orig_module.weight = nn.Parameter(self.orig_module.weight.to(device).to(torch.float32))
        self.orig_module.e_score_correction_bias = nn.Parameter(self.orig_module.e_score_correction_bias.to(device).to(torch.float32))

    def forward(self, hidden_states) -> torch.Tensor:
        h = hidden_states.shape[-1]
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight, None)
        topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
            logits,
            k=self.top_k,
            bias=self.e_score_correction_bias,
            k_group=self.topk_group,
            group_count=self.n_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=float(1e-20))
        return topk_idx.type(torch.int64), topk_weight
