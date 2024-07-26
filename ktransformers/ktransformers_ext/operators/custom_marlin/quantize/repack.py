import torch
import enum
from enum import Enum
from typing import Any, Dict, List, Optional
from torch.nn.parameter import Parameter

def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    reshaped_x = x.reshape(-1, x.shape[-1])

    size_m = reshaped_x.shape[0]
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    full_size_k = layer.input_size

    out_shape = x.shape[:-1] + (part_size_n, )

    if layer.marlin_state == GPTQMarlinState.REPACK:
        layer.marlin_state = GPTQMarlinState.READY

        # Newly generated tensors need to replace existing tensors that are
        # already registered as parameters by vLLM (and won't be freed)
        def replace_tensor(name, new_t):
            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            getattr(layer, name).resize_(new_t.shape)
            getattr(layer, name).copy_(new_t)
            del new_t

        cur_device = layer.qweight.device

        # Process act_order
        if self.quant_config.desc_act:
            # Get sorting based on g_idx
            g_idx_sort_indices = torch.argsort(layer.g_idx).to(torch.int)

            sorted_g_idx = layer.g_idx[g_idx_sort_indices]

            replace_tensor("g_idx", sorted_g_idx)
            replace_tensor("g_idx_sort_indices", g_idx_sort_indices)

        else:
            # Reset g_idx related tensors
            layer.g_idx = Parameter(
                torch.empty(0, dtype=torch.int, device=cur_device),
                requires_grad=False,
            )
            layer.g_idx_sort_indices = Parameter(
                torch.empty(0, dtype=torch.int, device=cur_device),
                requires_grad=False,
            )

        # Repack weights
        marlin_qweight = ops.gptq_marlin_repack(
            layer.qweight,
            layer.g_idx_sort_indices,
            part_size_k,
            part_size_n,
            self.quant_config.weight_bits,
        )
        replace_tensor("qweight", marlin_qweight)

        # Permute scales
        scales_size_k = part_size_k
        scales_size_n = part_size_n
        if self.quant_config.desc_act:
            scales_size_k = full_size_k

        marlin_scales = marlin_permute_scales(
            layer.scales,
            scales_size_k,
            scales_size_n,
            self.quant_config.group_size,
            self.quant_config.weight_bits,
        )
        replace_tensor("scales", marlin_scales)

    output = ops.gptq_marlin_gemm(
        reshaped_x,
        layer.qweight,
        layer.scales,
        layer.g_idx,
        layer.g_idx_sort_indices,
        layer.workspace,
        self.quant_config.weight_bits,
        size_m,
        part_size_n,
        part_size_k,
        layer.is_k_full,
    )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)
