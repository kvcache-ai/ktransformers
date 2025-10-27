import torch
import torch_npu

from ktransformers.util.ascend.ascend_utils import allredeuce_warpper
from ktransformers.util.utils import CUR_DEVICE
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3MLP

class KDeepseekV3MLPW8A8A2V1(BaseInjectedModule, DeepseekV3MLP):
    @allredeuce_warpper
    def forward(self, x, is_prefill=None, use_cuda_graph=False):
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        dynamic_scale = dynamic_scale.view(-1)
        quant_out = quant_out.view(-1, quant_out.shape[-1])
        gate_x = torch_npu.npu_quant_matmul(
            quant_out,
            self.orig_module.gate_proj.weight,
            self.orig_module.gate_proj.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=original_dtype,
        )
        up_x = torch_npu.npu_quant_matmul(
            quant_out,
            self.orig_module.up_proj.weight,
            self.orig_module.up_proj.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=original_dtype,
        )
        down_x = self.act_fn(gate_x) * up_x
        down_quant_out, down_dynamic_scale = torch_npu.npu_dynamic_quant(down_x)
        down_dynamic_scale = down_dynamic_scale.view(-1)
        down_proj = torch_npu.npu_quant_matmul(
            down_quant_out,
            self.orig_module.down_proj.weight,
            self.orig_module.down_proj.weight_scale,
            pertoken_scale=down_dynamic_scale,
            bias=None,
            output_dtype=original_dtype,
        )
        down_proj = down_proj.reshape(x.shape)
        return down_proj

class KDeepseekV3MLPW8A8A2V2(BaseInjectedModule, DeepseekV3MLP):
    @allredeuce_warpper
    def forward(self, x, is_prefill=None, use_cuda_graph=False):
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        dynamic_scale = dynamic_scale.view(-1)
        quant_out = quant_out.view(-1, quant_out.shape[-1])
        gate_up_x = torch_npu.npu_quant_matmul(
            quant_out,
            self.orig_module.gate_proj.weight,
            self.orig_module.gate_proj.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=original_dtype,
        )
        down_x = torch_npu.npu_swiglu(gate_up_x, -1)
        down_quant_out, down_dynamic_scale = torch_npu.npu_dynamic_quant(down_x)
        down_dynamic_scale = down_dynamic_scale.view(-1)
        down_proj = torch_npu.npu_quant_matmul(
            down_quant_out,
            self.orig_module.down_proj.weight,
            self.orig_module.down_proj.weight_scale,
            pertoken_scale=down_dynamic_scale,
            bias=None,
            output_dtype=original_dtype,
        )
        down_proj = down_proj.reshape(x.shape)
        return down_proj
