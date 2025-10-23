import os

import torch
import torch_npu
import warnings
from typing import Optional, Tuple
from torch import nn

from ktransformers.models.modeling_deepseek import DeepseekV2Attention, apply_rotary_pos_emb
from typing import Optional, Tuple
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.util.utils import get_compute_capability, get_use_npu_graph, get_current_device
from transformers.configuration_utils import PretrainedConfig
from ktransformers.models.custom_cache import KVC2StaticCache, StaticCache
from ktransformers.server.balance_serve.inference.forward_batch import ForwardMiniBatchSplit
from ktransformers.util.ascend.ascend_utils import get_tensor_parallel_size, allredeuce_warpper, get_tensor_parallel_group
from ktransformers.util.vendors import device_manager, GPUVendor
from ktransformers.util import utils


def apply_rotary_pos_emb_fusion(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


class MatMulOps(object):
    def execute(self, x_input):
        """
            :param x, weight, quant_bia, deq_scale
            :return:
        """
        quant_out = x_input[0]
        weight = x_input[1]
        quant_bia = x_input[2]
        deq_scale = x_input[3]
        return [torch_npu.npu_quant_matmul(quant_out, weight.T, deq_scale, bias=quant_bia, output_dtype=torch.float16)]


class DynamicQuantOps(object):
    """
        :param x, scale, offset
        :return
    """
    def execute(self, x_input):
        out = torch.empty_like(x_input[0], dtype=torch.int8)
        torch_npu._npu_quantize_per_tensor(x_input[0], x_input[1], x_input[2], out)
        return [out]

class KDeepseekV2AttentionW8A8A2(BaseInjectedModule, DeepseekV2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    attn_mask: Optional[torch.Tensor] = None

    class PageKVWrapper(object):
        """
        wrap the difference of KV Cache and Block info between offline model & direct serving & sched serving
        succession should keep the function api
        """
        def __init__(self, past_key_value: StaticCache):
            self.kv_cache = past_key_value
            self.page_size = self.kv_cache.page_size
            self.position = self.kv_cache.position

            self.page_idx = None # staticKV can get from itself
            self.page_offset = None

        def update(self, compressed_kv, k_pe, layer_idx, cache_kwargs):
            return self.kv_cache.update(compressed_kv, k_pe, layer_idx, cache_kwargs)
        
        def get_usable_length(self, kv_seq_len, layer_idx):
            return self.kv_cache.get_usable_length(kv_seq_len, layer_idx)
        
        def get_seq_length(self, layer_idx):
            return self.kv_cache.get_seq_length(layer_idx)
        
        def get_block_table(self, layer_idx):
            return self.kv_cache.page_table_list[layer_idx]

    def init_page_kv_wrapper(self, past_key_value: StaticCache):
        self.page_kv_wrapper = self.PageKVWrapper(past_key_value)

    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 absorb_for_prefill: bool = False,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
                                  orig_module.layer_idx)
        self.chunck_size = config.chunk_size
        self.mla_wrapper = None
        self.page_kv_wrapper = None
        self.absorb_for_prefill = absorb_for_prefill
        self.use_merge = os.getenv("USE_MERGE", "0")
        tp = get_tensor_parallel_size()
        if tp > 1:
            self.num_heads //= tp

        if self.use_merge == "0":
            self.elewise_quant = DynamicQuantOps()
            self.matmulDequant_operation = MatMulOps()
            self.matmulDequant_operation_aclnn = MatMulOps()
        elif self.use_merge == "1":
            print("--Use torch npu FA OP !--")
        else:
            print("--Use default op !--")
        
        self.sparse_mode = 0

    @allredeuce_warpper
    def forward_chunck(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[StaticCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        is_prefill: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            hidden_states_quant = self.elewise_quant.execute([hidden_states, self.q_a_proj.input_scale, self.q_a_proj.input_offset])[0]
            q_a_proj_out = self.matmulDequant_operation.execute([hidden_states_quant, self.q_a_proj.weight,
                                                                 self.q_a_proj.quant_bias, self.q_a_proj.deq_scale])[0]
            q_a_proj_out = self.q_a_layernorm(q_a_proj_out)
            q_a_proj_out = self.elewise_quant.execute([q_a_proj_out, self.q_b_proj.input_scale, self.q_b_proj.input_offset])[0]
            q = self.matmulDequant_operation.execute([q_a_proj_out, self.q_b_proj.weight,
                                                      self.q_b_proj.quant_bias, self.q_b_proj.deq_scale])[0]
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        hidden_states_quant = self.elewise_quant.execute([hidden_states, self.kv_a_proj_with_mqa.input_scale, self.kv_a_proj_with_mqa.input_offset])[0]
        compressed_kv = self.matmulDequant_operation.execute([hidden_states_quant, self.kv_a_proj_with_mqa.weight,
                                                              self.kv_a_proj_with_mqa.quant_bias, self.kv_a_proj_with_mqa.deq_scale])[0]
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since transformer version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += self.page_kv_wrapper.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb_fusion(q_pe, k_pe, cos, sin)

        # update KV
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            cache_kwargs["page_idx"] = self.page_kv_wrapper.page_idx
            cache_kwargs["page_offset"] = self.page_kv_wrapper.page_offset
            k_pe = k_pe.transpose(1, 2)                 # k_pe [bsz, 1, q_len, self.qk_rope_head_dim]
            compressed_kv = compressed_kv.unsqueeze(2)  # compressed_kv [bsz, q_len, self.kv_lora_rank]
            compressed_kv_with_k_pe, _ = self.page_kv_wrapper.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
            if is_prefill:
                compressed_kv_prefill = compressed_kv.clone() # clone for prefill infer
                k_pe_prefill = k_pe.clone()
            compressed_kv, k_pe = torch.split(
                compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        weight_uk = self.q_absorb
        weight_uv = self.out_absorb

        # ATB-MLA-FA+PA
        if self.use_merge == "0" and is_prefill:
            # if self.layer_idx == 0:
            #   print(self.page_kv_wrapper.get_seq_length(self.layer_idx)
            #   self.page_kv_wrapper.get_block_table(self.layer_idx), self.page_kv_wrapper.position)
            current_sqenLen = self.page_kv_wrapper.get_seq_length(self.layer_idx)
            attention_mask = attention_mask[0, :, :, :current_sqenLen].squeeze(0).squeeze(0)

            # FIXME this is wrong in random choose pages for sched, currently just use kv without history
            # compressed_kv = compressed_kv.view(bsz, 1, -1, self.kv_lora_rank)[:,:,:current_sqenLen,:]
            # k_pe = k_pe.view(bsz, 1, -1, self.qk_rope_head_dim)[:,:,:current_sqenLen,:]
            compressed_kv = compressed_kv_prefill.transpose(1,2).contiguous()
            k_pe = k_pe_prefill.transpose(1,2).contiguous()

            k_pe_repeated = k_pe.repeat(1, self.num_heads, 1, 1)
            k_up = torch.matmul(compressed_kv, weight_uk.mT)
            v_up = torch.matmul(compressed_kv, weight_uv)

            qTensor = torch.cat((q_nope, q_pe), dim=-1).transpose(1, 2).contiguous().view(
                                        bsz, q_len, self.num_heads, (self.qk_nope_head_dim + self.qk_rope_head_dim))
            kTensor = torch.cat((k_up, k_pe_repeated), dim=-1).transpose(1, 2).contiguous().view(
                                        bsz, current_sqenLen, self.num_heads, (self.qk_nope_head_dim + self.qk_rope_head_dim))
            vTensor = torch.cat((v_up, k_pe_repeated), dim=-1).transpose(1, 2).contiguous().view(
                                        bsz, current_sqenLen, self.num_heads, (self.v_head_dim + self.qk_rope_head_dim))

            seq_len_data = [q_len] * bsz

            infer_attention_output, _ = torch_npu.npu_fused_infer_attention_score(
                qTensor, kTensor, vTensor,
                atten_mask = attention_mask.type(torch.int8),
                actual_seq_lengths = seq_len_data,
                scale = self.softmax_scale,
                num_heads = self.num_heads,
                num_key_value_heads = self.num_heads,
                input_layout = "BSND")
            
            attn_output = infer_attention_output[..., :self.v_head_dim]
            if tuple(attn_output.size()) != (bsz, q_len, self.num_heads, self.v_head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.v_head_dim)}, but is"
                    f" {tuple(attn_output.size())}"
                )

            attn_output = attn_output.contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim)
            attn_output = self.elewise_quant.execute([attn_output, self.o_proj.input_scale, self.o_proj.input_offset])[0]
            attn_output = self.matmulDequant_operation_aclnn.execute([attn_output, self.o_proj.weight,
                                                                self.o_proj.quant_bias, self.o_proj.deq_scale])[0]

            return attn_output, None, past_key_value

        elif self.use_merge == "0" and not is_prefill:
            return self.forward_paged(q_pe=q_pe,
                                      q_nope=q_nope,
                                      compressed_kv_with_k_pe=compressed_kv_with_k_pe,
                                      past_key_value=past_key_value,
                                      cache_position=cache_position)

        if self.use_merge == "1":
            k_pe_repeated = k_pe.repeat(1, self.num_heads, 1, 1)
            k_up = torch.matmul(compressed_kv, weight_uk.mT)
            v_up = torch.matmul(compressed_kv, weight_uv)
            qTensor = torch.cat((q_nope, q_pe), dim=-1)
            kTensor = torch.cat((k_up, k_pe_repeated), dim=-1)
            vTensor = torch.cat((v_up, k_pe_repeated), dim=-1)

            if q_len != 1:
                attn_output = torch_npu.npu_prompt_flash_attention(
                    qTensor, kTensor, vTensor,
                    num_heads=self.num_heads, scale_value=self.softmax_scale, input_layout="BNSD")
            else:
                attn_output = torch_npu.npu_incre_flash_attention(
                    qTensor, kTensor, vTensor,
                    num_heads=self.num_heads, scale_value=self.softmax_scale, input_layout="BNSD")
            attn_output = attn_output[:, :, :, :self.v_head_dim]
        else:
            q_nope = torch.matmul(q_nope, self.q_absorb)

            attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.mT)) * self.softmax_scale

            compressed_kv = compressed_kv.squeeze(1)
            """
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            assert attention_mask is not None
            """
        if attention_mask is not None:
            """
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            """
            attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q_pe.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)

            attn_output = torch.matmul(attn_output, self.out_absorb)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def forward_paged(
        self,
        q_pe: torch.Tensor,
        q_nope: torch.Tensor,
        compressed_kv_with_k_pe: torch.Tensor,
        past_key_value: Optional[StaticCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if self.layer_idx == 1:
        #   print(self.page_kv_wrapper.get_block_table(self.layer_idx), self.page_kv_wrapper.position)
        bsz, _, q_len, _ = q_nope.size()
        q_nope = torch.einsum('b h q d, h d k -> b h q k', q_nope, self.q_absorb)  # torch.Size([1, 128, 1, 512])
        compressed_kv = compressed_kv_with_k_pe.permute(0, 2, 1, 3)
        kvCache = compressed_kv[:, :, :, :self.kv_lora_rank].contiguous()
        kRopeCache = compressed_kv[:, :, :, self.kv_lora_rank:].contiguous()
        if get_use_npu_graph():
            from ktransformers.util.npu_graph_runner import get_or_create_runner
            npu_graph_runner = get_or_create_runner(get_current_device())
            stream = npu_graph_runner.main_stream
            if npu_graph_runner.past_key_value is None:
                npu_graph_runner.past_key_value = past_key_value
            if npu_graph_runner.workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    q_nope,
                    kvCache,
                    kvCache,
                    query_rope=q_pe,
                    key_rope=kRopeCache,
                    num_heads=self.num_heads,
                    num_key_value_heads=1,
                    input_layout="BNSD",
                    scale=self.softmax_scale,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.page_kv_wrapper.get_block_table(self.layer_idx),
                    block_size=self.page_kv_wrapper.page_size,
                    actual_seq_lengths_kv=self.page_kv_wrapper.position,
                    sparse_mode = self.sparse_mode)
                npu_graph_runner.workspace = workspace
            attn_output = torch.zeros_like(q_nope, dtype=torch.float16, device=get_current_device())
            softmax_lse = torch.empty(1, dtype=torch.float16, device=get_current_device())
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                kvCache,
                kvCache,
                workspace=npu_graph_runner.workspace,
                query_rope=q_pe,
                key_rope=kRopeCache,
                num_heads=self.num_heads,
                num_key_value_heads=1,
                input_layout="BNSD",
                scale=self.softmax_scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.page_kv_wrapper.get_block_table(self.layer_idx),
                block_size=self.page_kv_wrapper.page_size,
                actual_seq_lengths_kv=self.page_kv_wrapper.position,
                sparse_mode = self.sparse_mode,
                out=[attn_output, softmax_lse])

        else:
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                kvCache,
                kvCache,
                query_rope=q_pe,
                key_rope=kRopeCache,
                num_heads=self.num_heads,
                num_key_value_heads=1,
                input_layout="BNSD",
                scale=self.softmax_scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.page_kv_wrapper.get_block_table(self.layer_idx),
                block_size=self.page_kv_wrapper.page_size,
                actual_seq_lengths_kv=self.page_kv_wrapper.position,
                sparse_mode = self.sparse_mode
            )

        attn_output = torch.einsum('b h q k, h k v -> b q h v', attn_output, self.out_absorb)
        attn_output = attn_output.contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.elewise_quant.execute([attn_output, self.o_proj.input_scale, self.o_proj.input_offset])[0]
        attn_output = self.matmulDequant_operation_aclnn.execute([attn_output, self.o_proj.weight,
                                                            self.o_proj.quant_bias, self.o_proj.deq_scale])[0]
        return attn_output, None, past_key_value

    def forward_windows(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[StaticCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        is_prefill: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        self.init_page_kv_wrapper(past_key_value)
        bsz, q_len, _ = hidden_states.size()

        if q_len <= self.chunck_size:
            return self.forward_chunck(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                is_prefill,
                **kwargs
            )

        assert output_attentions == False, "output_attentions is not supported when using chunked attention"
        attn_output = None
        cur_idx = 0
        while cur_idx < q_len:
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, cur_idx:min(cur_idx + self.chunck_size, q_len), ...]
            else:
                # generate chunk_mask automatically.
                self.attn_mask = \
                    torch.zeros(1, 1, self.chunck_size, past_key_value.max_cache_len, device=hidden_states.device) \
                        if self.attn_mask is None \
                        else self.attn_mask
                self.attn_mask[:, :, :, cur_idx:min(cur_idx + self.chunck_size, past_key_value.max_cache_len)] = \
                    -65504.0 * torch.triu(torch.ones(self.chunck_size, self.chunck_size, device=hidden_states.device), diagonal=1) \
                        [:, :min(self.chunck_size, min(past_key_value.max_cache_len - cur_idx, self.chunck_size))]
                self.attn_mask[:, :, :, cur_idx + self.chunck_size:] = -65504.0
                self.attn_mask[:, :, :, :cur_idx] = 0
                chunk_mask = torch.narrow(self.attn_mask, 2, 0, min(self.chunck_size, q_len - cur_idx))

            cur_output, _, _ = self.forward_chunck(
                hidden_states[:, cur_idx:min(cur_idx + self.chunck_size, q_len), ...],
                chunk_mask,
                position_ids[:, cur_idx:min(cur_idx + self.chunck_size, q_len)],
                past_key_value,
                output_attentions,
                use_cache,
                cache_position[cur_idx:min(cur_idx + self.chunck_size, q_len)],
                **kwargs
            )
            cur_idx += self.chunck_size
            if attn_output is None:
                attn_output = cur_output
            else:
                attn_output = torch.cat((attn_output, cur_output), dim=-2)

        return attn_output, None, past_key_value

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[StaticCache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            is_prefill: bool = True,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # TODO: remove cache_position since it do not support multi-batch 
        return self.forward_windows(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            is_prefill,
            **kwargs,
        )


class KDeepseekV2AttentionW8A8A2Serve(BaseInjectedModule, DeepseekV2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    attn_mask: Optional[torch.Tensor] = None

    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1024,
                 absorb_for_prefill: bool = False,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        self.orig_module.__init__(orig_module.config, orig_module.layer_idx)

        # self.chunck_size = chunck_size
        self.absorb_for_prefill = absorb_for_prefill
        self.elewise_quant = DynamicQuantOps()
        self.matmulDequant_operation = MatMulOps()
        self.matmulDequant_operation_aclnn = MatMulOps()
        # tp切分
        tp = get_tensor_parallel_size()
        if tp > 1:
            self.num_heads //= tp
        
        self.sparse_mode = 0
    
    def print_callback(self, param):
        with torch.npu.stream(torch.npu.Stream(device="npu:0")):
            hidden_states, position_ids, cache_position, page_idx, page_offset, block_table = param
            print("########################################")
            print("hidden_states is ", hidden_states)
            print("position_ids is ", position_ids)
            print("cache_position is ", cache_position)
            print("page_idx is ", page_idx)
            print("page_offset is ", page_offset)
            print("block_table is ", block_table)
            print("########################################")
    
    @allredeuce_warpper
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[StaticCache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            is_prefill: Optional[bool] = None,
            page_idx: Optional[torch.Tensor] = None,
            page_offset: Optional[torch.Tensor] = None,
            block_table: Optional[torch.Tensor] = None,
            q_len_raw: Optional[torch.Tensor] = None,
            kv_len_raw: Optional[torch.Tensor] = None,
            stream: Optional[torch.npu.Stream] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        def create_causal_mask(q_lens, kv_lens):
            q_lens = torch.tensor(q_lens)
            kv_lens = torch.tensor(kv_lens)
            bsz = q_lens.size(0)

            max_q_len = q_lens.max().item()
            max_kv_len = kv_lens.max().item()

            # causal mask [max_q_len, max_kv_len]
            base_causal = torch.tril(torch.ones((max_q_len, max_kv_len), dtype=torch.bool))

            # mask initialize: [bsz, max_q_len, max_kv_len] to False
            mask = torch.zeros((bsz, max_q_len, max_kv_len), dtype=torch.bool)

            for i in range(bsz):
                ql, kl = q_lens[i].item(), kv_lens[i].item()
                # copy base_causal to mask
                mask[i, :ql, :kl] = base_causal[:ql, :kl]
            
            return mask
        
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            hidden_states_quant = self.elewise_quant.execute([hidden_states, self.q_a_proj.input_scale, self.q_a_proj.input_offset])[0]
            q_a_proj_out = self.matmulDequant_operation.execute([hidden_states_quant, self.q_a_proj.weight,
                                                                 self.q_a_proj.quant_bias, self.q_a_proj.deq_scale])[0]
            q_a_proj_out = self.q_a_layernorm(q_a_proj_out)
            q_a_proj_out = self.elewise_quant.execute([q_a_proj_out, self.q_b_proj.input_scale, self.q_b_proj.input_offset])[0]
            q = self.matmulDequant_operation.execute([q_a_proj_out, self.q_b_proj.weight,
                                                      self.q_b_proj.quant_bias, self.q_b_proj.deq_scale])[0]
        
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        hidden_states_quant = self.elewise_quant.execute([hidden_states, self.kv_a_proj_with_mqa.input_scale, self.kv_a_proj_with_mqa.input_offset])[0]
        compressed_kv = self.matmulDequant_operation.execute([hidden_states_quant, self.kv_a_proj_with_mqa.weight,
                                                              self.kv_a_proj_with_mqa.quant_bias, self.kv_a_proj_with_mqa.deq_scale])[0]
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since transformer version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb_fusion(q_pe, k_pe, cos, sin)

        # update KV
        compressed_kv_prefill, k_pe_prefill = None, None
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} # Specific to RoPE models
            cache_kwargs["page_idx"], cache_kwargs["page_offset"] = page_idx, page_offset
            k_pe = k_pe.transpose(1, 2)                # k_pe [bsz, 1, q_len, self.qk_rope_head_dim]
            compressed_kv = compressed_kv.unsqueeze(2) # compressed_kv [bsz, q_len, self.kv_lora_rank]
            combined = torch.cat([compressed_kv, k_pe], dim=-1) # shape: [batch_size, num_heads, 2*self.kv_lora_rank]
            # combined = combined.contiguous()

            compressed_kv_with_k_pe, _ = past_key_value.update(combined, self.layer_idx, cache_kwargs)
            if is_prefill:
                compressed_kv_prefill = compressed_kv.clone()
                k_pe_prefill = k_pe.clone()
            
        weight_uk = self.q_absorb
        weight_uv = self.out_absorb

        if is_prefill:
            kTensor_list = []
            vTensor_list = []
            qTensor_list = []
            attention_mask_list = []
            seq_len_data = []
            kv_len_list = []

            for sample_idx in range(bsz):
                current_q_len = q_len_raw[sample_idx].item() if (q_len_raw is not None and sample_idx < len(q_len_raw)) else hidden_states.shape[1]
                current_kv_len = kv_len_raw[sample_idx].item() if (kv_len_raw is not None and sample_idx < len(kv_len_raw)) else current_q_len
                current_q_len = max(1, current_q_len)
                current_kv_len = max(1, current_kv_len)
                seq_len_data.append(current_q_len)
                kv_len_list.append(current_kv_len)

                if attention_mask is not None:
                    mask_sample = attention_mask[
                        sample_idx:sample_idx+1, :, :, :current_kv_len
                    ].squeeze(0).squeeze(0)
                    if mask_sample.shape[0] < current_q_len:
                        mask_sample = torch.nn.functional.pad(mask_sample, (0, 0, 0, current_q_len - mask_sample.shape[0]), value=1)
                    elif mask_sample.shape[0] > current_q_len:
                        mask_sample = mask_sample[:current_q_len, :]
                    if mask_sample.shape[1] < current_kv_len:
                        mask_sample = torch.nn.functional.pad(mask_sample, (0, current_kv_len - mask_sample.shape[1]), value=1)
                    elif mask_sample.shape[1] > current_kv_len:
                        mask_sample = mask_sample[:, :current_kv_len]
                    mask_sample = torch.where(
                        (mask_sample > -1e-6) & (mask_sample < 1e-6),
                        torch.tensor(0, device=mask_sample.device, dtype=torch.int8),
                        torch.tensor(1, device=mask_sample.device, dtype=torch.int8)
                    )
                else:
                    mask_sample = torch.ones((current_q_len, current_kv_len), device=hidden_states.device, dtype=torch.int8)
                    valid_len = min(current_q_len, current_kv_len)
                    mask_sample[:, :valid_len] = 0

                attention_mask_list.append(mask_sample)

                compressed_kv_sample = compressed_kv_prefill[sample_idx:sample_idx+1, :current_q_len, ...].transpose(1, 2).contiguous()
                k_pe_sample = k_pe_prefill[sample_idx:sample_idx+1, :current_q_len, ...].transpose(1, 2).contiguous()
                k_pe_repeated_sample = k_pe_sample.repeat(1, self.num_heads, 1, 1)

                q_nope_sample = q_nope[sample_idx:sample_idx+1, :, :current_q_len, :].contiguous()
                q_pe_sample = q_pe[sample_idx:sample_idx+1, :, :current_q_len, :].contiguous()
                q_concat_sample = torch.cat((q_nope_sample, q_pe_sample), dim=-1)
                q_transposed_sample = q_concat_sample.transpose(1, 2).contiguous()
                qTensor_sample = q_transposed_sample.view(current_q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
                qTensor_list.append(qTensor_sample)

                k_up_sample = torch.matmul(compressed_kv_sample, weight_uk.mT)
                k_concat_sample = torch.cat((k_up_sample, k_pe_repeated_sample), dim=-1)
                k_transposed_sample = k_concat_sample.transpose(1, 2).contiguous()
                kTensor_sample = k_transposed_sample.view(current_kv_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
                kTensor_list.append(kTensor_sample)

                v_up_sample = torch.matmul(compressed_kv_sample, weight_uv)
                v_concat_sample = torch.cat((v_up_sample, k_pe_repeated_sample), dim=-1)
                v_transposed_sample = v_concat_sample.transpose(1, 2).contiguous()
                vTensor_sample = v_transposed_sample.view(current_kv_len, self.num_heads, self.v_head_dim + self.qk_rope_head_dim)
                vTensor_list.append(vTensor_sample)
        
            max_kv_len = max(kv_len_list)
            max_q_len = max(seq_len_data)

            qTensor = torch.nn.utils.rnn.pad_sequence(qTensor_list, batch_first=True, padding_value=0.0).contiguous()
            kTensor = torch.nn.utils.rnn.pad_sequence(kTensor_list, batch_first=True, padding_value=0.0).contiguous()
            vTensor = torch.nn.utils.rnn.pad_sequence(vTensor_list, batch_first=True, padding_value=0.0).contiguous()

            attention_mask = ~create_causal_mask(seq_len_data, kv_len_list).to(qTensor.device)

            infer_attention_output, _ = torch_npu.npu_fused_infer_attention_score(
                    qTensor, kTensor, vTensor,
                    atten_mask = attention_mask.type(torch.int8),
                    actual_seq_lengths = seq_len_data,
                    scale = self.softmax_scale,
                    num_heads = self.num_heads,
                    num_key_value_heads = self.num_heads,
                    input_layout = "BSND")
                
            attn_output = infer_attention_output[..., :self.v_head_dim]

            if tuple(attn_output.size()) != (bsz, max_q_len, self.num_heads, self.v_head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, max_q_len, self.num_heads, self.v_head_dim)}, but is {tuple(attn_output.size())}"
                )
            attn_output = attn_output.contiguous().view(bsz, max_q_len, self.num_heads * self.v_head_dim)
            attn_output = self.elewise_quant.execute([attn_output, self.o_proj.input_scale, self.o_proj.input_offset])[0]
            attn_output = self.matmulDequant_operation_aclnn.execute([attn_output, self.o_proj.weight,
                                                                        self.o_proj.quant_bias, self.o_proj.deq_scale])[0]


            return attn_output, None, past_key_value
        else:
            return self.forward_paged(q_pe = q_pe,
                                      q_nope = q_nope,
                                      compressed_kv_with_k_pe = compressed_kv_with_k_pe,
                                      past_key_value = past_key_value,
                                      cache_position = cache_position,
                                      block_table = block_table,
                                      page_size = past_key_value.page_size,
                                      q_len_raw = q_len_raw,
                                      kv_len_raw = kv_len_raw,
                                      stream = stream)
    
    @allredeuce_warpper
    def forward_paged(
        self,
        q_pe: torch.Tensor,
        q_nope: torch.Tensor,
        compressed_kv_with_k_pe: torch.Tensor,
        past_key_value: Optional[StaticCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        q_len_raw: Optional[torch.Tensor] = None,
        kv_len_raw: Optional[torch.Tensor] = None,
        stream: Optional[torch.npu.Stream] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if self.layer_idx == 0:
        #     print(self.page_kv_wrapper.get_block_table(self.layer_idx), self.page_kv_wrapper.position)
        bsz, _, q_len, _ = q_nope.size()
        # print(f"{q_nope.size()=}")
        q_nope = torch.einsum('b h q d, h d k -> b h q k', q_nope, self.q_absorb)   # torch.size([1, 128, 1, 512])
        compressed_kv = compressed_kv_with_k_pe.permute(0,2,1,3)
        kvCache = compressed_kv[:,:,:,:self.kv_lora_rank].contiguous()
        kRopeCache = compressed_kv[:,:,:,self.kv_lora_rank:].contiguous()
        if get_use_npu_graph():
            from ktransformers.server.balance_serve.inference.model_runner import ModelRunner, get_or_create_model_runner
            npu_graph_runner = get_or_create_model_runner(device=get_current_device())
            npu_graph_idx = bsz - 1
            if npu_graph_runner.workspace[npu_graph_idx] is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    q_nope,
                    kvCache,
                    kvCache,
                    query_rope=q_pe,
                    key_rope=kRopeCache,
                    num_heads=self.num_heads,
                    num_key_value_heads=1,
                    input_layout="BNSD",
                    scale=self.softmax_scale,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=block_table,
                    block_size=page_size,
                    actual_seq_lengths_kv=kv_len_raw,
                    sparse_mode = self.sparse_mode)
                npu_graph_runner.workspace[npu_graph_idx] = workspace
            
            attn_output = torch.zeros_like(q_nope, dtype=torch.float16, device=get_current_device())
            softmax_lse = torch.empty(1, dtype=torch.float16, device=get_current_device())

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                kvCache,
                kvCache,
                workspace=npu_graph_runner.workspace[npu_graph_idx],
                query_rope = q_pe,
                key_rope = kRopeCache,
                num_heads = self.num_heads,
                num_key_value_heads = 1,
                input_layout = "BNSD",
                scale = self.softmax_scale,
                antiquant_mode = 0,
                antiquant_scale = None,
                block_table = block_table,
                block_size = page_size,
                actual_seq_lengths_kv = kv_len_raw,
                sparse_mode = self.sparse_mode,
                out=[attn_output, softmax_lse])
        else:
            tp_group = get_tensor_parallel_group()
            torch.distributed.barrier(tp_group)
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                kvCache,
                kvCache,
                query_rope = q_pe,
                key_rope = kRopeCache,
                num_heads = self.num_heads,
                num_key_value_heads = 1,
                input_layout = "BNSD",
                scale = self.softmax_scale,
                antiquant_mode = 0,
                antiquant_scale = None,
                block_table = block_table,
                block_size = page_size,
                actual_seq_lengths_kv = kv_len_raw,
                sparse_mode = self.sparse_mode
            )

        attn_output = torch.einsum('b h q k, h k v -> b q h v', attn_output, self.out_absorb)
        attn_output = attn_output.contiguous().view(bsz, q_len, self.num_heads*self.v_head_dim)
        attn_output = self.elewise_quant.execute([attn_output, self.o_proj.input_scale, self.o_proj.input_offset])[0]
        attn_output = self.matmulDequant_operation_aclnn.execute([attn_output, self.o_proj.weight,
                                                                  self.o_proj.quant_bias, self.o_proj.deq_scale])[0]
        return attn_output, None, past_key_value
