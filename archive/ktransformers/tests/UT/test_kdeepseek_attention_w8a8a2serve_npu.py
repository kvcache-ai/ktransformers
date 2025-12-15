import sys
import types

import torch
import torch.nn as nn
import pytest

torch_npu = pytest.importorskip("torch_npu")

from ktransformers.operators.ascend.ascend_attention import (
    KDeepseekV2AttentionW8A8A2Serve,
)
import ktransformers.operators.ascend.ascend_attention as attn_mod

class DummyConfig:
    def __init__(self, hidden_size=4, num_attention_heads=1):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads


class DummyOrigAttn(nn.Module):
    def __init__(self, config=None, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        hidden_dim = config.hidden_size if config is not None else 4

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_a_proj_with_mqa = None
        self.kv_a_layernorm = nn.LayerNorm(2)
        self.o_proj = None


class DummyDynamicQuantOps:
    def execute(self, inputs):
        x = inputs[0]
        return [x]


class DummyMatMulOps:
    def execute(self, inputs):
        x = inputs[0]
        return [x]


class DummyQuantProj(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input_scale = torch.tensor(1.0, dtype=torch.float16)
        self.input_offset = torch.tensor(0.0, dtype=torch.float16)
        self.weight = nn.Parameter(torch.zeros(dim, dim, dtype=torch.float16))
        self.quant_bias = torch.zeros(dim, dtype=torch.float16)
        self.deq_scale = torch.tensor(1.0, dtype=torch.float16)


class DummyStaticCache:
    def __init__(self, page_size=16):
        self.page_size = page_size

    def get_usable_length(self, kv_seq_len, layer_idx):
        return 0

    def update(self, combined, layer_idx, cache_kwargs):
        return combined, None


class DummyNpuFusedAttention:
    def __call__(self, q, k, v, **kwargs):
        bsz, max_q_len, num_heads, dim = q.shape
        out = torch.zeros(
            bsz, max_q_len, num_heads, dim, dtype=q.dtype, device=q.device
        )
        softmax_lse = torch.zeros(1, dtype=q.dtype, device=q.device)
        return out, softmax_lse

    def out(self, q, k, v, workspace=None,
            query_rope=None, key_rope=None,
            num_heads=None, num_key_value_heads=None,
            input_layout=None, scale=None,
            antiquant_mode=None, antiquant_scale=None,
            block_table=None, block_size=None,
            actual_seq_lengths_kv=None,
            sparse_mode=None,
            out=None):
        attn_output, softmax_lse = out
        attn_output.zero_()
        softmax_lse.zero_()
        return attn_output, softmax_lse


class DummyOpsNpu:
    def npu_fused_infer_attention_score(self, q, k, v, **kwargs):
        bsz, num_heads, q_len, dim = q.shape
        out = torch.zeros(
            bsz, num_heads, q_len, dim, dtype=q.dtype, device=q.device
        )
        softmax_lse = torch.zeros(1, dtype=q.dtype, device=q.device)
        return out, softmax_lse

def fake_apply_rotary_pos_emb_fusion(q_pe, k_pe, cos, sin):
    return q_pe, k_pe

def build_attention_module(q_lora_rank=None):
    if hasattr(attn_mod, "get_tensor_parallel_size"):
        attn_mod.get_tensor_parallel_size = lambda: 1  # type: ignore

    config = DummyConfig(hidden_size=4, num_attention_heads=1)
    orig = DummyOrigAttn(config=config, layer_idx=0)

    attn = KDeepseekV2AttentionW8A8A2Serve(
        key="test",
        gguf_loader=None,
        config=config,
        orig_module=orig,
        prefill_device="npu",
        generate_device="npu",
    )

    hidden_dim = 4
    num_heads = 1
    qk_nope_head_dim = 2
    qk_rope_head_dim = 2
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 4
    kv_lora_rank = 2
    v_head_dim = 2

    attn.num_heads = num_heads
    attn.q_head_dim = q_head_dim
    attn.qk_nope_head_dim = qk_nope_head_dim
    attn.qk_rope_head_dim = qk_rope_head_dim
    attn.kv_lora_rank = kv_lora_rank
    attn.v_head_dim = v_head_dim
    attn.softmax_scale = 1.0
    attn.layer_idx = 0
    attn.sparse_mode = 0
    attn.q_lora_rank = q_lora_rank

    attn.elewise_quant = DummyDynamicQuantOps()
    attn.matmulDequant_operation = DummyMatMulOps()
    attn.matmulDequant_operation_aclnn = DummyMatMulOps()

    orig_mod = attn.orig_module

    if q_lora_rank is None:
        orig_mod.q_proj = nn.Linear(hidden_dim, num_heads * q_head_dim, bias=False)
        orig_mod.q_proj = orig_mod.q_proj.to(dtype=torch.float16)
    else:
        orig_mod.q_a_proj = DummyQuantProj(hidden_dim)
        orig_mod.q_b_proj = DummyQuantProj(hidden_dim)
        orig_mod.q_a_layernorm = nn.LayerNorm(hidden_dim)

    orig_mod.kv_a_proj_with_mqa = DummyQuantProj(hidden_dim)
    orig_mod.kv_a_layernorm = nn.LayerNorm(kv_lora_rank)

    orig_mod.o_proj = DummyQuantProj(num_heads * v_head_dim)

    attn.q_absorb = torch.randn(
        num_heads, qk_nope_head_dim, kv_lora_rank, dtype=torch.float16
    )
    attn.out_absorb = torch.randn(
        num_heads, kv_lora_rank, v_head_dim, dtype=torch.float16
    )
    def fake_rotary_emb(q_pe, position_ids):
        bsz, n_heads, q_len, dim = q_pe.shape
        cos = torch.ones(1, 1, q_len, dim, dtype=q_pe.dtype, device=q_pe.device)
        sin = torch.zeros(1, 1, q_len, dim, dtype=q_pe.dtype, device=q_pe.device)
        return cos, sin

    attn.rotary_emb = fake_rotary_emb

    return attn

@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    if hasattr(attn_mod, "apply_rotary_pos_emb_fusion"):
        monkeypatch.setattr(
            attn_mod, "apply_rotary_pos_emb_fusion",
            fake_apply_rotary_pos_emb_fusion
        )

    if hasattr(attn_mod, "get_use_npu_graph"):
        monkeypatch.setattr(attn_mod, "get_use_npu_graph", lambda: False)

    if hasattr(attn_mod, "get_tensor_parallel_size"):
        monkeypatch.setattr(attn_mod, "get_tensor_parallel_size", lambda: 1)

    if hasattr(attn_mod, "get_tensor_parallel_group"):
        monkeypatch.setattr(attn_mod, "get_tensor_parallel_group", lambda: None)

    if hasattr(attn_mod, "get_current_device"):
        monkeypatch.setattr(attn_mod, "get_current_device", lambda: "cpu")

    # torch.distributed.barrier -> no-op
    if hasattr(torch, "distributed") and hasattr(torch.distributed, "barrier"):
        monkeypatch.setattr(
            torch.distributed, "barrier",
            lambda *args, **kwargs: None,
            raising=False,
        )

    dummy_op = DummyNpuFusedAttention()
    monkeypatch.setattr(
        torch_npu, "npu_fused_infer_attention_score",
        dummy_op, raising=False
    )

    def fake_get_workspace(q, k, v, **kwargs):
        return torch.empty(1, dtype=q.dtype, device=q.device)

    monkeypatch.setattr(
        torch_npu, "_npu_fused_infer_attention_score_get_max_workspace",
        fake_get_workspace, raising=False
    )

    monkeypatch.setattr(torch.ops, "npu", DummyOpsNpu(), raising=False)

    yield


# ==========================
#  测试用例
# ==========================

def test_print_callback_smoke():
    attn = build_attention_module()
    bsz, q_len, hidden_dim = 1, 3, 4
    hidden_states = torch.randn(bsz, q_len, hidden_dim)
    position_ids = torch.arange(q_len).unsqueeze(0)
    cache_position = torch.arange(q_len).unsqueeze(0)
    page_idx = torch.zeros(bsz, dtype=torch.int32)
    page_offset = torch.zeros(bsz, dtype=torch.int32)
    block_table = torch.zeros(bsz, 1, dtype=torch.int32)

    attn.print_callback(
        (hidden_states, position_ids, cache_position,
         page_idx, page_offset, block_table)
    )


def _common_inputs_prefill():
    bsz, q_len, hidden_dim = 1, 3, 4
    hidden_states = torch.randn(bsz, q_len, hidden_dim, dtype=torch.float16)
    attention_mask = torch.zeros(bsz, 1, q_len, q_len, dtype=torch.float32)
    position_ids = torch.arange(q_len).unsqueeze(0)
    cache_position = torch.arange(q_len).unsqueeze(0)
    page_idx = torch.zeros(bsz, dtype=torch.int32)
    page_offset = torch.zeros(bsz, dtype=torch.int32)
    block_table = torch.zeros(bsz, 1, dtype=torch.int32)
    past_key_value = DummyStaticCache(page_size=16)
    q_len_raw = torch.tensor([q_len], dtype=torch.int32)
    kv_len_raw = torch.tensor([q_len], dtype=torch.int32)

    return (
        hidden_states, attention_mask, position_ids, cache_position,
        page_idx, page_offset, block_table,
        past_key_value, q_len_raw, kv_len_raw
    )


def test_forward_prefill_with_mask():
    """
    is_prefill=True + attention_mask 不为 None + past_key_value 不为 None
    """
    attn = build_attention_module(q_lora_rank=None)

    (hidden_states, attention_mask, position_ids, cache_position,
     page_idx, page_offset, block_table,
     past_key_value, q_len_raw, kv_len_raw) = _common_inputs_prefill()

    outputs = attn.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=False,
        use_cache=True,
        cache_position=cache_position,
        is_prefill=True,
        page_idx=page_idx,
        page_offset=page_offset,
        block_table=block_table,
        q_len_raw=q_len_raw,
        kv_len_raw=kv_len_raw,
        stream=None,
    )

    attn_output, attn_weights, new_cache = outputs
    assert attn_output.shape == (
        1,  # bsz
        3,  # q_len
        attn.num_heads * attn.v_head_dim,
    )
    assert attn_weights is None
    assert new_cache is past_key_value


def test_forward_prefill_without_mask_and_q_lora():
    """
    is_prefill=True + attention_mask=None + q_lora_rank 非 None 分支
    """
    attn = build_attention_module(q_lora_rank=1)

    (hidden_states, attention_mask, position_ids, cache_position,
     page_idx, page_offset, block_table,
     past_key_value, q_len_raw, kv_len_raw) = _common_inputs_prefill()

    outputs = attn.forward(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=False,
        use_cache=True,
        cache_position=cache_position,
        is_prefill=True,
        page_idx=None,
        page_offset=None,
        block_table=None,
        q_len_raw=q_len_raw,
        kv_len_raw=kv_len_raw,
        stream=None,
    )

    attn_output, attn_weights, new_cache = outputs
    assert attn_output.shape == (
        1,
        3,
        attn.num_heads * attn.v_head_dim,
    )
    assert attn_weights is None
    assert new_cache is past_key_value


def test_forward_decode_paged_path():
    """
    is_prefill=False + get_use_npu_graph=False
    => 走 forward_paged + torch.ops.npu.npu_fused_infer_attention_score 分支
    """
    attn = build_attention_module(q_lora_rank=None)

    bsz, q_len, hidden_dim = 1, 1, 4
    hidden_states = torch.randn(bsz, q_len, hidden_dim, dtype=torch.float16)
    position_ids = torch.arange(q_len).unsqueeze(0)
    cache_position = torch.arange(q_len).unsqueeze(0)
    past_key_value = DummyStaticCache(page_size=16)
    q_len_raw = torch.tensor([q_len], dtype=torch.int32)
    kv_len_raw = torch.tensor([q_len], dtype=torch.int32)
    block_table = torch.zeros(bsz, 1, dtype=torch.int32)

    outputs = attn.forward(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=False,
        use_cache=True,
        cache_position=cache_position,
        is_prefill=False,
        page_idx=None,
        page_offset=None,
        block_table=block_table,
        q_len_raw=q_len_raw,
        kv_len_raw=kv_len_raw,
        stream=None,
    )

    attn_output, attn_weights, new_cache = outputs
    assert attn_output.shape == (
        bsz,
        q_len,
        attn.num_heads * attn.v_head_dim,
    )
    assert attn_weights is None
    assert new_cache is past_key_value


def test_forward_prefill_layer_idx_none_raises():
    """
    覆盖: past_key_value 不为 None 且 layer_idx 为 None 的异常分支。
    """
    attn = build_attention_module(q_lora_rank=None)
    attn.layer_idx = None  # 手动破坏 layer_idx

    (hidden_states, attention_mask, position_ids, cache_position,
     page_idx, page_offset, block_table,
     past_key_value, q_len_raw, kv_len_raw) = _common_inputs_prefill()

    with pytest.raises(ValueError):
        attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            is_prefill=True,
            page_idx=page_idx,
            page_offset=page_offset,
            block_table=block_table,
            q_len_raw=q_len_raw,
            kv_len_raw=kv_len_raw,
            stream=None,
        )


def test_forward_prefill_attn_output_shape_mismatch_raises(monkeypatch):
    """
    覆盖: attn_output 形状不符合期望时的 ValueError 分支。
    """
    attn = build_attention_module(q_lora_rank=None)

    def bad_fused(q, k, v, **kwargs):
        bsz, max_q_len, num_heads, dim = q.shape
        # 刻意制造 num_heads+1，触发 size 检查不通过
        out = torch.zeros(
            bsz, max_q_len, num_heads + 1, attn.v_head_dim,
            dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(1, dtype=q.dtype, device=q.device)
        return out, lse

    monkeypatch.setattr(
        torch_npu, "npu_fused_infer_attention_score",
        bad_fused, raising=False
    )

    (hidden_states, attention_mask, position_ids, cache_position,
     page_idx, page_offset, block_table,
     past_key_value, q_len_raw, kv_len_raw) = _common_inputs_prefill()

    with pytest.raises(ValueError):
        attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            is_prefill=True,
            page_idx=page_idx,
            page_offset=page_offset,
            block_table=block_table,
            q_len_raw=q_len_raw,
            kv_len_raw=kv_len_raw,
            stream=None,
        )


def test_forward_paged_use_npu_graph(monkeypatch):
    """
    覆盖: get_use_npu_graph() == True 的 graph 路径。
    """
    # 让 ascend_attention.get_use_npu_graph 返回 True
    monkeypatch.setattr(attn_mod, "get_use_npu_graph", lambda: True)

    # 伪造 model_runner 模块，满足 import ktransformers.server.balance_serve.inference.model_runner
    dummy_runner = type(
        "DummyRunner", (), {"__init__": lambda self: setattr(self, "workspace", [None] * 4)}
    )

    dummy_mr = types.SimpleNamespace(
        ModelRunner=dummy_runner,
        get_or_create_model_runner=lambda device=None: dummy_runner(),
    )

    sys.modules[
        "ktransformers.server.balance_serve.inference.model_runner"
    ] = dummy_mr

    attn = build_attention_module(q_lora_rank=None)

    bsz, q_len, hidden_dim = 1, 1, 4
    hidden_states = torch.randn(bsz, q_len, hidden_dim, dtype=torch.float16)
    position_ids = torch.arange(q_len).unsqueeze(0)
    cache_position = torch.arange(q_len).unsqueeze(0)
    past_key_value = DummyStaticCache(page_size=16)
    q_len_raw = torch.tensor([q_len], dtype=torch.int32)
    kv_len_raw = torch.tensor([q_len], dtype=torch.int32)
    block_table = torch.zeros(bsz, 1, dtype=torch.int32)

    outputs = attn.forward(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=False,
        use_cache=True,
        cache_position=cache_position,
        is_prefill=False,
        page_idx=None,
        page_offset=None,
        block_table=block_table,
        q_len_raw=q_len_raw,
        kv_len_raw=kv_len_raw,
        stream=None,
    )

    attn_output, attn_weights, new_cache = outputs
    assert attn_output.shape == (
        bsz,
        q_len,
        attn.num_heads * attn.v_head_dim,
    )
    assert attn_weights is None
    assert new_cache is past_key_value

