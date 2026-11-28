# SPDX-License-Identifier: Apache-2.0
"""Run with the V4 base Transformers dependency used by this example."""

import copy

import torch

from kt_kernel.sft._configuration_deepseek_v41 import DeepseekV41Config
from kt_kernel.sft._modeling_deepseek_v41 import DeepseekV41ForCausalLM


def test_checkpointed_shared_kv_gradients():
    # Exercise both compressed KV source/consumer pairs without generation caches.
    config = DeepseekV41Config(
        text_config={
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_hidden_layers": 5,
            "compress_ratios": [0, 2, 2, 1, 1],
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "max_position_embeddings": 128,
            "vocab_size": 128,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "q_lora_rank": 32,
            "o_groups": 2,
            "o_lora_rank": 16,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "sliding_window": 8,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 3,
            "hc_eps": 1e-06,
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 2,
            "candidate_topk_blocks": 4,
            "candidate_block_size": 4,
            "num_nextn_predict_layers": 0,
            "scoring_func": "sqrtsoftplus",
            "gate_temp": 1.0,
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "compress_rope_theta": 160000.0,
            "rms_norm_eps": 1e-20,
            "engram_layer_ids": [],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
        }
    )
    config._attn_implementation = "eager"
    config.get_text_config()._attn_implementation = "eager"
    torch.manual_seed(499)
    reference = DeepseekV41ForCausalLM(config).train()
    replay = copy.deepcopy(reference)
    replay.gradient_checkpointing_enable({"use_reentrant": False})
    ids = torch.arange(18).unsqueeze(0)
    for model in (reference, replay):
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    assert reference.model.layers[0].self_attn.q_b_proj.weight.grad is not None
    for expected, actual in zip(reference.parameters(), replay.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
