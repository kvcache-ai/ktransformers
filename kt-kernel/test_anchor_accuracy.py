#!/usr/bin/env python3
"""
Test anchor mechanism accuracy with varying anchor numbers.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import kt_kernel_ext
from kt_kernel_ext import kvcache

def test_anchor_accuracy(anchor_num):
    """Test accuracy with given number of anchors."""

    layer_num = 1
    kv_head_num = 2
    q_head_num = 4
    head_dim = 32
    block_len = 16
    max_block_num = 8
    max_batch_size = 1
    max_thread_num = 2
    cache_seqlen = 64

    config = kvcache.KVCacheConfig(
        layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num,
        kvcache.AnchorType.DYNAMIC,
        kvcache.ggml_type.FP16,
        kvcache.RetrievalType.LAYER,
        1, 1, 0,
        max_block_num, max_batch_size, max_thread_num
    )

    kv_cache = kvcache.KVCache(config)
    backend = kt_kernel_ext.WorkerPool(max_thread_num)

    block_table = torch.arange(max_block_num, dtype=torch.int32).view(1, -1)
    cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32)
    seqlens_zero = torch.zeros((1,), dtype=torch.int32)

    np.random.seed(42)
    torch.manual_seed(42)

    k_data = torch.randn(1, cache_seqlen, kv_head_num, head_dim, dtype=torch.float16)
    v_data = torch.randn(1, cache_seqlen, kv_head_num, head_dim, dtype=torch.float16)

    kv_cache.update_kvcache_fp16(
        k_in=k_data.data_ptr(),
        v_in=v_data.data_ptr(),
        layer_id=0,
        block_table=block_table.data_ptr(),
        batch_size=1,
        max_block_num=max_block_num,
        cache_seqlens=seqlens_zero.data_ptr(),
        q_len=cache_seqlen,
        backend=backend
    )

    q_data = torch.randn(1, 1, q_head_num, head_dim, dtype=torch.float16)

    output_sparse = torch.empty((1, 1, q_head_num, head_dim), dtype=torch.float16)
    attn_lse_sparse = torch.empty((1, 1, q_head_num), dtype=torch.float32)

    kv_cache.attn(
        q_in=q_data.data_ptr(),
        output=output_sparse.data_ptr(),
        attn_lse=attn_lse_sparse.data_ptr(),
        layer_idx=0,
        generate_token_idx=0,
        q_len=1,
        batch_size=1,
        max_block_num=max_block_num,
        block_table=block_table.data_ptr(),
        cache_seqlens=cache_seqlens.data_ptr(),
        pick_block_num=-1,
        init_block_num=-1,
        local_block_num=-1,
        backend=backend
    )

    # Dense baseline
    n_gqa = q_head_num // kv_head_num
    k_dense = k_data.transpose(1, 2).unsqueeze(2).expand(-1, -1, n_gqa, -1, -1).reshape(1, cache_seqlen, q_head_num, head_dim).transpose(1, 2)
    v_dense = v_data.transpose(1, 2).unsqueeze(2).expand(-1, -1, n_gqa, -1, -1).reshape(1, cache_seqlen, q_head_num, head_dim).transpose(1, 2)
    q_dense = q_data.transpose(1, 2)

    scale = 1.0 / (head_dim ** 0.5)
    attn_scores = torch.matmul(q_dense.to(torch.float32), k_dense.to(torch.float32).transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output_dense = torch.matmul(attn_weights, v_dense.to(torch.float32)).to(torch.float16)
    attn_lse_dense = torch.logsumexp(attn_scores, dim=-1)

    output_sparse_reshaped = output_sparse.transpose(1, 2)
    attn_lse_sparse_reshaped = attn_lse_sparse.transpose(1, 2)

    output_diff = torch.abs(output_sparse_reshaped.flatten().to(torch.float32) - output_dense.flatten())
    output_rel_error = output_diff / (torch.abs(output_dense.flatten()) + 1e-6)

    lse_diff = torch.abs(attn_lse_sparse_reshaped.flatten() - attn_lse_dense.flatten())
    lse_rel_error = lse_diff / (torch.abs(attn_lse_dense.flatten()) + 1e-6)

    max_output_rel = output_rel_error.max().item()
    mean_output_rel = output_rel_error.mean().item()
    max_lse_rel = lse_rel_error.max().item()
    mean_lse_rel = lse_rel_error.mean().item()

    has_nan = torch.isnan(output_sparse).any() or torch.isinf(output_sparse).any()

    return {
        'anchor_num': anchor_num,
        'max_output_rel': max_output_rel,
        'mean_output_rel': mean_output_rel,
        'max_lse_rel': max_lse_rel,
        'mean_lse_rel': mean_lse_rel,
        'has_nan': has_nan
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Anchor Accuracy vs Number of Anchors")
    print("=" * 70)

    for anchor_num in [1, 2, 4]:
        print(f"\nTesting with anchor_num = {anchor_num}...")
        try:
            result = test_anchor_accuracy(anchor_num)

            status = "✗ NaN" if result['has_nan'] else "✓ OK"
            print(f"  {status}")
            print(f"  Output - Max rel error: {result['max_output_rel']:.4f}, Mean: {result['mean_output_rel']:.4f}")
            print(f"  LSE    - Max rel error: {result['max_lse_rel']:.4f}, Mean: {result['mean_lse_rel']:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("Summary: More anchors = better accuracy (but more computation)")
    print("=" * 70)