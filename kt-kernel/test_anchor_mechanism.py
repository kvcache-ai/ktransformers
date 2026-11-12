#!/usr/bin/env python3
"""
Test script for anchor mechanism in KV Cache.
Tests the full pipeline matching the benchmark pattern.
"""

import sys
import os
import numpy as np
import torch

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import kt_kernel_ext
from kt_kernel_ext import kvcache

def test_anchor_mechanism_basic():
    """Test basic anchor mechanism matching the benchmark pattern."""

    print("=" * 60)
    print("Testing Basic Anchor Mechanism (Benchmark Pattern)")
    print("=" * 60)

    # Configuration matching benchmark
    layer_num = 2
    kv_head_num = 4
    q_head_num = 8  # GQA: 2 query heads per kv head
    head_dim = 64
    block_len = 32
    anchor_num = 1  # Benchmark uses 1
    max_block_num = 8
    max_batch_size = 1
    max_thread_num = 4
    cache_seqlen = 64  # 2 blocks

    # Create KVCache config (positional arguments only)
    config = kvcache.KVCacheConfig(
        layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num,
        kvcache.AnchorType.DYNAMIC,  # Benchmark uses DYNAMIC
        kvcache.ggml_type.FP16,
        kvcache.RetrievalType.LAYER,
        1,  # layer_step
        1,  # token_step
        0,  # layer_offset
        max_block_num,
        max_batch_size,
        max_thread_num
    )

    # Create KVCache and WorkerPool
    kv_cache = kvcache.KVCache(config)
    backend = kt_kernel_ext.WorkerPool(max_thread_num)

    print(f"Config: layer_num={layer_num}, kv_head_num={kv_head_num}, q_head_num={q_head_num}")
    print(f"         head_dim={head_dim}, block_len={block_len}, anchor_num={anchor_num}")
    print(f"         max_block_num={max_block_num}, cache_seqlen={cache_seqlen}")
    print(f"         anchor_type=DYNAMIC, kv_type=FP16, retrieval_type=LAYER")
    print()

    # Block table - sequential like benchmark
    block_table = torch.arange(max_block_num, dtype=torch.int32).view(1, -1)
    # cache_seqlens for attn - actual sequence length
    cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32)
    # seqlens_zero for update_kvcache_fp16 - offset
    seqlens_zero = torch.zeros((1,), dtype=torch.int32)

    print(f"Block table shape: {block_table.shape}")
    print(f"Cache seqlens: {cache_seqlens}")
    print(f"Seqlens zero: {seqlens_zero}")
    print()

    # Create random K/V data for the sequence
    np.random.seed(42)
    torch.manual_seed(42)

    # K/V data: [batch_size, seq_len, kv_head_num, head_dim]
    k_data_np = np.random.randn(1, cache_seqlen, kv_head_num, head_dim).astype(np.float16)
    v_data_np = np.random.randn(1, cache_seqlen, kv_head_num, head_dim).astype(np.float16)

    k_data = torch.from_numpy(k_data_np)
    v_data = torch.from_numpy(v_data_np)

    print(f"K data shape: {k_data.shape}")
    print(f"V data shape: {v_data.shape}")
    print()

    # Step 1: Update KV cache with K/V data
    print("Step 1: Updating KV cache with K/V data...")
    for layer_idx in range(layer_num):
        kv_cache.update_kvcache_fp16(
            k_in=k_data.data_ptr(),
            v_in=v_data.data_ptr(),
            layer_id=layer_idx,
            block_table=block_table.data_ptr(),
            batch_size=1,
            max_block_num=max_block_num,
            cache_seqlens=seqlens_zero.data_ptr(),  # offset = 0
            q_len=cache_seqlen,  # length to update
            backend=backend
        )
    print("  KV cache updated successfully")
    print()

    # Step 2: Run attention (this should internally compute anchors if needed)
    print("Step 2: Running attention with KV cache...")

    # Query: [batch_size, 1, q_head_num, head_dim]
    q_data_np = np.random.randn(1, 1, q_head_num, head_dim).astype(np.float16)
    q_data = torch.from_numpy(q_data_np)

    output = torch.empty((1, 1, q_head_num, head_dim), dtype=torch.float16)
    attn_lse = torch.empty((1, 1, q_head_num), dtype=torch.float32)

    # Use -1 for pick_block_num, init_block_num, local_block_num to use all blocks
    layer_idx = 0
    kv_cache.attn(
        q_in=q_data.data_ptr(),
        output=output.data_ptr(),
        attn_lse=attn_lse.data_ptr(),
        layer_idx=layer_idx,
        generate_token_idx=0,
        q_len=1,
        batch_size=1,
        max_block_num=max_block_num,
        block_table=block_table.data_ptr(),
        cache_seqlens=cache_seqlens.data_ptr(),
        pick_block_num=-1,  # Use all blocks
        init_block_num=-1,
        local_block_num=-1,
        backend=backend
    )
    print("  Attention completed")
    print()

    # Step 3: Compute dense attention baseline
    print("Step 3: Computing dense attention baseline...")

    # Reshape for attention
    # k_data: [1, seq_len, kv_head_num, head_dim] -> [1, kv_head_num, seq_len, head_dim]
    k_dense = k_data.transpose(1, 2)  # [1, kv_head_num, seq_len, head_dim]
    v_dense = v_data.transpose(1, 2)  # [1, kv_head_num, seq_len, head_dim]
    q_dense = q_data.transpose(1, 2)  # [1, q_head_num, 1, head_dim]

    # For GQA, repeat K/V for each query head
    n_gqa = q_head_num // kv_head_num
    k_dense = k_dense.unsqueeze(2).expand(-1, -1, n_gqa, -1, -1).reshape(1, cache_seqlen, q_head_num, head_dim).transpose(1, 2)
    v_dense = v_dense.unsqueeze(2).expand(-1, -1, n_gqa, -1, -1).reshape(1, cache_seqlen, q_head_num, head_dim).transpose(1, 2)
    q_dense = q_dense  # [1, q_head_num, 1, head_dim]

    # Compute attention
    scale = 1.0 / (head_dim ** 0.5)
    attn_scores = torch.matmul(q_dense.to(torch.float32), k_dense.to(torch.float32).transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output_dense = torch.matmul(attn_weights, v_dense.to(torch.float32)).to(torch.float16)
    attn_lse_dense = torch.logsumexp(attn_scores, dim=-1)

    print(f"  Dense output shape: {output_dense.shape}")
    print(f"  Sparse output shape: {output.shape}")
    print()

    # Step 4: Compare
    print("Step 4: Comparing sparse vs dense attention...")

    output_sparse_flat = output.flatten().to(torch.float32)
    output_dense_flat = output_dense.flatten()
    attn_lse_sparse_flat = attn_lse.flatten()
    attn_lse_dense_flat = attn_lse_dense.flatten()

    output_diff = torch.abs(output_sparse_flat - output_dense_flat)
    output_rel_error = output_diff / (torch.abs(output_dense_flat) + 1e-6)

    lse_diff = torch.abs(attn_lse_sparse_flat - attn_lse_dense_flat)
    lse_rel_error = lse_diff / (torch.abs(attn_lse_dense_flat) + 1e-6)

    print(f"  Output max abs diff: {output_diff.max().item():.6f}")
    print(f"  Output mean abs diff: {output_diff.mean().item():.6f}")
    print(f"  Output max rel error: {output_rel_error.max().item():.6f}")
    print(f"  Output mean rel error: {output_rel_error.mean().item():.6f}")
    print()
    print(f"  LSE max abs diff: {lse_diff.max().item():.6f}")
    print(f"  LSE mean abs diff: {lse_diff.mean().item():.6f}")
    print(f"  LSE max rel error: {lse_rel_error.max().item():.6f}")
    print(f"  LSE mean rel error: {lse_rel_error.mean().item():.6f}")
    print()

    max_output_rel_error = output_rel_error.max().item()
    max_lse_rel_error = lse_rel_error.max().item()

    tolerance = 0.15  # Slightly higher tolerance for sparse attention

    if max_output_rel_error < tolerance and max_lse_rel_error < tolerance:
        print(f"✓ TEST PASSED: Errors within tolerance ({tolerance})")
        return True
    else:
        print(f"✗ TEST FAILED: Errors exceed tolerance ({tolerance})")
        return False


def test_different_anchor_types():
    """Test different anchor types with the benchmark pattern."""

    print("\n" + "=" * 60)
    print("Testing Different Anchor Types")
    print("=" * 60)

    anchor_types = [
        ("FIXED", kvcache.AnchorType.FIXED),
        ("DYNAMIC", kvcache.AnchorType.DYNAMIC),
        ("QUEST", kvcache.AnchorType.QUEST),
        ("BLOCK_MEAN", kvcache.AnchorType.BLOCK_MEAN),
        ("BLOCK_MAX", kvcache.AnchorType.BLOCK_MAX),
    ]

    results = {}

    for name, anchor_type in anchor_types:
        print(f"\nTesting anchor type: {name}")

        layer_num = 1
        kv_head_num = 2
        q_head_num = 4
        head_dim = 32
        block_len = 16
        anchor_num = 1
        max_block_num = 4
        max_batch_size = 1
        max_thread_num = 2
        cache_seqlen = 32

        config = kvcache.KVCacheConfig(
            layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num,
            anchor_type,
            kvcache.ggml_type.FP16,
            kvcache.RetrievalType.LAYER,
            1, 1, 0,
            max_block_num, max_batch_size, max_thread_num
        )

        kv_cache = kvcache.KVCache(config)
        backend = kt_kernel_ext.WorkerPool(max_thread_num)

        np.random.seed(123)
        k_data_np = np.random.randn(1, cache_seqlen, kv_head_num, head_dim).astype(np.float16)
        v_data_np = np.random.randn(1, cache_seqlen, kv_head_num, head_dim).astype(np.float16)
        q_data_np = np.random.randn(1, 1, q_head_num, head_dim).astype(np.float16)

        k_data = torch.from_numpy(k_data_np)
        v_data = torch.from_numpy(v_data_np)
        q_data = torch.from_numpy(q_data_np)

        block_table = torch.arange(max_block_num, dtype=torch.int32).view(1, -1)
        cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32)
        seqlens_zero = torch.zeros((1,), dtype=torch.int32)

        try:
            # Update KV cache
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

            # Run attention
            output = torch.empty((1, 1, q_head_num, head_dim), dtype=torch.float16)
            attn_lse = torch.empty((1, 1, q_head_num), dtype=torch.float32)

            kv_cache.attn(
                q_in=q_data.data_ptr(),
                output=output.data_ptr(),
                attn_lse=attn_lse.data_ptr(),
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

            # Compute dense baseline
            n_gqa = q_head_num // kv_head_num
            k_dense = k_data.transpose(1, 2).unsqueeze(2).expand(-1, -1, n_gqa, -1, -1).reshape(1, cache_seqlen, q_head_num, head_dim).transpose(1, 2)
            v_dense = v_data.transpose(1, 2).unsqueeze(2).expand(-1, -1, n_gqa, -1, -1).reshape(1, cache_seqlen, q_head_num, head_dim).transpose(1, 2)
            q_dense = q_data.transpose(1, 2)

            scale = 1.0 / (head_dim ** 0.5)
            attn_scores = torch.matmul(q_dense.to(torch.float32), k_dense.to(torch.float32).transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output_dense = torch.matmul(attn_weights, v_dense.to(torch.float32)).to(torch.float16)
            attn_lse_dense = torch.logsumexp(attn_scores, dim=-1)

            # Compare
            output_diff = torch.abs(output.flatten().to(torch.float32) - output_dense.flatten())
            output_rel_error = output_diff / (torch.abs(output_dense.flatten()) + 1e-6)
            max_rel_error = output_rel_error.max().item()

            results[name] = max_rel_error
            print(f"  Max relative error: {max_rel_error:.6f}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = float('inf')

    print("\n" + "-" * 40)
    print("Summary:")
    for name, error in results.items():
        status = "✓" if error < 0.15 else "✗"
        print(f"  {status} {name}: max_rel_error = {error:.6f}")

    return results


def test_update_kvcache_incremental():
    """Test incremental KV cache update (like in generation)."""

    print("\n" + "=" * 60)
    print("Testing Incremental KV Cache Update")
    print("=" * 60)

    layer_num = 1
    kv_head_num = 2
    q_head_num = 4
    head_dim = 32
    block_len = 16
    anchor_num = 1
    max_block_num = 8
    max_batch_size = 1
    max_thread_num = 2

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
    seqlens_zero = torch.zeros((1,), dtype=torch.int32)

    np.random.seed(42)

    # Simulate incremental generation: add 1 token at a time
    for step in range(4):  # 4 steps, each adding 1 block (16 tokens)
        cache_seqlen = (step + 1) * block_len
        cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32)

        # New K/V for this step
        k_new = torch.randn(1, block_len, kv_head_num, head_dim, dtype=torch.float16)
        v_new = torch.randn(1, block_len, kv_head_num, head_dim, dtype=torch.float16)
        q_new = torch.randn(1, 1, q_head_num, head_dim, dtype=torch.float16)

        print(f"  Step {step + 1}: Adding {block_len} tokens (total: {cache_seqlen})")

        # Update KV cache with new tokens
        kv_cache.update_kvcache_fp16(
            k_in=k_new.data_ptr(),
            v_in=v_new.data_ptr(),
            layer_id=0,
            block_table=block_table.data_ptr(),
            batch_size=1,
            max_block_num=max_block_num,
            cache_seqlens=seqlens_zero.data_ptr(),
            q_len=block_len,  # Only adding block_len tokens this step
            backend=backend
        )

        # Run attention
        output = torch.empty((1, 1, q_head_num, head_dim), dtype=torch.float16)
        attn_lse = torch.empty((1, 1, q_head_num), dtype=torch.float32)

        kv_cache.attn(
            q_in=q_new.data_ptr(),
            output=output.data_ptr(),
            attn_lse=attn_lse.data_ptr(),
            layer_idx=0,
            generate_token_idx=step,
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

        # Verify output is not NaN
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"    ✗ Output contains NaN/Inf!")
            return False
        else:
            print(f"    ✓ Output valid (mean={output.mean().item():.4f})")

    print("  ✓ Incremental update test passed")
    return True


if __name__ == "__main__":
    print("Starting Anchor Mechanism Tests\n")

    test1_passed = test_anchor_mechanism_basic()
    anchor_results = test_different_anchor_types()
    test3_passed = test_update_kvcache_incremental()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if test1_passed:
        print("✓ Basic anchor mechanism test: PASSED")
    else:
        print("✗ Basic anchor mechanism test: FAILED")

    all_anchor_passed = all(e < 0.15 for e in anchor_results.values())
    if all_anchor_passed:
        print("✓ All anchor type tests: PASSED")
    else:
        print("✗ Some anchor type tests: FAILED")

    if test3_passed:
        print("✓ Incremental update test: PASSED")
    else:
        print("✗ Incremental update test: FAILED")

    if test1_passed and all_anchor_passed and test3_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        sys.exit(0)
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        sys.exit(1)