#!/usr/bin/env python3
"""
KV Cache Size Calculator for SGLang

This script calculates the KV cache size in GB for a given model and number of tokens.
It follows the same logic as in sglang/srt/model_executor/model_runner.py
"""

import os
import sys
import torch
from transformers import AutoConfig

# Add sglang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from sglang.srt.configs.model_config import ModelConfig, is_deepseek_nsa, get_nsa_index_head_dim
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool


def get_dtype_bytes(dtype_str: str) -> int:
    """Get the number of bytes for a given dtype string."""
    dtype_map = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "auto": 2,  # Usually defaults to bfloat16
    }
    return dtype_map.get(dtype_str, 2)


def get_kv_size_gb(
    model_path: str,
    max_total_tokens: int,
    tp: int = 1,
    dtype: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    Calculate the KV cache size in GB for a given model and number of tokens.

    Args:
        model_path: Path to the model
        max_total_tokens: Maximum number of tokens to cache
        tp: Tensor parallelism size
        dtype: Data type for KV cache (auto, float16, bfloat16, float8_e4m3fn, etc.)
        verbose: Whether to print detailed information

    Returns:
        dict: Dictionary containing calculation details
    """
    # Load model config
    model_config = ModelConfig(model_path, dtype=dtype)
    hf_config = model_config.hf_config

    # Determine dtype bytes
    dtype_bytes = get_dtype_bytes(dtype)
    if dtype == "auto":
        # Auto dtype usually becomes bfloat16
        dtype_bytes = 2

    # Number of layers
    num_layers = model_config.num_attention_layers

    # Check if it's MLA (Multi-head Latent Attention) model
    is_mla = hasattr(model_config, "attention_arch") and model_config.attention_arch.name == "MLA"

    result = {
        "model_path": model_path,
        "max_total_tokens": max_total_tokens,
        "tp": tp,
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "num_layers": num_layers,
        "is_mla": is_mla,
    }

    if is_mla:
        # MLA models (DeepSeek-V2/V3, MiniCPM3, etc.)
        kv_lora_rank = model_config.kv_lora_rank
        qk_rope_head_dim = model_config.qk_rope_head_dim

        # Calculate cell size (per token)
        cell_size = (kv_lora_rank + qk_rope_head_dim) * num_layers * dtype_bytes

        result.update(
            {
                "kv_lora_rank": kv_lora_rank,
                "qk_rope_head_dim": qk_rope_head_dim,
                "cell_size_bytes": cell_size,
            }
        )

        # Check if it's NSA (Native Sparse Attention) model
        if is_deepseek_nsa(hf_config):
            index_head_dim = get_nsa_index_head_dim(hf_config)
            indexer_size_per_token = index_head_dim + index_head_dim // NSATokenToKVPool.quant_block_size * 4
            indexer_dtype_bytes = torch._utils._element_size(NSATokenToKVPool.index_k_with_scale_buffer_dtype)
            indexer_cell_size = indexer_size_per_token * num_layers * indexer_dtype_bytes
            cell_size += indexer_cell_size

            result.update(
                {
                    "is_nsa": True,
                    "index_head_dim": index_head_dim,
                    "indexer_cell_size_bytes": indexer_cell_size,
                    "total_cell_size_bytes": cell_size,
                }
            )
        else:
            result["is_nsa"] = False
    else:
        # Standard MHA models
        num_kv_heads = model_config.get_num_kv_heads(tp)
        head_dim = model_config.head_dim
        v_head_dim = model_config.v_head_dim

        # Calculate cell size (per token)
        cell_size = num_kv_heads * (head_dim + v_head_dim) * num_layers * dtype_bytes

        result.update(
            {
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "v_head_dim": v_head_dim,
                "cell_size_bytes": cell_size,
            }
        )

    # Calculate total KV cache size
    total_size_bytes = max_total_tokens * cell_size
    total_size_gb = total_size_bytes / (1024**3)

    # For MHA models with separate K and V buffers
    if not is_mla:
        k_size_bytes = max_total_tokens * num_kv_heads * head_dim * num_layers * dtype_bytes
        v_size_bytes = max_total_tokens * num_kv_heads * v_head_dim * num_layers * dtype_bytes
        k_size_gb = k_size_bytes / (1024**3)
        v_size_gb = v_size_bytes / (1024**3)

        result.update(
            {
                "k_size_gb": k_size_gb,
                "v_size_gb": v_size_gb,
            }
        )

    result.update(
        {
            "total_size_bytes": total_size_bytes,
            "total_size_gb": total_size_gb,
        }
    )

    if verbose:
        print(f"Model: {model_path}")
        print(f"Tokens: {max_total_tokens}, TP: {tp}, Dtype: {dtype}")
        print(f"Architecture: {'MLA' if is_mla else 'MHA'}")
        print(f"Layers: {num_layers}")

        if is_mla:
            print(f"KV LoRA Rank: {kv_lora_rank}, QK RoPE Head Dim: {qk_rope_head_dim}")
            if result.get("is_nsa"):
                print(f"NSA Index Head Dim: {index_head_dim}")
                print(
                    f"Cell size: {cell_size} bytes (Main: {result['cell_size_bytes']}, Indexer: {result['indexer_cell_size_bytes']})"
                )
            else:
                print(f"Cell size: {cell_size} bytes")
        else:
            print(f"KV Heads: {num_kv_heads}, Head Dim: {head_dim}, V Head Dim: {v_head_dim}")
            print(f"Cell size: {cell_size} bytes")
            print(f"K size: {k_size_gb:.2f} GB, V size: {v_size_gb:.2f} GB")

        print(f"Total KV Cache Size: {total_size_gb:.2f} GB")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate KV cache size for a model")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("max_total_tokens", type=int, help="Maximum number of tokens")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type (auto, float16, bfloat16, etc.)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    result = get_kv_size_gb(
        args.model_path,
        args.max_total_tokens,
        tp=args.tp,
        dtype=args.dtype,
        verbose=not args.quiet,
    )

    if args.quiet:
        print(f"{result['total_size_gb']:.2f}")


if __name__ == "__main__":
    main()
