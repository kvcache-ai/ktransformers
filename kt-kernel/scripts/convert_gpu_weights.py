#!/usr/bin/env python
"""
GPU Weight Quantization Tool for KTransformers

This script quantizes model weights for CPU-GPU hybrid inference when integrating
KTransformers with SGLang. It applies selective quantization (GPTQ) to GPU-resident
layers while preserving certain components (e.g., attention, gates, shared experts)
in higher precision.

Usage:
    python convert_gpu_weights.py --model_id /path/to/model --output_dir /path/to/output --quant_type W4A16

Example:
    python convert_gpu_weights.py \
        --model_id /mnt/data2/models/Qwen3-Next-80B-A3B-Instruct \
        --output_dir /mnt/data2/models/Qwen3-Next-80B-A3B-Instruct-GPU-weight \
        --quant_type W4A16
    python convert_gpu_weights.py \
        --model_id /mnt/data/models/GLM-4.5-Air \
        --output_dir /mnt/data/models/GLM-4.5-Air-GPU-weights-test \
        --quant_type W4A16
"""

import os
import warnings
import argparse
import torch
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize MoE models with selective quantization")
    
    # Required arguments
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Path to the input model directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Path to save the quantized model"
    )
    
    # Optional arguments
    parser.add_argument(
        "--quant_type",
        type=str,
        choices=["W4A16", "W8A16"],
        default="W8A16",
        help="Quantization type: W4A16 (GPTQ4) or W8A16 (GPTQ8). Default: W8A16"
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=512,
        help="Number of calibration samples. Default: 512"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration. Default: 2048"
    )
    parser.add_argument(
        "--dampening_frac",
        type=float,
        default=0.1,
        help="Dampening fraction to mitigate quantization noise. Default: 0.1"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset for calibration. Default: HuggingFaceH4/ultrachat_200k"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train_sft",
        help="Dataset split to use. Default: train_sft"
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force all computations to CPU (sets CUDA_VISIBLE_DEVICES='')"
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        nargs="*",
        default=[
            "lm_head",
            r"re:.*\.mlp\.gate$",
            r"re:.*\.self_attn\..*$",
            r"re:.*\.shared_expert\..*$",
            r"re:.*\.shared_experts\..*$",
            r"re:.*\.mlp\.shared_expert_gate$",
            r"re:.*\.linear_attn\..*$"
        ],
        help="Regex patterns for layers to ignore during quantization"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="PyTorch dtype for model loading. Default: bfloat16"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading of remote code (required for some models)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling. Default: 42"
    )
    parser.add_argument(
        "--max_gpu_memory",
        type=str,
        default=None,
        help="Maximum GPU memory for model weights per device (e.g., '40GiB'). "
             "GPTQ quantization requires additional GPU memory for Hessian matrix computation, "
             "so reserve 40-50%% of total VRAM. For example, use '40GiB' on 80GB GPUs. "
             "Remaining layers will be offloaded to CPU. Default: use all available"
    )
    parser.add_argument(
        "--max_cpu_memory",
        type=str,
        default=None,
        help="Maximum CPU memory to use (e.g., '100GiB'). Default: use all available"
    )
    
    return parser.parse_args()


def setup_environment(force_cpu=False):
    """
    Setup environment variables and warnings.

    Args:
        force_cpu: If True, forces all computations to CPU by hiding GPUs
    """
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        print("🔧 Forced CPU-only mode")


def get_torch_dtype(dtype_str):
    """
    Convert string to torch dtype.

    Args:
        dtype_str: String representation of dtype ("bfloat16", "float16", "float32")

    Returns:
        torch.dtype: Corresponding PyTorch dtype
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    return dtype_map[dtype_str]


def check_dense_layers_and_update_ignore(model_id, ignore_patterns, trust_remote_code=False):
    """
    Check if the model has dense layers (first_k_dense_replace parameter) and add them to ignore list.

    Some MoE models have dense MLP layers in the first few layers instead of MoE layers.
    These dense layers should not be quantized using the same scheme as expert layers.

    Args:
        model_id: Path to the model
        ignore_patterns: List of existing ignore patterns
        trust_remote_code: Whether to trust remote code

    Returns:
        Updated ignore_patterns list with dense layer patterns added
    """
    print("🔍 Checking model configuration for dense layers...")
    
    try:
        # Load model configuration
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        
        # Check if the model has first_k_dense_replace parameter
        first_k_dense_replace = getattr(config, 'first_k_dense_replace', None)
        
        if first_k_dense_replace is not None and first_k_dense_replace > 0:
            print(f"✅ Found dense layers configuration: first_k_dense_replace = {first_k_dense_replace}")
            print(f"   Adding first {first_k_dense_replace} layers to ignore list...")
            
            # Create regex pattern for dense layers (layers 0 to first_k_dense_replace-1)
            if first_k_dense_replace == 1:
                dense_pattern = r"re:model\.layers\.0\.mlp\..*$"
            else:
                # For multiple layers, use range pattern
                layer_range = f"[0-{first_k_dense_replace-1}]"
                dense_pattern = f"re:model\\.layers\\.{layer_range}\\.mlp\\..*$"
            
            # Add the dense layer pattern to ignore list
            updated_ignore_patterns = ignore_patterns + [dense_pattern]
            
            print(f"   Dense layer pattern added: {dense_pattern}")
            print(f"   This will ignore MLP components in layers 0-{first_k_dense_replace-1}")
            
            return updated_ignore_patterns
        else:
            print("ℹ️  No dense layers detected (first_k_dense_replace not found or is 0)")
            return ignore_patterns
            
    except Exception as e:
        print(f"⚠️  Warning: Could not check model config for dense layers: {e}")
        print("   Proceeding with original ignore patterns...")
        return ignore_patterns


def load_and_prepare_dataset(dataset_name, dataset_split, num_samples, max_length, tokenizer, seed=42):
    """
    Load and prepare calibration dataset for GPTQ quantization.

    GPTQ requires calibration data to compute optimal quantization parameters.
    This function loads a conversation dataset, applies chat template, and tokenizes it.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use (e.g., "train_sft")
        num_samples: Number of samples to use for calibration
        max_length: Maximum sequence length for tokenization
        tokenizer: Model tokenizer
        seed: Random seed for shuffling

    Returns:
        Dataset with tokenized calibration samples
    """
    print(f"📊 Loading dataset: {dataset_name}")

    # Load dataset
    ds = load_dataset(dataset_name, split=f"{dataset_split}[:{num_samples}]")
    ds = ds.shuffle(seed=seed)

    # Preprocess the data into the format the model is trained with
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(preprocess)

    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    print(f"✅ Dataset prepared with {len(ds)} samples")

    return ds


def main():
    """
    Main function for GPU weight quantization.

    This performs GPTQ quantization on model weights intended for GPU execution
    in CPU-GPU hybrid inference scenarios. The quantization is selective:
    - Expert MLP weights are quantized to INT4/INT8 (GPTQ)
    - Attention layers, gates, and shared experts remain in original precision
    - Dense layers (if present) are excluded from quantization

    The quantized model can be used with SGLang+KTransformers for heterogeneous
    inference, where "hot" experts run on GPU and "cold" experts run on CPU.
    """
    args = parse_args()

    # Setup environment
    setup_environment(args.force_cpu)

    # Convert torch dtype
    torch_dtype = get_torch_dtype(args.torch_dtype)

    print(f"🚀 Starting quantization process")
    print(f"   Model: {args.model_id}")
    print(f"   Output: {args.output_dir}")
    print(f"   Quantization: {args.quant_type}")
    print(f"   Calibration samples: {args.num_calibration_samples}")
    print(f"   Max sequence length: {args.max_sequence_length}")

    # --------------------------------------------------------------------
    # 0) Check for dense layers and update ignore patterns
    # Dense layers in the first few layers should not be quantized
    updated_ignore_patterns = check_dense_layers_and_update_ignore(
        args.model_id,
        args.ignore_patterns,
        args.trust_remote_code
    )

    # --------------------------------------------------------------------
    # 1) Build a dummy model (no weights) to infer a device map
    # This determines optimal device placement for each module
    if args.force_cpu:
        # In force_cpu mode, directly get module names without calling infer_auto_device_map
        # to avoid GPU memory allocation
        print("🔍 Building CPU-only device map...")
        with init_empty_weights():
            dummy = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code
            )
            device_map = {name: "cpu" for name, _ in dummy.named_modules() if name}
            del dummy
    else:
        print("🔍 Inferring device map...")
        with init_empty_weights():
            dummy = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code
            )
            # Build max_memory dict if specified
            max_memory = None
            if args.max_gpu_memory or args.max_cpu_memory:
                max_memory = {}
                if args.max_gpu_memory:
                    # Apply to all available GPUs
                    num_gpus = torch.cuda.device_count()
                    for i in range(num_gpus):
                        max_memory[i] = args.max_gpu_memory
                    print(f"   GPU memory limit: {args.max_gpu_memory} per device ({num_gpus} GPUs)")

                # Always set CPU memory when max_memory is used
                # Otherwise infer_auto_device_map may trigger disk offloading
                if args.max_cpu_memory:
                    max_memory["cpu"] = args.max_cpu_memory
                    print(f"   CPU memory limit: {args.max_cpu_memory}")
                else:
                    # Use a very large value to allow using all available CPU memory
                    # This prevents disk offloading when user has enough RAM
                    max_memory["cpu"] = "1000GiB"
                    print(f"   CPU memory limit: 1000GiB (default, to prevent disk offloading)")

            device_map = infer_auto_device_map(
                dummy,
                no_split_module_classes=dummy._no_split_modules,
                max_memory=max_memory
            )

            # Check if disk offloading was triggered (not supported by llmcompressor)
            disk_modules = [k for k, v in device_map.items() if v == "disk"]
            if disk_modules:
                print(f"❌ Error: {len(disk_modules)} modules would be offloaded to disk.")
                print("   llmcompressor does not support disk offloading.")
                print("   Solutions:")
                print("   1. Increase --max_gpu_memory to use more GPU memory")
                print("   2. Add --max_cpu_memory with higher value (e.g., '200GiB')")
                print("   3. Ensure your machine has enough GPU + CPU memory")
                raise RuntimeError("Disk offloading is not supported by llmcompressor. "
                                 "Please ensure you have enough GPU + CPU memory.")

            del dummy
    # --------------------------------------------------------------------
    # 2) Load the full model weights with device mapping
    # Note: offload_folder=None disables disk offloading (not supported by llmcompressor)
    print("📥 Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            offload_folder=None,  # Disable disk offloading (not supported by llmcompressor)
        )
    except Exception as e:
        if "disk" in str(e).lower() or "offload" in str(e).lower():
            print(f"❌ Error: Not enough GPU + CPU memory to load the model.")
            print("   llmcompressor does not support disk offloading.")
            print("   Solutions:")
            print("   1. Increase --max_gpu_memory to use more GPU memory")
            print("   2. Ensure you have enough CPU RAM for remaining layers")
            print("   3. Use a machine with more memory")
            raise
        raise

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # --------------------------------------------------------------------
    # 3) Prepare calibration dataset
    # GPTQ needs calibration data to compute optimal quantization parameters
    ds = load_and_prepare_dataset(
        args.dataset,
        args.dataset_split,
        args.num_calibration_samples,
        args.max_sequence_length,
        tokenizer,
        args.random_seed
    )

    # --------------------------------------------------------------------
    # 4) Create quantization recipe with selective layer exclusion
    print(f"⚙️  Setting up {args.quant_type} quantization recipe...")
    recipe = GPTQModifier(
        targets="Linear",  # Target all Linear layers
        scheme=args.quant_type,  # W4A16 or W8A16
        ignore=updated_ignore_patterns,  # Exclude specific patterns
        dampening_frac=args.dampening_frac,
    )

    print("🔧 Ignoring the following patterns from quantization:")
    for i, pattern in enumerate(updated_ignore_patterns):
        marker = "🆕" if i >= len(args.ignore_patterns) else "   "
        print(f"   {marker} {pattern}")

    # --------------------------------------------------------------------
    # 5) Perform one-shot GPTQ quantization
    # This applies GPTQ to quantize weights while minimizing accuracy loss
    print("🎯 Starting one-shot quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=args.output_dir,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    print(f"\n✅ Quantized model written to: {args.output_dir}")
    print(f"   Quantization type: {args.quant_type}")
    print(f"   Ignored patterns remain in {args.torch_dtype}")
    print("🎉 Quantization completed successfully!")


if __name__ == "__main__":
    main()