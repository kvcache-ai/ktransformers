#!/bin/bash

python3 ktransformers/local_chat.py \
    --model_path "/mnt/data/models/DeepSeek-V2-Lite-Chat" \
    --model_config_path "/mnt/data/models/DeepSeek-V2-Lite-Chat" \
    --gguf_path "/mnt/data/models/DeepSeek-V2-Lite-Chat" \
    --cpu_infer 32 \
    --max_new_tokens 1000 \
    --optimize_config_path "ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml" \
    --is_sft False \
    --sft_data_path "test_adapter/sft_translation.json" \
    --save_adapter_path "test_adapter/demo_adapter_KT_target_kv" \
    --use_adapter True \
    --use_adapter_path "/mnt/data/lpl/test_adapter/KT_newLoader_singleGPU_deepseekV2_Neko_AFS/checkpoint-566" \
    --is_test_data False \
    --test_data_path "test_adapter/demo_adapter_origin_target_kv" \
    --output_dir "test_adapter/demo_adapter_origin_target_kv" \