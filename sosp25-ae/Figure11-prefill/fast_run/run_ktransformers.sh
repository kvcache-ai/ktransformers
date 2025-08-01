python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id ktransformers_qw2_bf16

python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_8+0.yaml \
    --config_id ktransformers_qw2_int8