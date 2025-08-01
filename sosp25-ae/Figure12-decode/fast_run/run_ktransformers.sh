python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id ktransformers_qw2_bf16_8+0

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_6+2.yaml \
    --config_id ktransformers_qw2_bf16_6+2

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_8+0.yaml \
    --config_id ktransformers_qw2_int8_8+0

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_4+4.yaml \
    --config_id ktransformers_qw2_int8_4+4
