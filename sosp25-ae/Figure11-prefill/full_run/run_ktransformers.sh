python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_8+0.yaml \
    --config_id ktransformers_ds3_bf16

python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_8+0.yaml \
    --config_id ktransformers_ds3_int4

python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_6+0.yaml \
    --config_id ktransformers_ds2_bf16

python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_6+0.yaml \
    --config_id ktransformers_ds2_int8

python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id ktransformers_qw2_bf16

python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_8+0.yaml \
    --config_id ktransformers_qw2_int8