python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_8+0.yaml \
    --config_id ktransformers_ds3_bf16_8+0

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_5+3.yaml \
    --config_id ktransformers_ds3_bf16_5+3
    
python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_8+0.yaml \
    --config_id ktransformers_ds3_int4_8+0

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_2+6.yaml \
    --config_id ktransformers_ds3_int4_2+6

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_6+0.yaml \
    --config_id ktransformers_ds2_bf16_6+0

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_2+4.yaml \
    --config_id ktransformers_ds2_bf16_2+4

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_6+0.yaml \
    --config_id ktransformers_ds2_int8_6+0

python ../../ktransformers-utils/eval_decode.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_2+4.yaml \
    --config_id ktransformers_ds2_int8_2+4

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
