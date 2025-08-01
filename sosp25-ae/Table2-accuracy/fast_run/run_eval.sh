# Qwen2-57B (qw2) evaluations
# mbpp
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_8+0.yaml \
    --task_name mbpp \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/mbpp/qw2_int8_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_4+4.yaml \
    --task_name mbpp \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/mbpp/qw2_int8_4+4.jsonl
