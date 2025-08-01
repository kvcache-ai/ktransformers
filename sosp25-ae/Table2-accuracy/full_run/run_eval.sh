# DeepSeek-V3 (ds3) evaluations
# human_eval
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_8+0.yaml \
    --task_name human_eval \
    --seed 2025 \
    --do_sample True \
    --temperature 0.3 \
    --num_trials 10 \
    --output_file ./results/human_eval/ds3_int4_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_2+6.yaml \
    --task_name human_eval \
    --seed 2025 \
    --do_sample True \
    --temperature 0.3 \
    --num_trials 10 \
    --output_file ./results/human_eval/ds3_int4_2+6.jsonl

# mbpp
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_8+0.yaml \
    --task_name mbpp \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/mbpp/ds3_int4_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_2+6.yaml \
    --task_name mbpp \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/mbpp/ds3_int4_2+6.jsonl

# gsm8k
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_8+0.yaml \
    --task_name gsm8k \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/gsm8k/ds3_int4_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_2+6.yaml \
    --task_name gsm8k \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/gsm8k/ds3_int4_2+6.jsonl

# strategy_qa
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_8+0.yaml \
    --task_name strategy_qa \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/strategy_qa/ds3_int4_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_int4_2+6.yaml \
    --task_name strategy_qa \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/strategy_qa/ds3_int4_2+6.jsonl


# DeepSeek-V2.5 (ds2) evaluations
# human_eval
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_6+0.yaml \
    --task_name human_eval \
    --seed 2025 \
    --do_sample True \
    --temperature 0.3 \
    --num_trials 10 \
    --output_file ./results/human_eval/ds2_int8_6+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_2+4.yaml \
    --task_name human_eval \
    --seed 2025 \
    --do_sample True \
    --temperature 0.3 \
    --num_trials 10 \
    --output_file ./results/human_eval/ds2_int8_2+4.jsonl

# mbpp
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_6+0.yaml \
    --task_name mbpp \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/mbpp/ds2_int8_6+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_2+4.yaml \
    --task_name mbpp \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/mbpp/ds2_int8_2+4.jsonl

# gsm8k
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_6+0.yaml \
    --task_name gsm8k \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/gsm8k/ds2_int8_6+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_2+4.yaml \
    --task_name gsm8k \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/gsm8k/ds2_int8_2+4.jsonl

# strategy_qa
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_6+0.yaml \
    --task_name strategy_qa \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/strategy_qa/ds2_int8_6+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$DS2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_int8_2+4.yaml \
    --task_name strategy_qa \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/strategy_qa/ds2_int8_2+4.jsonl


# Qwen2-57B (qw2) evaluations
# human_eval
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_8+0.yaml \
    --task_name human_eval \
    --seed 2025 \
    --do_sample True \
    --temperature 0.3 \
    --num_trials 10 \
    --output_file ./results/human_eval/qw2_int8_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_4+4.yaml \
    --task_name human_eval \
    --seed 2025 \
    --do_sample True \
    --temperature 0.3 \
    --num_trials 10 \
    --output_file ./results/human_eval/qw2_int8_4+4.jsonl

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

# gsm8k
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_8+0.yaml \
    --task_name gsm8k \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/gsm8k/qw2_int8_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_4+4.yaml \
    --task_name gsm8k \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/gsm8k/qw2_int8_4+4.jsonl

# strategy_qa
python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_8+0.yaml \
    --task_name strategy_qa \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/strategy_qa/qw2_int8_8+0.jsonl

python ../../ktransformers-utils/eval_accuracy.py \
    --model_path "$QW2_SAFETENSORS"/ \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_int8_4+4.yaml \
    --task_name strategy_qa \
    --do_sample False \
    --num_trials 1 \
    --output_file ./results/strategy_qa/qw2_int8_4+4.jsonl