# DeepSeek-V3 (ds3) evaluations
# human_eval
python ../../ktransformers-utils/scoring.py \
    --task_name human_eval \
    --load_generations_path ./results/human_eval/ds3_int4_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name human_eval \
    --load_generations_path ./results/human_eval/ds3_int4_2+6.jsonl \
    --metric_output_path scores.jsonl

# mbpp
python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/ds3_int4_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/ds3_int4_2+6.jsonl \
    --metric_output_path scores.jsonl

# gsm8k
python ../../ktransformers-utils/scoring.py \
    --task_name gsm8k \
    --load_generations_path ./results/gsm8k/ds3_int4_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name gsm8k \
    --load_generations_path ./results/gsm8k/ds3_int4_2+6.jsonl \
    --metric_output_path scores.jsonl

# strategy_qa
python ../../ktransformers-utils/scoring.py \
    --task_name strategy_qa \
    --load_generations_path ./results/strategy_qa/ds3_int4_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name strategy_qa \
    --load_generations_path ./results/strategy_qa/ds3_int4_2+6.jsonl \
    --metric_output_path scores.jsonl


# DeepSeek-V2.5 (ds2) evaluations
# human_eval
python ../../ktransformers-utils/scoring.py \
    --task_name human_eval \
    --load_generations_path ./results/human_eval/ds2_int8_6+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name human_eval \
    --load_generations_path ./results/human_eval/ds2_int8_2+4.jsonl \
    --metric_output_path scores.jsonl

# mbpp
python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/ds2_int8_6+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/ds2_int8_2+4.jsonl \
    --metric_output_path scores.jsonl

# gsm8k
python ../../ktransformers-utils/scoring.py \
    --task_name gsm8k \
    --load_generations_path ./results/gsm8k/ds2_int8_6+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name gsm8k \
    --load_generations_path ./results/gsm8k/ds2_int8_2+4.jsonl \
    --metric_output_path scores.jsonl

# strategy_qa
python ../../ktransformers-utils/scoring.py \
    --task_name strategy_qa \
    --load_generations_path ./results/strategy_qa/ds2_int8_6+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name strategy_qa \
    --load_generations_path ./results/strategy_qa/ds2_int8_2+4.jsonl \
    --metric_output_path scores.jsonl


# Qwen2-57B (qw2) evaluations
# human_eval
python ../../ktransformers-utils/scoring.py \
    --task_name human_eval \
    --load_generations_path ./results/human_eval/qw2_int8_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name human_eval \
    --load_generations_path ./results/human_eval/qw2_int8_4+4.jsonl \
    --metric_output_path scores.jsonl

# mbpp
python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/qw2_int8_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/qw2_int8_4+4.jsonl \
    --metric_output_path scores.jsonl

# gsm8k
python ../../ktransformers-utils/scoring.py \
    --task_name gsm8k \
    --load_generations_path ./results/gsm8k/qw2_int8_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name gsm8k \
    --load_generations_path ./results/gsm8k/qw2_int8_4+4.jsonl \
    --metric_output_path scores.jsonl

# strategy_qa
python ../../ktransformers-utils/scoring.py \
    --task_name strategy_qa \
    --load_generations_path ./results/strategy_qa/qw2_int8_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name strategy_qa \
    --load_generations_path ./results/strategy_qa/qw2_int8_4+4.jsonl \
    --metric_output_path scores.jsonl