# Qwen2-57B (qw2) evaluations
# mbpp
python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/qw2_int8_8+0.jsonl \
    --metric_output_path scores.jsonl

python ../../ktransformers-utils/scoring.py \
    --task_name mbpp \
    --load_generations_path ./results/mbpp/qw2_int8_4+4.jsonl \
    --metric_output_path scores.jsonl
