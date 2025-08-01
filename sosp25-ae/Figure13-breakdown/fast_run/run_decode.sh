# base
python ../../baseline-fiddler/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --config_id qw2

# +v
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=10000 python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+v \
    --use_cuda_graph False

# +m
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+m \
    --use_cuda_graph False

# +v +d
DISABLE_NUMA_AWARE=1 USE_AMX_THRESHOLD=10000 python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+v+d \
    --use_cuda_graph False

# +v +d +n
USE_AMX_THRESHOLD=10000 python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+v+d+n \
    --use_cuda_graph False

# +v +d +n +c
USE_AMX_THRESHOLD=10000 python ../../ktransformers-utils/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+v+d+n+c \
    --use_cuda_graph True