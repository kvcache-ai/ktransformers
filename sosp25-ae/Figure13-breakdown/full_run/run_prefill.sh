# base
python ../../baseline-fiddler/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --config_id ds3 \
    --prompt_length_list [8192]

# +v
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=10000 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_8+0.yaml \
    --config_id ds3+v \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_8+0.yaml \
    --config_id ds3+m \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d
DISABLE_NUMA_AWARE=1 USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_8+0.yaml \
    --config_id ds3+m+d \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d +n
USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_8+0.yaml \
    --config_id ds3+m+d+n \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d +n +c
USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS3_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds3_bf16_8+0.yaml \
    --config_id ds3+m+d+n+c \
    --prompt_length_list [8192] \
    --use_cuda_graph True


# base
python ../../baseline-fiddler/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --config_id ds2 \
    --prompt_length_list [8192]

# +v
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=10000 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_6+0.yaml \
    --config_id ds2+v \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_6+0.yaml \
    --config_id ds2+m \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d
DISABLE_NUMA_AWARE=1 USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_6+0.yaml \
    --config_id ds2+m+d \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d +n
USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_6+0.yaml \
    --config_id ds2+m+d+n \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d +n +c
USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$DS2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/ds2_bf16_6+0.yaml \
    --config_id ds2+m+d+n+c \
    --prompt_length_list [8192] \
    --use_cuda_graph True


# base
python ../../baseline-fiddler/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --config_id qw2 \
    --prompt_length_list [8192]

# +v
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=10000 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+v \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m
DISABLE_NUMA_AWARE=1 DISABLE_DYNAMIC_SCHEDULING=1 USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+m \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d
DISABLE_NUMA_AWARE=1 USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+m+d \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d +n
USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+m+d+n \
    --prompt_length_list [8192] \
    --use_cuda_graph False

# +m +d +n +c
USE_AMX_THRESHOLD=0 python ../../ktransformers-utils/eval_prefill.py \
    --model_path "$QW2_SAFETENSORS" \
    --optimize_rule_path ../../ktransformers-utils/optimize_rules/qw2_bf16_8+0.yaml \
    --config_id qw2+m+d+n+c \
    --prompt_length_list [8192] \
    --use_cuda_graph True