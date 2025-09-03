#!/bin/bash
#set -ex

# export area
export ASDOPS_LOG_TO_FILE=0
export ASDOPS_LOG_TO_STDOUT=0
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_FILE=0
export ATB_LOG_TO_STDOUT=0
export ATB_LOG_LEVEL=ERROR
export USE_MERGE=0
#export PROF_DECODE=1
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export INF_NAN_MODE_FORCE_DISABLE=1
export CAPTURE_PLUGIN_PATH=/home/x30058903/pack/npu_graph
export BLAS_NUM_THREADS=1
# export TASK_QUEUE_ENABLE=0
# source area
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# global vars
LOG_DIR="../logs"
LOG_NAME="test_cpuinfer49"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/${LOG_NAME}_${TIMESTAMP}.log"

# torchrun \
#   --master-port 41532 \
#   --nproc_per_node 1 \
#   -m ktransformers.server.main \
#   --cpu_infer 65 \
#   --model_path /home/mount/DeepSeek-R1-q4km-w8a8 \
#   --gguf_path /home/mount/DeepSeek-R1-q4km-w8a8 \
#   --max_new_tokens 200 \
#   --use_cuda_graph \
#   --optimize_config_path ./ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-800IA2-npu.yaml \
# 2>&1 | tee "${LOG_FILE}"

LOG_DIR="../logs"
WS=1
CPUINFER=65
LOG_NAME="combime_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${LOG_NAME}_${TIMESTAMP}.log"
echo ${LOG_FILE}
torchrun --nproc_per_node $WS \
         --master_port 6685 -m ktransformers.server.main \
         --cpu_infer $CPUINFER \
         --batch_size 1 \
         --chunk_size 16384 \
         --model_path /home/mount/DeepSeek-R1-q4km-w8a8 \
         --gguf_path /home/mount/DeepSeek-R1-q4km-w8a8 \
         --optimize_config_path ./ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-800IA2-npu.yaml \
         --port 10014 \
         --force_think \
         --use_cuda_graph \
         --max_new_tokens  2048 >&1 | tee "$LOG_FILE"
  
 # --gguf_path /mnt/DeepSeek-R1-BF16/ \  

