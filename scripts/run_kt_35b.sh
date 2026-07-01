#!/bin/bash
# 标准 kt 35B AMXINT4 benchmark（非 MESH）— 与 MESH 对照
# 用法: run_kt_35b.sh <PORT> <CUDA_LIST>
# GE=32, TP=4, 开 cudagraph
PORT=$1
CUDA=$2
source /mnt/data2/tmp/qujing_mesh/.venv/bin/activate
export TMPDIR=/mnt/data2/tmp/qujing_mesh/.tmp
export TRITON_CACHE_DIR=/mnt/data2/tmp/qujing_mesh/.triton_cache
export SGLANG_DISABLE_CUDNN_CHECK=1
export CUDA_VISIBLE_DEVICES=$CUDA
export KT_NUM_GPU_EXPERTS=32

# 禁用 MESH：删除配置文件（否则 experts.py 会回退读取 /tmp/kt_mesh_config.json 激活 MESH）
rm -f /tmp/kt_mesh_config.json
unset KT_ENABLE_MESH KT_MESH_CAP KT_MESH_TOTAL_LAYERS KT_MESH_WEIGHT_TYPE

exec python -m sglang.launch_server \
  --host 0.0.0.0 --port $PORT \
  --model /mnt/data2/tmp/qujing_mesh/Qwen3.5-35B-A3B-Unfused/ \
  --kt-weight-path /mnt/data2/models/Qwen3.5-35B-A3B-AMXINT4-NUMA2-MESH/ \
  --kt-cpuinfer 153 --kt-threadpool-count 2 --kt-num-gpu-experts 32 \
  --kt-method AMXINT4 --kt-gpu-prefill-token-threshold 4096 \
  --kt-max-deferred-experts-per-token 3 \
  --kt-enable-dynamic-expert-update --kt-numa-nodes 0 1 \
  --attention-backend flashinfer --trust-remote-code \
  --mem-fraction-static 0.90 --chunked-prefill-size 4096 \
  --max-running-requests 16 --max-total-tokens 20000 \
  --watchdog-timeout 3000 --enable-mixed-chunk \
  --tensor-parallel-size 4 --enable-p2p-check \
  --disable-shared-experts-fusion
