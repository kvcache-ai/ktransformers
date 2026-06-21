#!/bin/bash
# MESH 397B AMXINT4 benchmark — 修复 w2_weight bug（--model 指向 bf16）
# 用法: run_mesh_397b_cap.sh <CAP> <PORT> <CUDA_LIST>
# CAP: slot 容量 (128/192/256/480), 480=full (CPU 专家总数 = 512 - 32)
# GE=32, TP=4, 开 cudagraph
CAP=$1
PORT=$2
CUDA=$3
source /mnt/data2/tmp/qujing_mesh/.venv/bin/activate
export TMPDIR=/mnt/data2/tmp/qujing_mesh/.tmp
export TRITON_CACHE_DIR=/mnt/data2/tmp/qujing_mesh/.triton_cache
export SGLANG_DISABLE_CUDNN_CHECK=1
export CUDA_VISIBLE_DEVICES=$CUDA
export KT_ENABLE_MESH=1
export KT_MESH_CAP=$CAP
export KT_NUM_GPU_EXPERTS=32
export KT_MESH_TOTAL_LAYERS=60
export KT_MESH_WEIGHT_TYPE=amxint4

python -c "
from kt_kernel.utils.mesh.config import MeshConfig
cfg = MeshConfig.from_env()
cfg.to_file('/tmp/kt_mesh_config.json')
print('MESH config written: cap=' + str(cfg.cap))
"

exec python -m sglang.launch_server \
  --host 0.0.0.0 --port $PORT \
  --model /mnt/data2/models/Qwen3.5-397B-A17B-TEXTONLY \
  --kt-weight-path /mnt/data2/models/Qwen3.5-397B-A17B-AMXINT4-NUMA2-MESH-FIXED \
  --kt-cpuinfer 153 --kt-threadpool-count 2 --kt-num-gpu-experts 32 \
  --kt-method AMXINT4 --kt-gpu-prefill-token-threshold 4096 \
  --kt-enable-dynamic-expert-update --kt-numa-nodes 0 1 \
  --attention-backend flashinfer --trust-remote-code \
  --mem-fraction-static 0.90 --chunked-prefill-size 4096 \
  --max-running-requests 16 --max-total-tokens 20000 \
  --watchdog-timeout 3000 --enable-mixed-chunk \
  --tensor-parallel-size 4 --enable-p2p-check \
  --disable-shared-experts-fusion --language-only
