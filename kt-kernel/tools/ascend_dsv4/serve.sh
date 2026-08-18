#!/usr/bin/env bash
# =============================================================================
# Launch the DeepSeek-V4-Flash single-NPU server.
#
#   bash kt-kernel/tools/ascend_dsv4/serve.sh              # background + log
#   bash kt-kernel/tools/ascend_dsv4/serve.sh --foreground  # stay attached
#
# Every parameter comes from dsv4_env.sh; override by exporting before running.
# Extra SGLang flags can be appended through DSV4_EXTRA_FLAGS.
# =============================================================================
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

FOREGROUND=0
[ "${1:-}" = "--foreground" ] && FOREGROUND=1

mkdir -p "${DSV4_LOG_DIR}"
LOG="${DSV4_LOG_DIR}/serve.log"

# Self-check: prove we are about to run the clone, not a bundled SGLang. This
# is the single most common way to spend a day debugging code that never ran.
_sglang_file="$("${DSV4_PYTHON}" -c 'import sglang; print(sglang.__file__)' | tail -1)"
echo "[serve] sglang = ${_sglang_file}"
case "${_sglang_file}" in
  "${SGLANG_REPO}"/python/sglang/__init__.py) ;;
  *) echo "[serve] FATAL: sglang resolves outside ${SGLANG_REPO}. PYTHONPATH was shadowed." >&2
     exit 1 ;;
esac

# --- runtime environment ----------------------------------------------------
export ASCEND_RT_VISIBLE_DEVICES="${DSV4_NPU_DEVICE_ID}"
# Without expandable segments HBM fragments across prefill chunks: the reported
# weight footprint grows by roughly 1.8 GB and the KV pool is sized down to match.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
ulimit -n 65536 2>/dev/null || true

# --- optional: streaming prefill -------------------------------------------
# Off by default. It makes prefill time roughly constant in prompt length but
# costs a fixed ~19 s, so it only pays above ~1500 tokens, and it needs its own
# HBM budget (see the tuning section of the tutorial).
if [ "${DSV4_PREFILL_STREAM:-0}" = "1" ]; then
  export KT_PREFILL_STREAM=1
  export KT_PREFILL_STREAM_THRESHOLD="${KT_PREFILL_STREAM_THRESHOLD:-512}"
  export KT_PREFILL_STREAM_CKPT="${KT_PREFILL_STREAM_CKPT:-${DSV4_MODEL_PATH}}"
  export KT_MXFP4_CKPT="${KT_MXFP4_CKPT:-${DSV4_NATIVE_CKPT}}"
  export KT_MXFP4_OP_DIR="${KT_MXFP4_OP_DIR:-${KTRANSFORMERS_REPO}/kt-kernel/tools/ascendc_mxfp4}"
  export KT_MXFP4_DEPOOL="${KT_MXFP4_DEPOOL:-1}"
  export KT_MXFP4_GGUF_DEDUP="${KT_MXFP4_GGUF_DEDUP:-1}"
  # The dedup path reads the GGUF template from its own variable rather than from
  # --kt-weight-path. Without it the switch is accepted and then does nothing, logging
  # only "KT_MXFP4_GGUF_DEDUP=1 but KT_GGUF_TEMPLATE is empty".
  export KT_GGUF_TEMPLATE="${KT_GGUF_TEMPLATE:-${DSV4_GGUF_TEMPLATE}}"
  export KT_DYNAMIC_RESIDENT="${KT_DYNAMIC_RESIDENT:-1}"
  export KT_SIDE_STREAM="${KT_SIDE_STREAM:-1}"
  # Streaming prefill never exercises the CPU MoE, so without this the server warmup runs
  # straight through the streaming path and leaves kt_kernel cold -- the first requests then
  # decode noticeably slower. Forcing one pass through the hybrid path warms it; one is enough.
  export KT_STREAM_WARMUP="${KT_STREAM_WARMUP:-1}"
  echo "[serve] streaming prefill ENABLED (threshold ${KT_PREFILL_STREAM_THRESHOLD} tokens)"
fi

CMD=( "${DSV4_PYTHON}" -m sglang.launch_server
  --model-path            "${DSV4_MODEL_PATH}"
  --device                npu
  --attention-backend     ascend
  --tensor-parallel-size  1
  --expert-parallel-size  1
  --moe-a2a-backend       none
  --page-size             128
  --quantization          compressed-tensors
  --disable-shared-experts-fusion
  --dtype                 bfloat16
  --trust-remote-code
  --disable-radix-cache
  --mem-fraction-static   "${DSV4_MEM_FRACTION}"
  --context-length        "${DSV4_CONTEXT_LENGTH}"
  --max-prefill-tokens    "$(( DSV4_CONTEXT_LENGTH - 1 ))"
  --chunked-prefill-size  "${DSV4_CHUNKED_PREFILL_SIZE}"
  --watchdog-timeout      18000
  --kt-method             LLAMAFILE
  --kt-num-gpu-experts    "${DSV4_NUM_GPU_EXPERTS}"
  --kt-weight-path        "${DSV4_GGUF_TEMPLATE}"
  --kt-threadpool-count   "${DSV4_THREADPOOL_COUNT}"
  --kt-cpuinfer           "${DSV4_CPUINFER}"
  --max-running-requests  1
  --host                  "${DSV4_HOST}"
  --port                  "${DSV4_PORT}"
)
# shellcheck disable=SC2206
[ -n "${DSV4_EXTRA_FLAGS:-}" ] && CMD+=( ${DSV4_EXTRA_FLAGS} )

printf '[serve] %s\n' "${CMD[*]}"

if [ "${FOREGROUND}" -eq 1 ]; then
  exec "${CMD[@]}"
fi

nohup "${CMD[@]}" > "${LOG}" 2>&1 &
echo $! > "${LOG}.pid"
echo "[serve] pid=$(cat "${LOG}.pid")  log=${LOG}"
echo "[serve] follow with:  tail -f ${LOG}"
echo "[serve] accept  with:  bash ${_here}/verify.sh"
echo "[serve] stop    with:  kill -INT \$(cat ${LOG}.pid) && sleep 5 && kill -TERM \$(cat ${LOG}.pid)"
