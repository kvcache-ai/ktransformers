#!/usr/bin/env bash
# =============================================================================
# Build the per-layer MXFP4 GGUF set the CPU-offloaded experts are served from.
#
#   bash kt-kernel/tools/ascend_dsv4/convert_mxfp4_gguf.sh            # convert + verify
#   bash kt-kernel/tools/ascend_dsv4/convert_mxfp4_gguf.sh verify     # verify only
#
# Reads  : ${DSV4_NATIVE_CKPT}   the official DeepSeek-V4-Flash checkpoint
# Writes : ${DSV4_GGUF_DIR}/dsv4_layer{0..N-1}_mxfp4.gguf   (~3.19 GiB each)
#
# This is a lossless bit repack, not a re-quantization: the checkpoint already
# stores E2M1 codes with a ue8m0 per-32 scale, and only the nibble order within
# each 32-element block differs from GGUF's half-block interleave.
#
# Budget roughly 138 GiB of disk and a few hours on a 40-core host.
# =============================================================================
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

TOOLS="${KTRANSFORMERS_REPO}/kt-kernel/tools/mxfp4_gguf"
log() { printf '\n[convert_mxfp4_gguf] %s\n' "$*"; }
die() { printf '[convert_mxfp4_gguf] FATAL: %s\n' "$*" >&2; exit 1; }

[ -f "${DSV4_NATIVE_CKPT}/config.json" ] \
  || die "DSV4_NATIVE_CKPT=${DSV4_NATIVE_CKPT} has no config.json.
    Download the official checkpoint first, e.g.
      huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ${DSV4_NATIVE_CKPT}"

N_LAYERS="$("${DSV4_PYTHON}" -c \
  "import json;print(json.load(open('${DSV4_NATIVE_CKPT}/config.json'))['num_hidden_layers'])")"
LAST=$(( N_LAYERS - 1 ))
mkdir -p "${DSV4_GGUF_DIR}"

# The reader needs GGML_TYPE_MXFP4, which kt-kernel's CMake step patches into
# third_party/llama.cpp's gguf-py. The tools find it themselves; KT_GGUF_PY is
# only needed if your llama.cpp checkout lives somewhere else.
export KT_GGUF_PY="${KT_GGUF_PY:-${KTRANSFORMERS_REPO}/third_party/llama.cpp/gguf-py}"
[ -d "${KT_GGUF_PY}" ] || die "no gguf-py at ${KT_GGUF_PY}.
    git -C ${KTRANSFORMERS_REPO} submodule update --init --progress third_party/llama.cpp"
"${DSV4_PYTHON}" -c "import sys;sys.path.insert(0,'${KT_GGUF_PY}');import gguf;assert int(gguf.GGMLQuantizationType.MXFP4)==39" 2>/dev/null \
  || die "gguf-py at ${KT_GGUF_PY} does not know GGML_TYPE_MXFP4.
    The patch series is applied by kt-kernel's CMake configure step; build
    kt-kernel once (build_kt_kernel.sh) and re-run this."

if [ "${1:-convert}" != "verify" ]; then
  log "converting layers 0..${LAST} from ${DSV4_NATIVE_CKPT}"
  log "output ${DSV4_GGUF_DIR} (needs ~$(( N_LAYERS * 32 / 10 )) GiB free)"
  # Layers are independent; each worker owns one output file. Never point two
  # workers at the same layer — a partially written file is not detected by
  # --skip-existing, which only skips files larger than 1 GiB.
  "${DSV4_PYTHON}" "${TOOLS}/batch_convert_mxfp4_layers_mp.py" \
      --input       "${DSV4_NATIVE_CKPT}" \
      --output-dir  "${DSV4_GGUF_DIR}" \
      --layer-start 0 --layer-end "${LAST}" \
      --jobs        "${DSV4_JOBS}" \
      --skip-existing
fi

log "verifying (L1 count+size, L2 sha256 self-manifest, L3 bit-exact sample)"
"${DSV4_PYTHON}" "${TOOLS}/verify_mxfp4_gguf_set.py" \
    --dir "${DSV4_GGUF_DIR}" \
    --expect-layers "${N_LAYERS}" \
    --deep 3 --model-dir "${DSV4_NATIVE_CKPT}"

log "done. ${DSV4_GGUF_DIR} holds $(ls -1 "${DSV4_GGUF_DIR}"/dsv4_layer*_mxfp4.gguf | wc -l) files."
log "Point the server at them with:"
log "  export DSV4_GGUF_TEMPLATE='${DSV4_GGUF_DIR}/dsv4_layer{layer_idx}_mxfp4.gguf'"
log "  (single quotes are required — in double quotes bash eats the first '}')"
