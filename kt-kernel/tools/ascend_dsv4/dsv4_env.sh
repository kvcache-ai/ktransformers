#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash on a single Ascend NPU — environment definition.
#
# Every other script in this directory sources this file. Source it yourself
# before running anything by hand:
#
#     source kt-kernel/tools/ascend_dsv4/dsv4_env.sh
#
# Nothing here is machine specific: every value is either auto-detected or
# overridable from the environment. To pin values for your machine, either
# export them before sourcing, or copy this file's "user settings" block into
# a small wrapper of your own. Run
#
#     bash kt-kernel/tools/ascend_dsv4/dsv4_env.sh --show
#
# to print what it resolved to without changing your shell.
# =============================================================================

# The CANN set_env.sh scripts reference unset variables and return non-zero on
# absent optional paths, so `set -u` and `set -e` have to be off while this file
# runs. Remember what the caller had and restore it at the very bottom.
_dsv4_saved_u=0; _dsv4_saved_e=0
case "$-" in *u*) _dsv4_saved_u=1; set +u ;; esac
case "$-" in *e*) _dsv4_saved_e=1; set +e ;; esac

# ---------------------------------------------------------------------------
# 1. CANN toolkit
# ---------------------------------------------------------------------------
# ASCEND_INSTALL_ROOT is the directory that holds `ascend-toolkit`, `nnal`, and
# the versioned `cann-X.Y.Z` directory. It is /usr/local/Ascend on most images
# but a user-writable install puts it under $HOME/Ascend.
if [ -z "${ASCEND_INSTALL_ROOT}" ]; then
  for _cand in "${HOME}/Ascend" /usr/local/Ascend /opt/Ascend; do
    if [ -f "${_cand}/ascend-toolkit/set_env.sh" ]; then
      ASCEND_INSTALL_ROOT="${_cand}"
      break
    fi
  done
fi
if [ -z "${ASCEND_INSTALL_ROOT}" ]; then
  echo "[dsv4_env] ERROR: no CANN toolkit found." >&2
  echo "[dsv4_env]   Looked for <root>/ascend-toolkit/set_env.sh under" >&2
  echo "[dsv4_env]   \$HOME/Ascend, /usr/local/Ascend, /opt/Ascend." >&2
  echo "[dsv4_env]   Set ASCEND_INSTALL_ROOT to your install prefix and retry." >&2
  return 1 2>/dev/null || exit 1
fi
export ASCEND_INSTALL_ROOT

# The versioned directory. `ascend-toolkit/latest` is a symlink into it; we
# resolve the real path so CANN_VERSION never has to be hardcoded.
if [ -z "${CANN_ROOT}" ]; then
  if [ -d "${ASCEND_INSTALL_ROOT}/ascend-toolkit/latest" ]; then
    CANN_ROOT="$(cd "${ASCEND_INSTALL_ROOT}/ascend-toolkit/latest" 2>/dev/null && pwd -P)"
    # .../cann-9.0.0/<arch>-linux -> .../cann-9.0.0
    case "${CANN_ROOT}" in
      */aarch64-linux|*/x86_64-linux) CANN_ROOT="$(dirname "${CANN_ROOT}")" ;;
    esac
  fi
fi
[ -z "${CANN_ROOT}" ] && CANN_ROOT="${ASCEND_INSTALL_ROOT}"
export CANN_ROOT

if [ -z "${CANN_VERSION}" ]; then
  CANN_VERSION="$(sed -n 's/^version=//p' \
      "${ASCEND_INSTALL_ROOT}/ascend-toolkit/latest/"*/ascend_toolkit_install.info \
      2>/dev/null | head -1)"
fi
export CANN_VERSION="${CANN_VERSION:-unknown}"

export CANN_VENDORS_DIR="${CANN_VENDORS_DIR:-${CANN_ROOT}/opp/vendors}"

# ---------------------------------------------------------------------------
# 2. Workspace and repositories
# ---------------------------------------------------------------------------
# DSV4_WORKSPACE holds the clones and the build artifacts. Default: the parent
# of this repository, so a plain `git clone` needs no configuration at all.
_dsv4_env_self="${BASH_SOURCE[0]:-$0}"
_dsv4_tools_dir="$(cd "$(dirname "${_dsv4_env_self}")" && pwd -P)"
export KTRANSFORMERS_REPO="${KTRANSFORMERS_REPO:-$(cd "${_dsv4_tools_dir}/../../.." && pwd -P)}"
export DSV4_WORKSPACE="${DSV4_WORKSPACE:-$(dirname "${KTRANSFORMERS_REPO}")}"
# The tutorial clones SGLang next to this repository, but a checkout that keeps
# it under third_party/ works just as well — prefer whichever exists.
if [ -z "${SGLANG_REPO}" ]; then
  if [ -d "${KTRANSFORMERS_REPO}/third_party/sglang/python/sglang" ]; then
    SGLANG_REPO="${KTRANSFORMERS_REPO}/third_party/sglang"
  else
    SGLANG_REPO="${DSV4_WORKSPACE}/sglang"
  fi
fi
export SGLANG_REPO

# Third-party CANN operator sources. Cloned by build_cann_ops.sh if missing.
export CANN_RECIPES_REPO="${CANN_RECIPES_REPO:-${DSV4_WORKSPACE}/cann-recipes-infer}"
export OPS_TRANSFORMER_REPO="${OPS_TRANSFORMER_REPO:-${DSV4_WORKSPACE}/ops-transformer}"
export CANN_RECIPES_URL="${CANN_RECIPES_URL:-https://gitcode.com/cann/cann-recipes-infer.git}"
export OPS_TRANSFORMER_URL="${OPS_TRANSFORMER_URL:-https://gitcode.com/cann/ops-transformer.git}"
# Both repositories track a moving `master` with no tags. These are the commits
# this tutorial was validated against; bump them only together with a re-run of
# the acceptance checks in the tutorial.
export CANN_RECIPES_COMMIT="${CANN_RECIPES_COMMIT:-1c8e6bcc2333d95b3db47d873210f921113d6d11}"
export OPS_TRANSFORMER_COMMIT="${OPS_TRANSFORMER_COMMIT:-8edcd591e83e536e9ee98a9ce0de3af02ea4f3ea}"

export DSV4_ARTIFACT_DIR="${DSV4_ARTIFACT_DIR:-${DSV4_WORKSPACE}/dsv4-artifacts}"
export DSV4_LOG_DIR="${DSV4_LOG_DIR:-${DSV4_WORKSPACE}/dsv4-logs}"

# ---------------------------------------------------------------------------
# 3. Model weights
# ---------------------------------------------------------------------------
# DSV4_MODEL_PATH  — W8A8 compressed-tensors checkpoint, served on the NPU.
# DSV4_NATIVE_CKPT — the official DeepSeek-V4-Flash checkpoint. Only needed to
#                    produce the GGUF set; not read at serving time.
# DSV4_GGUF_DIR    — where the 43 per-layer MXFP4 GGUF files live (138 GiB).
export DSV4_MODEL_ROOT="${DSV4_MODEL_ROOT:-${DSV4_WORKSPACE}/models}"
export DSV4_MODEL_PATH="${DSV4_MODEL_PATH:-${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash-W8A8}"
export DSV4_NATIVE_CKPT="${DSV4_NATIVE_CKPT:-${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash}"
export DSV4_GGUF_DIR="${DSV4_GGUF_DIR:-${DSV4_MODEL_ROOT}/cache}"
# `{layer_idx}` is a literal placeholder substituted per MoE layer by SGLang.
# NOTE the plain assignment below. Writing this as
#     ${DSV4_GGUF_TEMPLATE:-${DSV4_GGUF_DIR}/dsv4_layer{layer_idx}_mxfp4.gguf}
# does not work: inside a `${...:-...}` default, bash treats the first `}` of
# `{layer_idx}` as the end of the expansion and silently yields
# `dsv4_layer{layer_idx_mxfp4.gguf}`. Same trap applies when you export it by
# hand — always single-quote the value there.
if [ -z "${DSV4_GGUF_TEMPLATE}" ]; then
  DSV4_GGUF_TEMPLATE="${DSV4_GGUF_DIR}/dsv4_layer{layer_idx}_mxfp4.gguf"
fi
export DSV4_GGUF_TEMPLATE

# ---------------------------------------------------------------------------
# 4. Hardware-derived serving parameters
# ---------------------------------------------------------------------------
# One CPU-MoE sub-pool per NUMA node. Exceeding the node count fails at startup
# with `NUMA node N not found`.
if [ -z "${DSV4_THREADPOOL_COUNT}" ]; then
  DSV4_THREADPOOL_COUNT="$(find /sys/devices/system/node -maxdepth 1 -type d \
      -name 'node[0-9]*' 2>/dev/null | wc -l | tr -d ' ')"
  [ "${DSV4_THREADPOOL_COUNT}" -ge 1 ] 2>/dev/null || DSV4_THREADPOOL_COUNT=1
fi
export DSV4_THREADPOOL_COUNT
# Total CPU-MoE threads, ~16 per NUMA node, capped so spin threads, the ACL
# host callback and the OS keep some headroom.
if [ -z "${DSV4_CPUINFER}" ]; then
  _dsv4_nproc="$(nproc 2>/dev/null || echo 16)"
  DSV4_CPUINFER=$((DSV4_THREADPOOL_COUNT * 16))
  _dsv4_cap=$((_dsv4_nproc * 3 / 4))
  [ "${DSV4_CPUINFER}" -gt "${_dsv4_cap}" ] && DSV4_CPUINFER="${_dsv4_cap}"
  [ "${DSV4_CPUINFER}" -lt 1 ] && DSV4_CPUINFER=1
fi
export DSV4_CPUINFER

export DSV4_NPU_DEVICE_ID="${DSV4_NPU_DEVICE_ID:-0}"
export DSV4_PORT="${DSV4_PORT:-18080}"
export DSV4_HOST="${DSV4_HOST:-0.0.0.0}"
export DSV4_NUM_GPU_EXPERTS="${DSV4_NUM_GPU_EXPERTS:-32}"
export DSV4_MEM_FRACTION="${DSV4_MEM_FRACTION:-0.81}"
export DSV4_CONTEXT_LENGTH="${DSV4_CONTEXT_LENGTH:-65536}"
export DSV4_CHUNKED_PREFILL_SIZE="${DSV4_CHUNKED_PREFILL_SIZE:-32768}"

# ---------------------------------------------------------------------------
# 5. Build settings
# ---------------------------------------------------------------------------
export DSV4_PYTHON="${DSV4_PYTHON:-$(command -v python3.11 || command -v python3)}"
export DSV4_JOBS="${DSV4_JOBS:-$(( $(nproc 2>/dev/null || echo 8) < 16 ? $(nproc 2>/dev/null || echo 8) : 16 ))}"
# SoC name for the CANN operator builds. ascend910b = Atlas A2 (910B series),
# ascend910_93 = Atlas A3 (910C / 910_93 series). Getting this wrong produces
# operator binaries that load but fail at the first kernel launch, so the value
# is derived from the chip rather than guessed, and left empty when unknown.
dsv4_soc_from_chip_name() {
  case "$1" in
    *910_93*|*910C*|*910c*) echo "ascend910_93" ;;
    *910B*|*910b*)          echo "ascend910b" ;;
    *)                      echo "" ;;
  esac
}
dsv4_detect_soc() {
  local name=""
  # npu-smi is the cheap path, but it is missing or broken in some containers.
  if command -v npu-smi >/dev/null 2>&1; then
    name="$(npu-smi info 2>/dev/null | awk '/^\| *[0-9]+ +[0-9A-Za-z_]+ +\|/ {print $3; exit}')"
  fi
  # Fall back to the runtime, which works wherever torch_npu does.
  if [ -z "$(dsv4_soc_from_chip_name "${name}")" ] && [ -n "${DSV4_PYTHON}" ]; then
    name="$("${DSV4_PYTHON}" -c \
      'import torch, torch_npu; print(torch.npu.get_device_name(0))' 2>/dev/null | tail -1)"
  fi
  dsv4_soc_from_chip_name "${name}"
}
export DSV4_SOC="${DSV4_SOC:-}"

# ---------------------------------------------------------------------------
# 6. Source the CANN environment — order matters
# ---------------------------------------------------------------------------
# These scripts *prepend* to PYTHONPATH, so PYTHONPATH must be exported AFTER
# all of them (see the end of this file).
_dsv4_source_if_readable() { [ -r "$1" ] && . "$1"; return 0; }

_dsv4_source_if_readable "${ASCEND_INSTALL_ROOT}/ascend-toolkit/set_env.sh"
_dsv4_source_if_readable "${ASCEND_INSTALL_ROOT}/nnal/atb/set_env.sh"
_dsv4_source_if_readable "${CANN_ROOT}/share/info/ascendnpu-ir/bin/set_env.sh"

# Custom operator vendors. Missing here is not fatal at build time — it is
# fatal at serve time, and preflight.sh checks for it explicitly.
for _vendor in customize custom_transformer; do
  _dsv4_source_if_readable "${CANN_VENDORS_DIR}/${_vendor}/bin/set_env.bash"
  if [ -d "${CANN_VENDORS_DIR}/${_vendor}" ]; then
    export ASCEND_CUSTOM_OPP_PATH="${CANN_VENDORS_DIR}/${_vendor}:${ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="${CANN_VENDORS_DIR}/${_vendor}/op_api/lib:${CANN_VENDORS_DIR}/${_vendor}/op_proto/lib/linux/$(uname -m):${LD_LIBRARY_PATH}"
  fi
done

# ---------------------------------------------------------------------------
# 7. PYTHONPATH — must come last
# ---------------------------------------------------------------------------
# First match wins, and the CANN scripts above prepend their own entries. If
# PYTHONPATH were set before them, the image's bundled SGLang would shadow the
# clone: the server starts, answers normally, and runs none of your code.
#
# PYTHONNOUSERSITE keeps a stray ~/.local from shadowing the image's packages —
# but only set it when the interpreter's own site-packages is writable. On a
# host where you are not root, pip installs into ~/.local by design, and
# disabling user-site there hides every dependency you just installed.
if [ -z "${DSV4_NO_USER_SITE}" ]; then
  if "${DSV4_PYTHON}" -c 'import os,site,sys; sys.exit(0 if os.access(site.getsitepackages()[0], os.W_OK) else 1)' 2>/dev/null; then
    DSV4_NO_USER_SITE=1
  else
    DSV4_NO_USER_SITE=0
  fi
fi
if [ "${DSV4_NO_USER_SITE}" = "1" ]; then
  export PYTHONNOUSERSITE=1
else
  unset PYTHONNOUSERSITE
fi
export PYTHONPATH="${SGLANG_REPO}/python:${KTRANSFORMERS_REPO}/kt-kernel/python:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# 8. aarch64 static-TLS workaround
# ---------------------------------------------------------------------------
# On aarch64 the dynamic loader reserves a small static TLS surplus. libgomp
# needs static TLS, and by the time SGLang's import chain dlopen()s something
# that pulls it in, that surplus is gone:
#     OSError: libgomp.so.1: cannot allocate memory in static TLS block
# Preloading libgomp puts it in the initial link set, where the allocation
# always succeeds. Set DSV4_PRELOAD_LIBGOMP=0 to opt out.
if [ "${DSV4_PRELOAD_LIBGOMP:-1}" = "1" ] && [ "$(uname -m)" = "aarch64" ]; then
  case ":${LD_PRELOAD}:" in
    *libgomp.so.1*) ;;
    *)
      _dsv4_gomp="$(ldconfig -p 2>/dev/null | awk '/libgomp\.so\.1 /{print $NF; exit}')"
      [ -z "${_dsv4_gomp}" ] && [ -e /lib/aarch64-linux-gnu/libgomp.so.1 ] \
        && _dsv4_gomp=/lib/aarch64-linux-gnu/libgomp.so.1
      if [ -n "${_dsv4_gomp}" ]; then
        export LD_PRELOAD="${_dsv4_gomp}${LD_PRELOAD:+:${LD_PRELOAD}}"
      fi
      ;;
  esac
fi

# An HTTP proxy makes the startup warmup POST to the server's own port return
# 502 and aborts initialization.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

dsv4_show_env() {
  cat <<EOF
CANN
  ASCEND_INSTALL_ROOT     ${ASCEND_INSTALL_ROOT}
  CANN_ROOT               ${CANN_ROOT}
  CANN_VERSION            ${CANN_VERSION}
  CANN_VENDORS_DIR        ${CANN_VENDORS_DIR}
  vendors present         $(for v in customize custom_transformer; do
                              [ -d "${CANN_VENDORS_DIR}/$v" ] && printf '%s ' "$v"; done; echo)
Repositories
  KTRANSFORMERS_REPO      ${KTRANSFORMERS_REPO}
  SGLANG_REPO             ${SGLANG_REPO}
  CANN_RECIPES_REPO       ${CANN_RECIPES_REPO} @ ${CANN_RECIPES_COMMIT}
  OPS_TRANSFORMER_REPO    ${OPS_TRANSFORMER_REPO} @ ${OPS_TRANSFORMER_COMMIT}
  DSV4_ARTIFACT_DIR       ${DSV4_ARTIFACT_DIR}
  DSV4_LOG_DIR            ${DSV4_LOG_DIR}
Weights
  DSV4_MODEL_PATH         ${DSV4_MODEL_PATH}
  DSV4_NATIVE_CKPT        ${DSV4_NATIVE_CKPT}
  DSV4_GGUF_DIR           ${DSV4_GGUF_DIR}
  DSV4_GGUF_TEMPLATE      ${DSV4_GGUF_TEMPLATE}
Hardware / serving
  DSV4_SOC                ${DSV4_SOC:-<unset — detected on demand: $(dsv4_detect_soc || true)>}
  DSV4_NPU_DEVICE_ID      ${DSV4_NPU_DEVICE_ID}
  DSV4_THREADPOOL_COUNT   ${DSV4_THREADPOOL_COUNT}   (NUMA nodes)
  DSV4_CPUINFER           ${DSV4_CPUINFER}
  DSV4_NUM_GPU_EXPERTS    ${DSV4_NUM_GPU_EXPERTS}
  DSV4_MEM_FRACTION       ${DSV4_MEM_FRACTION}
  DSV4_CONTEXT_LENGTH     ${DSV4_CONTEXT_LENGTH}
  DSV4_CHUNKED_PREFILL_SIZE ${DSV4_CHUNKED_PREFILL_SIZE}
  DSV4_HOST:DSV4_PORT     ${DSV4_HOST}:${DSV4_PORT}
Build
  DSV4_PYTHON             ${DSV4_PYTHON}
  DSV4_JOBS               ${DSV4_JOBS}
Python
  user site-packages      $([ "${DSV4_NO_USER_SITE}" = "1" ] && echo "disabled (system site-packages is writable)" || echo "ENABLED (system site-packages is read-only, pip installs into ~/.local)")
  LD_PRELOAD              ${LD_PRELOAD:-<none>}
  PYTHONPATH              ${PYTHONPATH}
EOF
}

case "${1}" in
  --show|show) dsv4_show_env ;;
esac

[ "${_dsv4_saved_u}" = "1" ] && set -u
[ "${_dsv4_saved_e}" = "1" ] && set -e
true
