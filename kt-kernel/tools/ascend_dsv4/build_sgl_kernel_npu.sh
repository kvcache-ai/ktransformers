#!/usr/bin/env bash
# =============================================================================
# Build and install sgl-kernel-npu (sgl_kernel_npu, deep_ep, attentions,
# torch_memory_saver).
#
#   bash kt-kernel/tools/ascend_dsv4/build_sgl_kernel_npu.sh
#
# Skip this if `python -c "import sgl_kernel_npu"` already works — the Ascend
# SGLang container images ship it. A native CANN install almost certainly does
# not, and the import is NOT optional: sglang/srt/mem_cache/pool_host/mha.py
# imports it unconditionally, so the server dies at startup with
#
#   ModuleNotFoundError: No module named 'sgl_kernel_npu'
#
# Budget 20-40 minutes.
# =============================================================================
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

log() { printf '\n[build_sgl_kernel_npu] %s\n' "$*"; }
die() { printf '[build_sgl_kernel_npu] FATAL: %s\n' "$*" >&2; exit 1; }

REPO="${SGL_KERNEL_NPU_REPO:-${DSV4_WORKSPACE}/sgl-kernel-npu}"
URL="${SGL_KERNEL_NPU_URL:-https://github.com/sgl-project/sgl-kernel-npu.git}"
TAG="${SGL_KERNEL_NPU_TAG:-2026.6.2}"

umask 0022

if [ ! -d "${REPO}/.git" ]; then
  log "cloning ${URL} -> ${REPO}"
  git clone --progress "${URL}" "${REPO}"
fi
git -C "${REPO}" fetch --quiet --tags origin || true
git -C "${REPO}" checkout --quiet "${TAG}" || die "cannot check out ${TAG} in ${REPO}"
git -C "${REPO}" submodule update --init --recursive --progress

cd "${REPO}"
chmod -R go-w . 2>/dev/null || true

# Known issue in this tag: PTAExtensionOPS does not link libdl, so the build
# fails with `undefined reference to dlopen/dlsym`. build.sh runs under `set -e`,
# so the failure just leaves output/ empty with no obvious cause.
_cm="csrc/attentions/csrc/CMakeLists.txt"
if [ -f "${_cm}" ] && ! grep -q 'CMAKE_DL_LIBS' "${_cm}"; then
  log "patching ${_cm} to link \${CMAKE_DL_LIBS}"
  sed -i 's/\(target_link_libraries(PTAExtensionOPS[^\n]*\)/\1 ${CMAKE_DL_LIBS}/' "${_cm}"
  grep -q 'CMAKE_DL_LIBS' "${_cm}" || log "WARNING: automatic -ldl patch did not apply; add it by hand"
fi

# The deep_ep vendor tree is installed read-only, so a second build dies in
# `rm uninstall.sh` with Permission denied.
chmod -R u+w python/deep_ep/deep_ep/vendors 2>/dev/null || true
rm -rf python/deep_ep/deep_ep/vendors/hwcomputing 2>/dev/null || true

# NOTE: do not delete csrc/attentions/build/ — it is a tracked source directory,
# not build output. Only csrc/build_out/ and output/ are generated.

log "building (SoC: ${DSV4_SOC:-$(dsv4_detect_soc)})"
if [ -n "${SGL_KERNEL_NPU_SOC:-}" ]; then
  SOC_VERSION="${SGL_KERNEL_NPU_SOC}" bash build.sh
else
  bash build.sh
fi

_whls=()
for pkg in sgl_kernel_npu deep_ep attentions torch_memory_saver; do
  for w in output/${pkg}*.whl; do
    [ -e "${w}" ] && _whls+=("${w}")
  done
done
[ "${#_whls[@]}" -gt 0 ] || die "no wheels under output/ — see the build log above"

log "installing: ${_whls[*]}"
"${DSV4_PYTHON}" -m pip install --no-deps "${_whls[@]}"

"${DSV4_PYTHON}" -c 'import sgl_kernel_npu; print("sgl_kernel_npu OK")'
log "done."
