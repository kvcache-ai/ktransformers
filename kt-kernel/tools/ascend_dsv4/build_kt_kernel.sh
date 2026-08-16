#!/usr/bin/env bash
# =============================================================================
# Build kt-kernel with the Ascend NPU backend and produce an installable wheel.
#
# Usage:
#   bash kt-kernel/tools/ascend_dsv4/build_kt_kernel.sh          # build + wheel
#   bash kt-kernel/tools/ascend_dsv4/build_kt_kernel.sh inplace  # build_ext only
#
# Takes 10-30 minutes on a first build.
# =============================================================================
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

log() { printf '\n[build_kt_kernel] %s\n' "$*"; }
die() { printf '[build_kt_kernel] FATAL: %s\n' "$*" >&2; exit 1; }

MODE="${1:-wheel}"

# --- preconditions ----------------------------------------------------------
[ -f "${KTRANSFORMERS_REPO}/kt-kernel/CMakeLists.txt" ] \
  || die "KTRANSFORMERS_REPO does not look like a ktransformers checkout: ${KTRANSFORMERS_REPO}"

for sm in third_party/llama.cpp third_party/pybind11; do
  if [ -z "$(ls -A "${KTRANSFORMERS_REPO}/${sm}" 2>/dev/null)" ]; then
    die "${sm} is empty. Run:
    git -C ${KTRANSFORMERS_REPO} submodule update --init --progress ${sm}"
  fi
done

pkg-config --exists hwloc 2>/dev/null \
  || die "hwloc development files not found. kt-kernel marks hwloc REQUIRED.
    Debian/Ubuntu: apt-get install -y libhwloc-dev libhwloc15 libnuma-dev pkg-config
    openEuler/CentOS: dnf install -y hwloc-devel numactl-devel pkgconfig"

# --- compiler ---------------------------------------------------------------
# kt-kernel is C++20 and uses <barrier>, which libstdc++ only ships from GCC 11.
# Pick the newest GCC available unless the caller pinned one.
if [ -z "${CC:-}" ]; then
  for cand in gcc-14 gcc-13 gcc-12 gcc-11; do
    if command -v "${cand}" >/dev/null 2>&1; then
      CC="$(command -v "${cand}")"
      CXX="$(command -v "${cand/gcc/g++}")"
      break
    fi
  done
fi
export CC="${CC:-$(command -v gcc)}" CXX="${CXX:-$(command -v g++)}"

# Whether that choice is honoured depends on the checkout: older revisions of
# kt-kernel/CMakeLists.txt force CMAKE_C_COMPILER to /usr/bin/gcc whenever that
# path exists (to avoid picking up a conda wrapper from PATH), which silently
# overrides CC. Detect which behaviour this tree has instead of assuming.
if grep -q 'DEFINED ENV{CC}' CMakeLists.txt 2>/dev/null; then
  _effective_cc="${CC}"
elif [ -x /usr/bin/gcc ]; then
  _effective_cc=/usr/bin/gcc
  [ "${CC}" != "/usr/bin/gcc" ] && \
    log "NOTE: CC=${CC} will be ignored — this CMakeLists.txt force-selects /usr/bin/gcc."
else
  _effective_cc="${CC}"
fi
_gcc_major="$("${_effective_cc}" -dumpversion 2>/dev/null | cut -d. -f1)"
log "compiler CMake will use: ${_effective_cc} (major ${_gcc_major:-?})"

if [ "${_gcc_major:-0}" -lt 11 ] 2>/dev/null; then
  die "GCC ${_gcc_major} is too old: kt-kernel is C++20 and needs <barrier>, which
    libstdc++ ships from GCC 11 onward. The build fails with
      fatal error: barrier: No such file or directory
    Install a newer GCC (Debian/Ubuntu: apt-get install -y gcc-13 g++-13).
    If /usr/bin/gcc is still the old one and this CMakeLists.txt does not honour
    CC/CXX, either update /usr/bin/gcc or use a checkout that does."
fi

# ARM extension flags.
#
# The validated CPU MoE target for this deployment is the NEON baseline
# `armv8.2-a+fp16+dotprod`, so all three optional extensions default to OFF:
#
#   SVE   — the SVE branch of the llamafile sgemm has no MXFP4 tile, so decode
#           dies with "llamafile not supported" at the first token.
#   BF16  } setup.py turns these on whenever /proc/cpuinfo advertises them, but
#   I8MM  } GCC < 10 cannot encode the modifiers and the build fails with
#          `invalid feature modifier 'bf16'`.
#
# Set DSV4_ARM_NATIVE=1 to let setup.py auto-detect instead — only worthwhile
# with GCC >= 10, and re-validate the numerical probe afterwards.
if [ "${DSV4_ARM_NATIVE:-0}" = "1" ] && [ "${_gcc_major:-0}" -ge 10 ] 2>/dev/null; then
  log "DSV4_ARM_NATIVE=1: letting setup.py auto-detect the ARM extensions"
else
  if [ "${DSV4_ARM_NATIVE:-0}" = "1" ]; then
    log "DSV4_ARM_NATIVE=1 ignored: GCC ${_gcc_major} cannot encode +bf16/+i8mm"
  fi
  export CPUINFER_ARM_SVE="${CPUINFER_ARM_SVE:-OFF}"
  export CPUINFER_ARM_BF16="${CPUINFER_ARM_BF16:-OFF}"
  export CPUINFER_ARM_I8MM="${CPUINFER_ARM_I8MM:-OFF}"
  log "ARM extensions: SVE=${CPUINFER_ARM_SVE} BF16=${CPUINFER_ARM_BF16} I8MM=${CPUINFER_ARM_I8MM}"
fi
export CPUINFER_USE_ASCEND_NPU=1
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-${ASCEND_INSTALL_ROOT}/ascend-toolkit/latest}"
export CPUINFER_PARALLEL="${CPUINFER_PARALLEL:-${DSV4_JOBS}}"

cd "${KTRANSFORMERS_REPO}/kt-kernel"

log "configuring with CPUINFER_USE_ASCEND_NPU=1 ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME}"

if [ "${MODE}" = "inplace" ]; then
  "${DSV4_PYTHON}" setup.py build_ext --inplace
  ls python/kt_kernel_ext*.so >/dev/null \
    || die "no kt_kernel_ext*.so produced"
  log "built python/$(basename "$(ls python/kt_kernel_ext*.so | head -1)")"
  exit 0
fi

mkdir -p "${DSV4_ARTIFACT_DIR}/wheels"
# --no-deps / --no-build-isolation: never let pip re-resolve the image's torch
# stack. Replacing torch on a CANN image unbinds torch_npu and the NPU stack
# has to be rebuilt from scratch.
CPUINFER_FORCE_REBUILD=0 "${DSV4_PYTHON}" -m pip wheel \
    --no-deps --no-build-isolation \
    --wheel-dir "${DSV4_ARTIFACT_DIR}/wheels" .

log "wheel -> $(ls -1 "${DSV4_ARTIFACT_DIR}"/wheels/kt_kernel-*.whl | tail -1)"
log "install it with:  ${DSV4_PYTHON} -m pip install --no-deps ${DSV4_ARTIFACT_DIR}/wheels/kt_kernel-*.whl"
