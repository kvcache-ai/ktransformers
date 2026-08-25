#!/usr/bin/env bash
# Build and install everything the server needs. See
# doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md for the walkthrough.
#
#   setup.sh probe        report what this image already provides
#   setup.sh all          deps -> kt-kernel -> sgl-kernel -> cann-ops -> gguf -> check
#   setup.sh <step>       run one step
#
#   probe       report what the image ships and which steps will be skipped
#   deps        SGLang NPU runtime dependencies
#   kt-kernel   build kt-kernel, produce a wheel
#   sgl-kernel  build sgl_kernel_npu, deep_ep, attentions, torch_memory_saver
#   cann-ops    build the customize / custom_ops / custom_transformer packages
#   gguf        convert the checkpoint to the per-layer MXFP4 GGUF set
#   check       verify the environment; exit 0 means safe to launch
#
# sgl-kernel and cann-ops return early when the image already provides them.
# Set DSV4_FORCE_SGL_KERNEL=1 / DSV4_FORCE_CANN_OPS=1 to build anyway.
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

_STEP="setup"
log() { printf '
[%s] %s
' "${_STEP}" "$*"; }
die() { printf '[%s] FATAL: %s
' "${_STEP}" "$*" >&2; exit 1; }


step_deps() {
  DRY=0
  [ "${1:-}" = "--dry-run" ] && DRY=1
PYPROJECT="${SGLANG_REPO}/python/pyproject_npu.toml"
if [ ! -f "${PYPROJECT}" ]; then
  _found="$(find "${DSV4_WORKSPACE}" "${KTRANSFORMERS_REPO}" -maxdepth 4 \
      -name pyproject_npu.toml -path '*/python/*' 2>/dev/null | head -5)"
  die "not found: ${PYPROJECT}
    SGLANG_REPO=${SGLANG_REPO} does not look like an SGLang checkout.
$( [ -n "${_found}" ] \
     && printf '    Found one under:\n%s\n    Set it explicitly, e.g.\n      export SGLANG_REPO=%s' \
          "$(printf '%s\n' "${_found}" | sed 's|^|      |')" \
          "$(dirname "$(dirname "$(printf '%s\n' "${_found}" | head -1)")")" \
     || printf '    No pyproject_npu.toml found under %s either — clone SGLang first.' "${DSV4_WORKSPACE}" )"
fi

WORK="${DSV4_ARTIFACT_DIR}/python-deps"
mkdir -p "${WORK}"
REQS="${WORK}/sglang-npu-requirements.txt"
LOCK="${WORK}/torch-constraints.txt"

log "reading the NPU dependency list from ${PYPROJECT}"
"${DSV4_PYTHON}" - "${PYPROJECT}" "${REQS}" <<'PY'
import sys, tomllib
src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    data = tomllib.load(f)
deps = data["project"]["dependencies"]
with open(dst, "w") as f:
    f.write("# Generated from pyproject_npu.toml — do not edit by hand.\n")
    for d in deps:
        f.write(d + "\n")
print(f"{len(deps)} dependencies -> {dst}")
PY

log "pinning the torch family that is already installed"
"${DSV4_PYTHON}" - "${LOCK}" <<'PY'
import importlib.metadata as md, sys
dst = sys.argv[1]
lines = ["# Generated from the installed environment. Any dependency that wants a\n",
         "# different torch will now fail loudly instead of silently upgrading it.\n"]
found, seen = [], set()
for name in ("torch", "torch_npu", "torch-npu"):
    try:
        v = md.version(name)
    except md.PackageNotFoundError:
        continue
    canon = name.replace("_", "-").lower()  # pip treats torch_npu and torch-npu as one
    if canon in seen:
        continue
    seen.add(canon)
    lines.append(f"{canon}=={v}\n")
    found.append(f"{canon}=={v}")
if not any(f.startswith("torch==") for f in found):
    raise SystemExit("torch is not installed — install torch and torch_npu for your CANN "
                     "version before running this script")
open(dst, "w").writelines(lines)
print("constraints: " + " ".join(found))
PY

if [ "${DRY}" -eq 1 ]; then
  log "dry run; would install:"
  sed -n '2,$p' "${REQS}" | sed 's/^/    /'
  log "with constraints:"
  sed -n '3,$p' "${LOCK}" | sed 's/^/    /'
  return 0
fi

log "installing (this does not touch torch)"
"${DSV4_PYTHON}" -m pip install -r "${REQS}" -c "${LOCK}"

log "done. Verify with:  bash ${_here}/setup.sh check"
}


step_kt_kernel() {
  MODE="${1:-wheel}"
MODE="${1:-wheel}"

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
  return 0
fi

mkdir -p "${DSV4_ARTIFACT_DIR}/wheels"
CPUINFER_FORCE_REBUILD=0 "${DSV4_PYTHON}" -m pip wheel \
    --no-deps --no-build-isolation \
    --wheel-dir "${DSV4_ARTIFACT_DIR}/wheels" .

# newest wheel, not a glob: older builds linger in this directory
_whl="$(ls -1t "${DSV4_ARTIFACT_DIR}"/wheels/kt_kernel-*.whl | head -1)"
[ -n "${_whl}" ] || die "no kt_kernel wheel produced"
log "installing ${_whl}"
"${DSV4_PYTHON}" -m pip install --no-deps --force-reinstall "${_whl}"

"${DSV4_PYTHON}" - <<'PY' || die "the installed kt_kernel is not an Ascend build; see the configure output above"
import sys
from kt_kernel import kt_kernel_ext
from kt_kernel.utils.loader import GGMLQuantizationType
ok = True
if not hasattr(kt_kernel_ext, "init_ascend_callback_worker"):
    print("  FAIL no init_ascend_callback_worker - built without CPUINFER_USE_ASCEND_NPU=1")
    ok = False
if int(GGMLQuantizationType.MXFP4) != 39:
    print(f"  FAIL GGMLQuantizationType.MXFP4 == {int(GGMLQuantizationType.MXFP4)}, expected 39")
    ok = False
print("  ascend backend and MXFP4 present" if ok else "")
sys.exit(0 if ok else 1)
PY
}


step_sgl_kernel() {
if [ "${DSV4_FORCE_SGL_KERNEL:-0}" != "1" ] \
   && "${DSV4_PYTHON}" -c 'import sgl_kernel_npu, deep_ep, attentions' >/dev/null 2>&1; then
  log "already provided by this image; skipping (DSV4_FORCE_SGL_KERNEL=1 to build anyway)"
  return 0
fi
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

_cm="csrc/attentions/csrc/CMakeLists.txt"
if [ -f "${_cm}" ] && ! grep -q 'CMAKE_DL_LIBS' "${_cm}"; then
  log "patching ${_cm} to link \${CMAKE_DL_LIBS}"
  sed -i 's/\(target_link_libraries(PTAExtensionOPS[^\n]*\)/\1 ${CMAKE_DL_LIBS}/' "${_cm}"
  grep -q 'CMAKE_DL_LIBS' "${_cm}" || log "WARNING: automatic -ldl patch did not apply; add it by hand"
fi

chmod -R u+w python/deep_ep/deep_ep/vendors 2>/dev/null || true
rm -rf python/deep_ep/deep_ep/vendors/hwcomputing 2>/dev/null || true

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
}


step_cann_ops() {
if [ "${1:-all}" = "all" ] && [ "${DSV4_FORCE_CANN_OPS:-0}" != "1" ] \
   && [ -d "${CANN_VENDORS_DIR}/customize" ] \
   && [ -d "${CANN_VENDORS_DIR}/custom_transformer" ] \
   && "${DSV4_PYTHON}" -c 'import torch, torch_npu, custom_ops' >/dev/null 2>&1; then
  log "already provided by this image; skipping (DSV4_FORCE_CANN_OPS=1 to build anyway)"
  return 0
fi
umask 0022

mkdir -p "${DSV4_ARTIFACT_DIR}/vendor_packages" "${DSV4_ARTIFACT_DIR}/wheels"

if [ -z "${DSV4_SOC}" ]; then
  DSV4_SOC="$(dsv4_detect_soc)"
fi
[ -n "${DSV4_SOC}" ] || die "cannot determine the SoC.
    npu-smi and torch_npu both failed to report a chip name. Set it by hand:
      export DSV4_SOC=ascend910b     # Atlas A2 / 910B series
      export DSV4_SOC=ascend910_93   # Atlas A3 / 910_93 series"
export DSV4_SOC
log "SoC: ${DSV4_SOC}"

clone_pinned() {
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "${dir}/.git" ]; then
    log "cloning ${url} -> ${dir}"
    git clone --progress "${url}" "${dir}"
  fi
  git -C "${dir}" fetch --quiet origin || true
  git -C "${dir}" checkout --quiet --detach "${commit}" \
    || die "cannot check out ${commit} in ${dir} (dirty tree? run: git -C ${dir} status)"
  log "${dir} @ $(git -C "${dir}" rev-parse --short HEAD)"
}

build_customize() {
  log "1/3 customize vendor (cann-recipes-infer @ ${CANN_RECIPES_COMMIT}, soc=${DSV4_SOC})"
  clone_pinned "${CANN_RECIPES_URL}" "${CANN_RECIPES_REPO}" "${CANN_RECIPES_COMMIT}"
  cd "${CANN_RECIPES_REPO}/ops/ascendc"
  chmod -R go-w .
  OPS_CPU_NUMBER="${DSV4_JOBS}" bash build.sh -c "${DSV4_SOC}"
  local run
  run="$(ls -1 output/CANN-custom_ops-*-linux*.run 2>/dev/null | head -1)" \
    || die "no .run produced; see the build log above"
  [ -n "${run}" ] || die "no .run produced; see the build log above"
  install -m 0755 "${run}" "${DSV4_ARTIFACT_DIR}/vendor_packages/customize.run"
  bash "${DSV4_ARTIFACT_DIR}/vendor_packages/customize.run" --quiet \
      --install-path="${CANN_ROOT}/opp"
  log "installed ${CANN_VENDORS_DIR}/customize"
}

build_custom_ops() {
  log "2/3 custom_ops torch bindings (cann-recipes-infer)"
  [ -d "${CANN_RECIPES_REPO}" ] || die "run the 'customize' step first (it clones the repo)"
  cd "${CANN_RECIPES_REPO}/ops/ascendc/torch_ops_extension"
  USE_NINJA=1 bash build_and_install.sh
  local whl
  whl="$(ls -1 dist/custom_ops-*-linux_*.whl 2>/dev/null | head -1)"
  [ -n "${whl}" ] || die "no custom_ops wheel produced"
  install -m 0644 "${whl}" "${DSV4_ARTIFACT_DIR}/wheels/"
  log "wheel -> ${DSV4_ARTIFACT_DIR}/wheels/$(basename "${whl}")"
}

build_transformer() {
  log "3/3 custom_transformer vendor (ops-transformer @ ${OPS_TRANSFORMER_COMMIT}, soc=${DSV4_SOC})"
  clone_pinned "${OPS_TRANSFORMER_URL}" "${OPS_TRANSFORMER_REPO}" "${OPS_TRANSFORMER_COMMIT}"
  cd "${OPS_TRANSFORMER_REPO}"
  chmod -R go-w .
  bash build.sh --pkg --experimental --soc="${DSV4_SOC}" --vendor_name=custom \
    --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata,compressor,quant_lightning_indexer,quant_lightning_indexer_metadata \
    --cann_3rd_lib_path="${OPS_TRANSFORMER_REPO}/third_party" \
    -j"${DSV4_JOBS}"
  local run="build/cann-ops-transformer-custom_linux-$(uname -m).run"
  [ -f "${run}" ] || die "no .run produced at ${run}"
  install -m 0755 "${run}" "${DSV4_ARTIFACT_DIR}/vendor_packages/custom_transformer.run"
  bash "${DSV4_ARTIFACT_DIR}/vendor_packages/custom_transformer.run" --quiet \
      --install-path="${CANN_ROOT}/opp"
  log "installed ${CANN_VENDORS_DIR}/custom_transformer"
}

case "${1:-all}" in
  customize)   build_customize ;;
  custom_ops)  build_custom_ops ;;
  transformer) build_transformer ;;
  all)         build_customize; build_custom_ops; build_transformer ;;
  *) die "unknown step '${1}' (expected: all | customize | custom_ops | transformer)" ;;
esac

dsv4_export_vendor_paths
log "done."
}


step_gguf() {
TOOLS="${KTRANSFORMERS_REPO}/kt-kernel/tools/mxfp4_gguf"

[ -f "${DSV4_NATIVE_CKPT}/config.json" ] \
  || die "DSV4_NATIVE_CKPT=${DSV4_NATIVE_CKPT} has no config.json.
    Download the official checkpoint first, e.g.
      huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ${DSV4_NATIVE_CKPT}"

N_LAYERS="$("${DSV4_PYTHON}" -c \
  "import json;print(json.load(open('${DSV4_NATIVE_CKPT}/config.json'))['num_hidden_layers'])")"
LAST=$(( N_LAYERS - 1 ))
mkdir -p "${DSV4_GGUF_DIR}"

export KT_GGUF_PY="${KT_GGUF_PY:-${KTRANSFORMERS_REPO}/third_party/llama.cpp/gguf-py}"
[ -d "${KT_GGUF_PY}" ] || die "no gguf-py at ${KT_GGUF_PY}.
    git -C ${KTRANSFORMERS_REPO} submodule update --init --progress third_party/llama.cpp"
"${DSV4_PYTHON}" -c "import sys;sys.path.insert(0,'${KT_GGUF_PY}');import gguf;assert int(gguf.GGMLQuantizationType.MXFP4)==39" 2>/dev/null \
  || die "gguf-py at ${KT_GGUF_PY} does not know GGML_TYPE_MXFP4.
    The patch series is applied by kt-kernel's CMake configure step; build
    kt-kernel once (setup.sh kt-kernel) and re-run this."

if [ "${1:-convert}" != "verify" ]; then
  log "converting layers 0..${LAST} from ${DSV4_NATIVE_CKPT}"
  log "output ${DSV4_GGUF_DIR} (needs ~$(( N_LAYERS * 32 / 10 )) GiB free)"
  "${DSV4_PYTHON}" "${TOOLS}/convert_mxfp4_gguf.py" batch \
      --input       "${DSV4_NATIVE_CKPT}" \
      --output-dir  "${DSV4_GGUF_DIR}" \
      --layer-start 0 --layer-end "${LAST}" \
      --jobs        "${DSV4_JOBS}" \
      --skip-existing
fi

log "verifying (L1 count+size, L2 sha256 self-manifest, L3 bit-exact sample)"
"${DSV4_PYTHON}" "${TOOLS}/verify_mxfp4_gguf.py" set \
    --dir "${DSV4_GGUF_DIR}" \
    --expect-layers "${N_LAYERS}" \
    --deep 3 --model-dir "${DSV4_NATIVE_CKPT}"

log "done. ${DSV4_GGUF_DIR} holds $(ls -1 "${DSV4_GGUF_DIR}"/dsv4_layer*_mxfp4.gguf | wc -l) files."
log "Point the server at them with:"
log "  export DSV4_GGUF_TEMPLATE='${DSV4_GGUF_DIR}/dsv4_layer{layer_idx}_mxfp4.gguf'"
log "  (single quotes are required — in double quotes bash eats the first '}')"
}


step_probe() {
have() { printf '  \033[32mhave\033[0m   %s\n' "$*"; }
need() { printf '  \033[33mbuild\033[0m  %s\n' "$*"; }
sec()  { printf '\n%s\n' "$*"; }

_imp() { "${DSV4_PYTHON}" -c "import $1" >/dev/null 2>&1; }

_SGL_MISS=""
for m in sgl_kernel_npu deep_ep attentions; do
  _imp "${m}" || _SGL_MISS="${_SGL_MISS} ${m}"
done
_SGL_OPT=""
_imp torch_memory_saver || _SGL_OPT=" torch_memory_saver (only needed for --enable-memory-saver)"

_OPS_MISS=""
"${DSV4_PYTHON}" -c 'import torch, torch_npu, custom_ops' >/dev/null 2>&1 \
  || _OPS_MISS="${_OPS_MISS} custom_ops"
[ -d "${CANN_VENDORS_DIR}/customize" ]          || _OPS_MISS="${_OPS_MISS} vendors/customize"
[ -d "${CANN_VENDORS_DIR}/custom_transformer" ] || _OPS_MISS="${_OPS_MISS} vendors/custom_transformer"

sec "Image"
have "CANN ${CANN_ROOT##*/} at ${CANN_ROOT}"
have "python $("${DSV4_PYTHON}" -V 2>&1 | awk '{print $2}'), torch $("${DSV4_PYTHON}" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '?')"

sec "SGLang NPU kernels           (setup.sh sgl-kernel)"
if [ -z "${_SGL_MISS}" ]; then have "sgl_kernel_npu deep_ep attentions"
else need "missing:${_SGL_MISS}"; fi
if [ -n "${_SGL_OPT}" ]; then printf '  \033[2mskip\033[0m  not installed:%s\n' "${_SGL_OPT}"; fi

sec "AscendC custom operators     (setup.sh cann-ops)"
if [ -z "${_OPS_MISS}" ]; then have "custom_ops, ${CANN_VENDORS_DIR}/{customize,custom_transformer}"
else need "missing:${_OPS_MISS}"; fi

sec "KT-Kernel                    (setup.sh kt-kernel)"
if "${DSV4_PYTHON}" -c 'from kt_kernel import kt_kernel_ext; import sys; sys.exit(0 if hasattr(kt_kernel_ext,"init_ascend_callback_worker") else 1)' >/dev/null 2>&1
then have "kt_kernel with the Ascend backend"
else need "no Ascend-enabled kt_kernel — always built from this repo"; fi

sec "MXFP4 GGUF experts           (setup.sh gguf)"
_n=$(ls -1 "${DSV4_GGUF_DIR}"/dsv4_layer*_mxfp4.gguf 2>/dev/null | wc -l)
if [ "${_n}" -gt 0 ]; then have "${_n} files in ${DSV4_GGUF_DIR}"
else need "none in ${DSV4_GGUF_DIR}"; fi

sec "Verdict"
if [ -z "${_SGL_MISS}" ] && [ -z "${_OPS_MISS}" ]; then
  printf '  This image already ships the operator stack. "setup.sh all" will skip\n'
  printf '  sgl-kernel and cann-ops, leaving kt-kernel + gguf + check.\n'
else
  printf '  This image does not ship the full operator stack. "setup.sh all" will\n'
  printf '  build what is missing.\n'
fi
}

step_check() {
FAIL=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=1; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$*"; }
sec()  { printf '\n%s\n' "$*"; }

sec "CANN"
if [ -f "${ASCEND_INSTALL_ROOT}/ascend-toolkit/set_env.sh" ]; then
  ok "toolkit ${CANN_VERSION} at ${ASCEND_INSTALL_ROOT}"
else
  bad "no toolkit at ${ASCEND_INSTALL_ROOT}"
fi
for v in customize custom_transformer; do
  if [ -d "${CANN_VENDORS_DIR}/${v}" ]; then
    ok "vendor ${v}"
  else
    bad "vendor ${v} missing under ${CANN_VENDORS_DIR} — run: setup.sh cann-ops"
  fi
done
case ":${ASCEND_CUSTOM_OPP_PATH}:" in
  *":${CANN_VENDORS_DIR}/customize:"*) ok "ASCEND_CUSTOM_OPP_PATH includes the vendors" ;;
  *) bad "ASCEND_CUSTOM_OPP_PATH does not include ${CANN_VENDORS_DIR}/customize" ;;
esac

sec "Python stack"
"${DSV4_PYTHON}" - <<'PY'
import importlib, sys

failed = 0

def probe(label, fn):
    global failed
    try:
        print(f"  ok   {label}: {fn()}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
        failed = 1
        return False

torch = None
if probe("torch", lambda: importlib.import_module("torch").__version__):
    torch = importlib.import_module("torch")
probe("torch_npu", lambda: importlib.import_module("torch_npu").__version__)
if torch is not None:
    probe("npu available",
          lambda: f"{torch.npu.is_available()} devices={torch.npu.device_count()}")
probe("transformers", lambda: importlib.import_module("transformers").__version__)

for _companion in ("torchvision", "torchaudio"):
    try:
        importlib.import_module(_companion)
    except ModuleNotFoundError:
        pass  # genuinely absent is fine; nothing here requires it
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {_companion}: {type(exc).__name__}: {exc}")
        if torch is not None:
            base = torch.__version__.split("+")[0]
            print(f"       This build does not match torch {torch.__version__}. Install the "
                  f"matching release:\n"
                  f"         pip install --no-deps '{_companion}=={base}'")
        failed = 1

if probe("custom_ops", lambda: (importlib.import_module("custom_ops"), "imported")[1]) and torch is not None:
    missing = [op for op in ("compressor", "npu_sparse_attn_sharedkv",
                             "npu_quant_lightning_indexer", "npu_moe_gating_top_k")
               if not hasattr(torch.ops.custom, op)]
    if missing:
        print(f"  FAIL torch.ops.custom.*: missing {missing} — "
              "the operator vendor packages are not loaded")
        failed = 1
    else:
        print("  ok   torch.ops.custom.* all present")

sys.exit(failed)
PY
[ $? -ne 0 ] && FAIL=1

sec "kt-kernel"
"${DSV4_PYTHON}" - <<'PY'
import sys
try:
    from kt_kernel import kt_kernel_ext
    from kt_kernel.utils.loader import GGMLQuantizationType
except Exception as exc:  # noqa: BLE001
    print(f"  FAIL import kt_kernel: {type(exc).__name__}: {exc}")
    print("       Build it with: setup.sh kt-kernel")
    sys.exit(1)
print(f"  ok   kt_kernel_ext at {kt_kernel_ext.__file__}")
if not hasattr(kt_kernel_ext, "init_ascend_callback_worker"):
    print("  FAIL kt_kernel_ext has no init_ascend_callback_worker — "
          "it was built without CPUINFER_USE_ASCEND_NPU=1")
    sys.exit(1)
print("  ok   Ascend callback worker binding present")
if int(GGMLQuantizationType.MXFP4) != 39:
    print(f"  FAIL GGMLQuantizationType.MXFP4 == {int(GGMLQuantizationType.MXFP4)}, expected 39")
    sys.exit(1)
print("  ok   GGMLQuantizationType.MXFP4 == 39")
PY
[ $? -ne 0 ] && FAIL=1

sec "sgl-kernel-npu"
if "${DSV4_PYTHON}" -c 'import sgl_kernel_npu' >/dev/null 2>&1; then
  ok "sgl_kernel_npu importable"
else
  bad "sgl_kernel_npu is missing. SGLang imports it unconditionally from
       srt/mem_cache/pool_host/mha.py, so the server dies at startup.
       Container images ship it; on a native install build it:
         bash ${_here}/setup.sh sgl-kernel"
fi

sec "SGLang"
_sglang_err="$(mktemp)"
_sglang_out="$("${DSV4_PYTHON}" -c 'import sglang; print(sglang.__file__)' 2>"${_sglang_err}" | tail -1)"
_sglang_rc=${PIPESTATUS[0]}
if [ "${_sglang_rc}" -ne 0 ]; then
  bad "importing sglang failed:
$(tail -3 "${_sglang_err}" | sed 's/^/       /')
       If this is a missing module, install SGLang's NPU runtime dependencies:
         bash ${_here}/setup.sh deps
       Do NOT run 'pip install -e python/' from the clone on a CANN image: the
       default pyproject.toml is the CUDA variant and it replaces torch."
else
  case "${_sglang_out}" in
    "${SGLANG_REPO}"/python/sglang/__init__.py) ok "sglang resolves to the clone: ${_sglang_out}" ;;
    *) bad "sglang resolves to ${_sglang_out}
       expected ${SGLANG_REPO}/python/sglang/__init__.py
       PYTHONPATH was shadowed. Export it after every environment script, not before." ;;
  esac
fi
rm -f "${_sglang_err}"

sec "Weights"
if [ -f "${DSV4_MODEL_PATH}/config.json" ]; then
  _fmt="$("${DSV4_PYTHON}" -c "import json,sys;c=json.load(open(sys.argv[1]));q=c.get('quantization_config') or {};print(q.get('quant_method') or q.get('format') or 'unknown')" "${DSV4_MODEL_PATH}/config.json" 2>/dev/null)"
  ok "W8A8 checkpoint ${DSV4_MODEL_PATH} (quantization: ${_fmt})"
  [ "${_fmt}" = "compressed-tensors" ] || warn "expected quant_method=compressed-tensors, got '${_fmt}'"
else
  bad "no config.json under DSV4_MODEL_PATH=${DSV4_MODEL_PATH}"
fi

_n_gguf="$(ls -1 "${DSV4_GGUF_DIR}"/dsv4_layer*_mxfp4.gguf 2>/dev/null | wc -l | tr -d ' ')"
_n_layers="$("${DSV4_PYTHON}" -c "import json,sys;print(json.load(open(sys.argv[1]))['num_hidden_layers'])" "${DSV4_MODEL_PATH}/config.json" 2>/dev/null || echo 43)"
if [ "${_n_gguf}" = "${_n_layers}" ]; then
  ok "${_n_gguf} per-layer MXFP4 GGUF files in ${DSV4_GGUF_DIR}"
else
  bad "found ${_n_gguf} GGUF files in ${DSV4_GGUF_DIR}, expected ${_n_layers} — run: setup.sh gguf"
fi
case "${DSV4_GGUF_TEMPLATE}" in
  *"{layer_idx}"*) ok "DSV4_GGUF_TEMPLATE contains the {layer_idx} placeholder" ;;
  *) bad "DSV4_GGUF_TEMPLATE has no {layer_idx} placeholder: ${DSV4_GGUF_TEMPLATE}
       A path without it is used literally for every layer." ;;
esac

sec "Host resources"
ok "NUMA nodes=${DSV4_THREADPOOL_COUNT} -> --kt-threadpool-count ${DSV4_THREADPOOL_COUNT}, --kt-cpuinfer ${DSV4_CPUINFER}"
_ram_gb="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)"
if [ "${_ram_gb}" -ge 200 ] 2>/dev/null; then
  ok "host RAM ${_ram_gb} GB"
else
  warn "host RAM ${_ram_gb} GB — the offloaded experts alone need about 140 GB resident"
fi
_free_gb="$(df -BG --output=avail "${DSV4_GGUF_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9')"
[ -n "${_free_gb}" ] && ok "free space on the GGUF filesystem: ${_free_gb} GB"

sec "Serving parameters"
if [ $(( DSV4_CHUNKED_PREFILL_SIZE % 128 )) -eq 0 ] && [ "${DSV4_CHUNKED_PREFILL_SIZE}" -gt 0 ]; then
  ok "--chunked-prefill-size ${DSV4_CHUNKED_PREFILL_SIZE} is a positive multiple of --page-size 128"
else
  bad "--chunked-prefill-size must be a positive multiple of 128; got ${DSV4_CHUNKED_PREFILL_SIZE}"
fi
case ":${http_proxy:-}:${https_proxy:-}:${all_proxy:-}:" in
  ::::) ok "no HTTP proxy set" ;;
  *) bad "an HTTP proxy is set; the startup warmup will fail with 502" ;;
esac

printf '\n'
if [ "${FAIL}" -eq 0 ]; then
  printf '\033[32mPREFLIGHT OK\033[0m\n'
else
  printf '\033[31mPREFLIGHT FAILED\033[0m — fix the FAIL lines above before launching.\n'
fi
exit "${FAIL}"
}


# --- dispatch ---------------------------------------------------------------
usage() {
  sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^#$//' -e 's/^# //'
  exit "${1:-0}"
}

_cmd="${1:-}"
[ $# -gt 0 ] && shift

case "${_cmd}" in
  probe)       _STEP="probe";       step_probe "$@" ;;
  deps)        _STEP="deps";        step_deps "$@" ;;
  kt-kernel)   _STEP="kt-kernel";   step_kt_kernel "$@" ;;
  sgl-kernel)  _STEP="sgl-kernel";  step_sgl_kernel "$@" ;;
  cann-ops)    _STEP="cann-ops";    step_cann_ops "$@" ;;
  gguf)        _STEP="gguf";        step_gguf "$@" ;;
  # `check` is the only step that must survive its own failures: it is a report,
  # and one failing probe must not hide the rest. Run it with -e off and pass its
  # exit status through.
  check)       _STEP="check";       set +e; step_check "$@"; exit $? ;;
  all)
    _STEP="deps";       step_deps
    _STEP="kt-kernel";  step_kt_kernel
    _STEP="sgl-kernel"; step_sgl_kernel
    _STEP="cann-ops";   step_cann_ops
    _STEP="gguf";       step_gguf
    _STEP="check";      set +e; step_check; _rc=$?
    [ "${_rc}" -eq 0 ] || die "setup finished but 'check' reported problems (see above)"
    ;;
  ""|-h|--help|help) usage 0 ;;
  *) printf 'unknown step: %s\n\n' "${_cmd}" >&2; usage 1 ;;
esac
