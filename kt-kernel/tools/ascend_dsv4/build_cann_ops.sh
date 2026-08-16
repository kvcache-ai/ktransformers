#!/usr/bin/env bash
# =============================================================================
# Build and install the three CANN custom-operator packages DeepSeek-V4-Flash
# needs on Ascend:
#
#   1. `customize`          vendor  — fused ops (RmsNormDynamicQuant, SwigluClipQuant,
#                                     MoeGatingTopKHash, ...)         [cann-recipes-infer]
#   2. `custom_ops`         wheel   — the torch.ops.custom.* bindings [cann-recipes-infer]
#   3. `custom_transformer` vendor  — the NSA/DSA attention ops       [ops-transformer]
#
# Skip this entirely if ${CANN_VENDORS_DIR} already contains `customize` and
# `custom_transformer` and `python -c "import custom_ops"` works: vendor images
# often ship them. Building all three takes roughly 40-90 minutes.
#
# Usage:
#   bash kt-kernel/tools/ascend_dsv4/build_cann_ops.sh [all|customize|custom_ops|transformer]
# =============================================================================
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

log() { printf '\n[build_cann_ops] %s\n' "$*"; }
die() { printf '[build_cann_ops] FATAL: %s\n' "$*" >&2; exit 1; }

# CANN's msopgen refuses to work on files that are group- or world-writable and
# aborts the whole build with a "security risks" message. Every build step
# below runs under this umask and chmods its source tree.
umask 0022

mkdir -p "${DSV4_ARTIFACT_DIR}/vendor_packages" "${DSV4_ARTIFACT_DIR}/wheels"

# The SoC name decides which kernel binaries get built. A wrong value produces
# packages that install cleanly and then fail at the first kernel launch, so
# refuse to guess.
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
  # The NSA/DSA operators exist only on `master`; the 9.0.0 release branch
  # dropped them. `--vendor_name=custom` is correct: the build appends
  # "_transformer" itself, producing the vendor name `custom_transformer`.
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

log "done. Re-source dsv4_env.sh so ASCEND_CUSTOM_OPP_PATH picks up the new vendors."
