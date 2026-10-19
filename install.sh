#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $0 [SUBCOMMAND] [OPTIONS]

One-click installer for ktransformers (sglang + kt-kernel).

SUBCOMMANDS:
  all             Full install: submodules → sglang → kt-kernel (default)
  sglang          Install sglang only
  kt-kernel       Install kt-kernel only
  deps            Install system dependencies only
  -h, --help      Show this help message

OPTIONS:
  --sycl             Build Intel SYCL experts with the tested local configuration
  --skip-sglang       Skip sglang installation (for "all" subcommand)
  --skip-kt-kernel    Skip kt-kernel installation (for "all" subcommand)
  --editable          Install sglang in editable/dev mode (-e)
  --manual            Pass through to kt-kernel (manual CPU config)
  --no-clean          Pass through to kt-kernel (skip build clean)

EXAMPLES:
  # Full install (recommended)
  $0

  # Intel iGPU SYCL source install
  $0 --sycl

  # Install everything in editable mode for development
  $0 all --editable

  # Install sglang only
  $0 sglang

  # Install kt-kernel only (manual CPU config)
  $0 kt-kernel --manual

  # Full install, skip sglang (already installed)
  $0 all --skip-sglang

EOF
  exit 1
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

log_step() {
  echo ""
  echo "=========================================="
  echo "  $1"
  echo "=========================================="
  echo ""
}

log_info() {
  echo "[INFO] $1"
}

log_warn() {
  echo "[WARN] $1"
}

log_error() {
  echo "[ERROR] $1" >&2
}

# Read ktransformers version from version.py and export for sglang-kt
read_kt_version() {
  local version_file="$REPO_ROOT/version.py"
  if [ -f "$version_file" ]; then
    KT_VERSION=$(python3 -c "exec(open('$version_file').read()); print(__version__)")
    export SGLANG_KT_VERSION="$KT_VERSION"
    log_info "ktransformers version: $KT_VERSION (will be used for sglang-kt)"
  else
    log_warn "version.py not found; sglang-kt will use its default version"
  fi
}

# ─── Submodule init ────────────────────────────────────────────────────────────

init_submodules() {
  log_step "Initializing git submodules"

  if [ ! -e "$REPO_ROOT/.git" ]; then
    log_warn "Not a git repository. Skipping submodule init."
    log_warn "If you need sglang, clone with: git clone --recursive https://github.com/kvcache-ai/ktransformers.git"
    return 0
  fi

  cd "$REPO_ROOT"
  git submodule update --init --recursive
  log_info "Submodules initialized successfully."
}

# Configure the optional Intel build once; the ordinary install path is unchanged.
configure_sycl() {
  local name
  for name in CPUINFER_USE_CUDA CPUINFER_USE_ROCM CPUINFER_USE_MUSA CPUINFER_USE_MACA CPUINFER_USE_ASCEND_NPU; do
    case "${!name:-0}" in
      1|ON|on|true|TRUE|yes|YES)
        log_error "--sycl conflicts with $name=${!name}; unset it before installing."
        exit 1
        ;;
    esac
  done
  local setvars="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
  if [ -r "$setvars" ]; then
    # Vendor environment scripts are not guaranteed to support errexit/nounset.
    # Validate the compiler after initialization instead of treating their final
    # shell status as proof that the toolchain was initialized.
    set +eu
    source "$setvars" --force
    set -eu
  elif ! command -v icpx >/dev/null 2>&1; then
    log_error "Install oneAPI DPC++ or set ONEAPI_SETVARS to its setvars.sh."
    exit 1
  fi
  command -v icpx >/dev/null 2>&1 || { log_error "oneAPI initialization did not provide icpx"; exit 1; }
  export CPUINFER_USE_SYCL=1 CPUINFER_USE_CUDA=0
  export CPUINFER_CPU_INSTRUCT="${CPUINFER_CPU_INSTRUCT:-AVX2}"
  export CPUINFER_ENABLE_AMX="${CPUINFER_ENABLE_AMX:-OFF}"
  export CPUINFER_ENABLE_AVX512="${CPUINFER_ENABLE_AVX512:-OFF}"
  export CPUINFER_ENABLE_MLA="${CPUINFER_ENABLE_MLA:-OFF}"
  export CPUINFER_ENABLE_CPPTRACE="${CPUINFER_ENABLE_CPPTRACE:-OFF}"
  export CPUINFER_ENABLE_LTO="${CPUINFER_ENABLE_LTO:-OFF}"
  export CPUINFER_LLAMA_BUILD_COPY="${CPUINFER_LLAMA_BUILD_COPY:-ON}"
  export CPUINFER_PARALLEL="${CPUINFER_PARALLEL:-2}"
}

# ─── sglang install ───────────────────────────────────────────────────────────

install_sglang() {
  local editable="${1:-0}"

  log_step "Installing sglang (kvcache-ai fork)"

  local sglang_dir="$REPO_ROOT/third_party/sglang"
  local pyproject="$sglang_dir/python/pyproject.toml"

  if [ ! -f "$pyproject" ]; then
    log_error "sglang source not found at $sglang_dir"
    log_error "Run 'git submodule update --init --recursive' first, or clone with --recursive."
    exit 1
  fi

  local install_target="./python[all]"
  if [ "$SYCL_INSTALL" = "1" ]; then
    # Build the local kernel first, then resolve SGLang against that exact source
    # version. Never point users at a private submodule commit or replace the
    # checked-out SGLang source with generated packaging edits.
    python3 -c '
import os
from importlib.metadata import version
from kt_kernel import kt_kernel_ext as e

if e.moe.SYCLXMXGPTQInt4_MOE.implementation_revision() < 7:
    raise RuntimeError("Build kt-kernel first with ./install.sh --sycl")
if version("kt-kernel") != os.environ["SGLANG_KT_VERSION"]:
    raise RuntimeError("Installed kt-kernel version does not match this checkout; rebuild with ./install.sh kt-kernel --sycl")
'
    mkdir -p "$REPO_ROOT/build"
    local prepared
    prepared=$(mktemp -d "$REPO_ROOT/build/sglang-sycl.XXXXXX")
    python3 "$REPO_ROOT/kt-kernel/scripts/prepare_sglang.py" --output "$prepared/source"
    sglang_dir="$prepared/source"
    # Routed-expert inference needs the base runtime; diffusion/tracing extras
    # remain part of the ordinary all-platform install.
    install_target="./python"
  fi

  cd "$sglang_dir"

  if [ "$editable" = "1" ]; then
    log_info "Installing sglang in editable mode..."
    python3 -m pip install -e "$install_target"
  else
    log_info "Installing sglang..."
    python3 -m pip install "$install_target"
  fi

  log_info "sglang installed successfully."
}

# ─── kt-kernel install ────────────────────────────────────────────────────────

install_kt_kernel() {
  # Forward all remaining args to kt-kernel/install.sh
  local kt_args=("$@")

  log_step "Installing kt-kernel"

  local kt_install="$REPO_ROOT/kt-kernel/install.sh"

  if [ ! -f "$kt_install" ]; then
    log_error "kt-kernel/install.sh not found at $kt_install"
    exit 1
  fi

  cd "$REPO_ROOT/kt-kernel"
  if [ "$SYCL_INSTALL" = "1" ]; then
    kt_args+=("--manual")
  fi
  bash ./install.sh build "${kt_args[@]}"
  if [ "$SYCL_INSTALL" = "1" ]; then
    log_info "Checking installed SYCL extension without LD_LIBRARY_PATH"
    env -u LD_LIBRARY_PATH -u LD_PRELOAD python3 -c 'from kt_kernel import kt_kernel_ext as e; assert e.moe.SYCLXMXGPTQInt4_MOE.implementation_revision() >= 7, "Installed XMX extension is too old"; print("SYCL runtime library lookup passed")'
  fi
}

# ─── deps install ─────────────────────────────────────────────────────────────

install_deps() {
  log_step "Installing system dependencies"

  local kt_install="$REPO_ROOT/kt-kernel/install.sh"

  if [ ! -f "$kt_install" ]; then
    log_error "kt-kernel/install.sh not found at $kt_install"
    exit 1
  fi

  cd "$REPO_ROOT/kt-kernel"
  bash ./install.sh deps
}

# ─── "all" subcommand ─────────────────────────────────────────────────────────

install_all() {
  local skip_sglang=0
  local skip_kt_kernel=0
  local editable=0
  local kt_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-sglang)    skip_sglang=1; shift ;;
      --skip-kt-kernel) skip_kt_kernel=1; shift ;;
      --editable)       editable=1; shift ;;
      --manual)         kt_args+=("--manual"); shift ;;
      --no-clean)       kt_args+=("--no-clean"); shift ;;
      -h|--help)        usage ;;
      *)
        log_error "Unknown option: $1"
        usage
        ;;
    esac
  done

  # 1. Init submodules
  init_submodules

  # 2. System dependencies
  install_deps

  # 3. Read version for sglang-kt
  read_kt_version

  if [ "$SYCL_INSTALL" = "1" ]; then
    # The source-aligned SGLang metadata depends on this kernel version.
    if [ "$skip_kt_kernel" = "0" ]; then install_kt_kernel "${kt_args[@]}"; fi
    if [ "$skip_sglang" = "0" ]; then install_sglang "$editable"; fi
    log_step "SYCL source installation complete!"
    return
  fi

  # 4. Install sglang
  if [ "$skip_sglang" = "0" ]; then
    install_sglang "$editable"
  else
    log_info "Skipping sglang installation (--skip-sglang)."
  fi

  # 4. Build & install kt-kernel
  if [ "$skip_kt_kernel" = "0" ]; then
    install_kt_kernel "${kt_args[@]}"
  else
    log_info "Skipping kt-kernel installation (--skip-kt-kernel)."
  fi

  log_step "Installation complete!"
  echo "  Verify with: kt doctor"
  echo ""
}

# ─── Subcommand dispatcher ────────────────────────────────────────────────────

SYCL_INSTALL=0
case "${CPUINFER_USE_SYCL:-0}" in 1|ON|on|true|TRUE|yes|YES) SYCL_INSTALL=1 ;; esac
INSTALL_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--sycl" ]; then SYCL_INSTALL=1; else INSTALL_ARGS+=("$arg"); fi
done
set -- "${INSTALL_ARGS[@]}"
SUBCMD="all"
if [[ $# -gt 0 ]]; then
  case "$1" in
    all|sglang|kt-kernel|deps)
      SUBCMD="$1"
      shift
      ;;
    -h|--help)
      usage
      ;;
    -*)
      # Flags without subcommand → default to "all"
      SUBCMD="all"
      ;;
    *)
      log_error "Unknown subcommand: $1"
      usage
      ;;
  esac
fi

if [ "$SYCL_INSTALL" = "1" ]; then configure_sycl; fi

case "$SUBCMD" in
  all)
    install_all "$@"
    ;;
  sglang)
    # Parse sglang-specific options
    editable=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --editable) editable=1; shift ;;
        -h|--help) usage ;;
        *) log_error "Unknown option for sglang: $1"; usage ;;
      esac
    done
    init_submodules
    read_kt_version
    install_sglang "$editable"
    ;;
  kt-kernel)
    install_kt_kernel "$@"
    ;;
  deps)
    install_deps
    ;;
esac
