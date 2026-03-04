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
  --skip-sglang       Skip sglang installation (for "all" subcommand)
  --skip-kt-kernel    Skip kt-kernel installation (for "all" subcommand)
  --editable          Install sglang in editable/dev mode (-e)
  --manual            Pass through to kt-kernel (manual CPU config)
  --no-clean          Pass through to kt-kernel (skip build clean)

EXAMPLES:
  # Full install (recommended)
  $0

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

  if [ ! -d "$REPO_ROOT/.git" ]; then
    log_warn "Not a git repository. Skipping submodule init."
    log_warn "If you need sglang, clone with: git clone --recursive https://github.com/kvcache-ai/ktransformers.git"
    return 0
  fi

  cd "$REPO_ROOT"
  git submodule update --init --recursive
  log_info "Submodules initialized successfully."
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

  cd "$sglang_dir"

  if [ "$editable" = "1" ]; then
    log_info "Installing sglang in editable mode..."
    pip install -e "./python[all]"
  else
    log_info "Installing sglang..."
    pip install "./python[all]"
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
  bash ./install.sh build "${kt_args[@]}"
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
