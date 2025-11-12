#!/usr/bin/env bash
set -euo pipefail

install_dependencies() {
  echo "Checking and installing system dependencies..."

  # Determine if we need to use sudo
  SUDO=""
  if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
      SUDO="sudo"
    else
      echo "Warning: Not running as root and sudo not found. Package installation may fail."
      echo "Please run as root or install sudo."
    fi
  fi

  if command -v conda &> /dev/null; then
    echo "Installing cmake via conda..."
    conda install -y cmake
  else
    echo "Warning: conda not found. Skipping cmake installation via conda."
    echo "Please install conda or manually install cmake."
  fi

  # Detect OS type
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
  elif [ -f /etc/debian_version ]; then
    OS="debian"
  elif [ -f /etc/redhat-release ]; then
    OS="rhel"
  else
    echo "Warning: Unable to detect OS type. Skipping dependency installation."
    return 0
  fi

  # Install dependencies based on OS
  case "$OS" in
    debian|ubuntu|linuxmint|pop)
      echo "Detected Debian-based system. Installing libhwloc-dev and pkg-config..."
      $SUDO apt update
      $SUDO apt install -y libhwloc-dev pkg-config
      ;;
    fedora|rhel|centos|rocky|almalinux)
      echo "Detected Red Hat-based system. Installing hwloc-devel and pkgconfig..."
      $SUDO dnf install -y hwloc-devel pkgconfig || $SUDO yum install -y hwloc-devel pkgconfig
      ;;
    arch|manjaro)
      echo "Detected Arch-based system. Installing hwloc and pkgconf..."
      $SUDO pacman -S --noconfirm hwloc pkgconf
      ;;
    opensuse*|sles)
      echo "Detected openSUSE-based system. Installing hwloc-devel and pkg-config..."
      $SUDO zypper install -y hwloc-devel pkg-config
      ;;
    *)
      echo "Warning: Unsupported OS '$OS'. Please manually install libhwloc-dev and pkg-config."
      ;;
  esac
}

install_dependencies

usage() {
  cat <<EOF
Usage: $0 [SUBCOMMAND] [BUILD_OPTIONS]

Two-step installation in one file. Choose a subcommand:

SUBCOMMANDS:
  deps            Install system prerequisites only
  build           Build and install kt-kernel (no dependency install)
  all             Run deps then build (default when no subcommand)
  -h, --help      Show this help message

BUILD_OPTIONS (for "build" or "all"):
  (none)          Auto-detect CPU and configure automatically (recommended)
  --manual        Skip auto-detection, use manual configuration (see below)
  --skip-deps     Skip deps step even with subcommand "all"
  --no-clean      Do not delete local build/ before building (default cleans)

AUTO-DETECTION (Default):
  The script will automatically detect your CPU capabilities and configure:
  - If AMX instructions detected → NATIVE + AMX=ON
  - Otherwise                    → NATIVE + AMX=OFF

MANUAL CONFIGURATION:
  Use --manual flag and set these environment variables before running:

  CPUINFER_CPU_INSTRUCT   - CPU instruction set
                            Options: NATIVE, AVX512, AVX2
  CPUINFER_ENABLE_AMX     - Enable Intel AMX support
                            Options: ON, OFF

Manual configuration examples:

┌─────────────────────────────────────────────────────────────────────────┐
│ Configuration                    │ Use Case                             │
├──────────────────────────────────┼──────────────────────────────────────┤
│ NATIVE + AMX=ON                  │ Best performance on AMX CPUs         │
│ AVX512 + AMX=OFF                 │ AVX512 CPUs without AMX              │
│ AVX2 + AMX=OFF                   │ Older CPUs or maximum compatibility  │
└──────────────────────────────────┴──────────────────────────────────────┘

  Example manual build:
    export CPUINFER_CPU_INSTRUCT=AVX512
    export CPUINFER_ENABLE_AMX=OFF
    $0 --manual

Advanced option (for binary distribution):
  FANCY - AVX512 with full extensions for Ice Lake+/Zen 4+
          Use this when building pre-compiled binaries to distribute.

Optional variables (with defaults):
  CPUINFER_BUILD_TYPE=Release      Build type (Debug/RelWithDebInfo/Release)
  CPUINFER_PARALLEL=8              Number of parallel build jobs
  CPUINFER_VERBOSE=1               Verbose build output (0/1)

EOF
  exit 1
}

install_dependencies() {
  echo "Checking and installing system dependencies..."

  # Determine if we need to use sudo
  SUDO=""
  if [ "${EUID:-0}" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
      SUDO="sudo"
    else
      echo "Warning: Not running as root and sudo not found. Package installation may fail."
      echo "Please run as root or install sudo."
    fi
  fi

  if command -v conda &> /dev/null; then
    echo "Installing cmake via conda..."
    conda install -y cmake
  else
    echo "Warning: conda not found. Skipping cmake installation via conda."
    echo "Please install conda or manually install cmake."
  fi

  # Detect OS type
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
  elif [ -f /etc/debian_version ]; then
    OS="debian"
  elif [ -f /etc/redhat-release ]; then
    OS="rhel"
  else
    echo "Warning: Unable to detect OS type. Skipping dependency installation."
    return 0
  fi

  # Install dependencies based on OS
  case "$OS" in
    debian|ubuntu|linuxmint|pop)
      echo "Detected Debian-based system. Installing libhwloc-dev and pkg-config..."
      $SUDO apt update
      $SUDO apt install -y libhwloc-dev pkg-config
      ;;
    fedora|rhel|centos|rocky|almalinux)
      echo "Detected Red Hat-based system. Installing hwloc-devel and pkgconfig..."
      $SUDO dnf install -y hwloc-devel pkgconfig || $SUDO yum install -y hwloc-devel pkgconfig
      ;;
    arch|manjaro)
      echo "Detected Arch-based system. Installing hwloc and pkgconf..."
      $SUDO pacman -S --noconfirm hwloc pkgconf
      ;;
    opensuse*|sles)
      echo "Detected openSUSE-based system. Installing hwloc-devel and pkg-config..."
      $SUDO zypper install -y hwloc-devel pkg-config
      ;;
    *)
      echo "Warning: Unsupported OS '$OS'. Please manually install libhwloc-dev and pkg-config."
      ;;
  esac
}

# Function to detect CPU features
detect_cpu_features() {
  local has_amx=0

  if [ -f /proc/cpuinfo ]; then
    # Check for AMX support on Linux
    if grep -q "amx_tile\|amx_int8\|amx_bf16" /proc/cpuinfo; then
      has_amx=1
    fi
  elif [ "$(uname)" = "Darwin" ]; then
    # macOS doesn't have AMX (ARM or Intel without AMX)
    has_amx=0
  fi

  echo "$has_amx"
}

build_step() {
  # Parse build-only flags from arguments to this function
  local MANUAL_MODE=0
  local CLEAN_BUILD=1
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --manual) MANUAL_MODE=1; shift ;;
      --skip-deps) shift ;; # ignore here
      --no-clean) CLEAN_BUILD=0; shift ;;
      -h|--help) usage ;;
      *) break ;;
    esac
  done

  # Clean local build directory to ensure a fresh CMake/configure
  local REPO_ROOT
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "$CLEAN_BUILD" -eq 1 ]]; then
    if [[ -d "$REPO_ROOT/build" ]]; then
      echo "Cleaning previous build directory: $REPO_ROOT/build"
      rm -rf "$REPO_ROOT/build"
    fi
  else
    echo "Skipping clean of $REPO_ROOT/build (requested by --no-clean)"
  fi

  if [ "$MANUAL_MODE" = "0" ]; then
  # Auto-detection mode
  echo "=========================================="
  echo "Auto-detecting CPU capabilities..."
  echo "=========================================="
  echo ""

  HAS_AMX=$(detect_cpu_features)

  if [ "$HAS_AMX" = "1" ]; then
    echo "✓ AMX instructions detected"
    export CPUINFER_CPU_INSTRUCT=NATIVE
    export CPUINFER_ENABLE_AMX=ON
    echo "  Configuration: NATIVE + AMX=ON (best performance)"
    echo ""
    echo "  ⚠️  Note: If you plan to use LLAMAFILE backend, use manual mode:"
    echo "     export CPUINFER_CPU_INSTRUCT=AVX512(AVX2/FANCY)"
    echo "     export CPUINFER_ENABLE_AMX=OFF"
    echo "     ./install.sh --manual"
  else
    echo "ℹ AMX instructions not detected"
    export CPUINFER_CPU_INSTRUCT=NATIVE
    export CPUINFER_ENABLE_AMX=OFF
    echo "  Configuration: NATIVE + AMX=OFF"
  fi

  echo ""
  echo "To use manual configuration instead, run: $0 --manual"
  echo ""
  else
  # Manual mode - validate user configuration (no exports)
  if [ -z "$CPUINFER_CPU_INSTRUCT" ] || [ -z "$CPUINFER_ENABLE_AMX" ]; then
    echo "Error: Manual mode requires CPUINFER_CPU_INSTRUCT and CPUINFER_ENABLE_AMX to be set."
    echo ""
    usage
  fi

  # Validate CPUINFER_CPU_INSTRUCT
  case "$CPUINFER_CPU_INSTRUCT" in
    NATIVE|FANCY|AVX512|AVX2)
      ;;
    *)
      echo "Error: Invalid CPUINFER_CPU_INSTRUCT='$CPUINFER_CPU_INSTRUCT'"
      echo "Must be one of: NATIVE, FANCY, AVX512, AVX2"
      exit 1
      ;;
  esac

  # Validate CPUINFER_ENABLE_AMX
  case "$CPUINFER_ENABLE_AMX" in
    ON|OFF)
      ;;
    *)
      echo "Error: Invalid CPUINFER_ENABLE_AMX='$CPUINFER_ENABLE_AMX'"
      echo "Must be either: ON or OFF"
      exit 1
      ;;
  esac

  # Warn about problematic configuration
  if [ "$CPUINFER_CPU_INSTRUCT" = "NATIVE" ] && [ "$CPUINFER_ENABLE_AMX" = "OFF" ]; then
    HAS_AMX=$(detect_cpu_features)
    if [ "$HAS_AMX" = "1" ]; then
      echo "⚠️  WARNING: NATIVE + AMX=OFF on AMX-capable CPU may cause compilation issues!"
      echo "   Recommended: Use AVX512 or AVX2 instead of NATIVE when AMX=OFF"
      echo ""
      read -p "Continue anyway? (y/N) " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
    fi
  fi

# Close MANUAL_MODE conditional
  fi

# Set defaults for optional variables
export CPUINFER_BUILD_TYPE=${CPUINFER_BUILD_TYPE:-Release}
export CPUINFER_PARALLEL=${CPUINFER_PARALLEL:-8}
export CPUINFER_VERBOSE=${CPUINFER_VERBOSE:-1}

echo "Building kt-kernel with configuration:"
echo "  CPUINFER_CPU_INSTRUCT=$CPUINFER_CPU_INSTRUCT"
echo "  CPUINFER_ENABLE_AMX=$CPUINFER_ENABLE_AMX"
echo "  CPUINFER_BUILD_TYPE=$CPUINFER_BUILD_TYPE"
echo "  CPUINFER_PARALLEL=$CPUINFER_PARALLEL"
echo "  CPUINFER_VERBOSE=$CPUINFER_VERBOSE"
echo ""

pip install . -v
}

# Subcommand dispatcher: default to "all"
SUBCMD="all"
if [[ $# -gt 0 ]]; then
  case "$1" in
    deps|build|all) SUBCMD="$1"; shift ;;
    -h|--help) usage ;;
    *) SUBCMD="build" ;; # backward compatibility: flags-only => build
  esac
fi

case "$SUBCMD" in
  deps)
    install_dependencies
    ;;
  build)
    build_step "$@"
    ;;
  all)
    if [[ " ${*:-} " == *" --skip-deps "* ]]; then
      build_step "$@"
    else
      install_dependencies
      build_step "$@"
    fi
    ;;
esac
