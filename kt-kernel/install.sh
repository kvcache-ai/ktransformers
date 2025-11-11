#!/usr/bin/env bash
set -e

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
Usage: $0 [OPTIONS]

This script builds kt-kernel with optimal settings for your CPU.

OPTIONS:
  (none)          Auto-detect CPU and configure automatically (recommended)
  -h, --help      Show this help message
  --manual        Skip auto-detection, use manual configuration (see below)

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

# Check if user requested help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
fi

# Check if manual mode
MANUAL_MODE=0
if [ "$1" = "--manual" ]; then
  MANUAL_MODE=1
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


echo "Successfully built and installed kt-kernel! with configuration:"
echo "  CPUINFER_CPU_INSTRUCT=$CPUINFER_CPU_INSTRUCT"
echo "  CPUINFER_ENABLE_AMX=$CPUINFER_ENABLE_AMX"
echo "  CPUINFER_BUILD_TYPE=$CPUINFER_BUILD_TYPE"